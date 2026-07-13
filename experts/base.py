"""
experts/base.py
---------------
BaseExpert — abstract base class for all IntelliMoE domain experts.

SOLID principles applied
─────────────────────────
  S — Single Responsibility:
        BaseExpert handles model I/O, tokenisation, and generation.
        Each subclass is responsible for one domain's prompt and config.

  O — Open/Closed:
        New experts are added by subclassing BaseExpert — the base class
        itself never changes when a new domain is added.

  L — Liskov Substitution:
        All experts are interchangeable via the BaseExpert type. The router
        calls expert.answer(query) without knowing the concrete subclass.

  I — Interface Segregation:
        BaseExpert exposes exactly one public method (answer) and two
        abstract properties (prompt_name, generation_config). Subclasses
        are not forced to implement unneeded methods.

  D — Dependency Inversion:
        BaseExpert depends on the abstract PromptManager and the
        load_model_and_tokenizer() function — not on any concrete model class.

Before this refactor: 7 expert files × ~180 lines = ~1,260 lines of nearly
identical code. After: 1 BaseExpert (~120 lines) + 7 thin subclasses (~10
lines each) = ~190 lines total. >85% reduction in duplication.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

from config.settings import GenerationConfig
from models.loader import load_model_and_tokenizer, inference_lock
from utils.prompt_manager import PromptManager

if TYPE_CHECKING:
    from utils.memory import ConversationMemory

logger = logging.getLogger(__name__)


class BaseExpert(ABC):
    """
    Abstract base class for all IntelliMoE expert modules.

    Subclasses must implement two read-only properties:

    ``prompt_name`` → str
        The stem of the prompt file to load (e.g. ``"coding"`` → ``prompts/coding.txt``).

    ``generation_config`` → GenerationConfig
        The inference hyperparameters (temperature, top_p, etc.) for this domain.

    All infrastructure — lazy model loading, ChatML prompt formatting,
    tokenisation, and inference — is implemented once here and inherited
    by every subclass.
    """

    def __init__(self) -> None:
        # Lazy-loaded shared resources (populated on first call to answer()).
        self._tokenizer = None
        self._model     = None
        self._system_prompt: Optional[str] = None

        # PromptManager is injected here; can be overridden in tests.
        self._prompt_manager: PromptManager = PromptManager()

        # Monitoring metrics for evaluation dashboard
        self.last_prompt_tokens: int = 0
        self.last_tokens_generated: int = 0

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement these two properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def prompt_name(self) -> str:
        """
        Stem of the prompt file in ``prompts/``, without the ``.txt`` extension.

        Examples: ``"coding"``, ``"math"``, ``"systemdesign"``.
        """

    @property
    @abstractmethod
    def generation_config(self) -> GenerationConfig:
        """
        Inference hyperparameters specific to this expert's domain.

        Returns a ``GenerationConfig`` instance from ``config.settings``.
        """

    # ------------------------------------------------------------------
    # Public API (LSP: all experts expose the same interface)
    # ------------------------------------------------------------------

    def answer(self, question: str, memory: "Optional[ConversationMemory]" = None) -> str:
        """
        Generate a domain-expert answer for the given user question.

        Parameters
        ----------
        question : str
            The raw user question. Will be stripped of leading/trailing whitespace.
        memory : ConversationMemory | None
            Optional conversation history. When provided, previous turns are
            injected into the ChatML prompt so TinyLlama can maintain context.

        Returns
        -------
        str
            The model-generated answer. Only newly generated tokens are
            returned — the prompt is never echoed in the output.

        Raises
        ------
        ValueError
            If ``question`` is empty or whitespace-only.
        RuntimeError
            If model loading or inference fails unexpectedly.
        """
        question = self._validate_question(question)

        try:
            self._ensure_loaded()
            prompt    = self._build_prompt(question, memory)
            input_ids = self._tokenize(prompt)
            result    = self._generate(input_ids)

        except (ValueError, RuntimeError):
            raise
        except Exception as exc:
            name = self.__class__.__name__
            logger.exception("%s encountered an unexpected error.", name)
            raise RuntimeError(
                f"{name} failed to generate an answer. "
                f"Cause: {type(exc).__name__}: {exc}"
            ) from exc

        logger.info(
            "%s answered (%d chars) | question: '%.60s...'",
            self.__class__.__name__, len(result), question,
        )
        return result

    # ------------------------------------------------------------------
    # Internal — lazy loading (SRP: loading is separate from generation)
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """
        Lazy-load the model, tokenizer, and system prompt on the first call.

        Subsequent calls are no-ops: the None-guard short-circuits immediately,
        making this safe to call before every ``answer()`` invocation.
        """
        if self._tokenizer is None or self._model is None:
            logger.info("%s: loading model and tokenizer …", self.__class__.__name__)
            self._tokenizer, self._model = load_model_and_tokenizer()

        if self._system_prompt is None:
            logger.info(
                "%s: loading system prompt '%s.txt' …",
                self.__class__.__name__, self.prompt_name,
            )
            self._system_prompt = self._prompt_manager.get_prompt(self.prompt_name)

    # ------------------------------------------------------------------
    # Internal — prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, question: str, memory: "Optional[ConversationMemory]" = None) -> str:
        """
        Format the conversation using TinyLlama's ChatML template.

        When ``memory`` contains previous turns, they are injected between
        the system prompt and the current user turn so TinyLlama can
        maintain context across the conversation::

            <|system|>
            {system_prompt}</s>
            <|user|>                       ← history turn 1
            {past_question_1}</s>
            <|assistant|>
            {past_answer_1}</s>
            <|user|>                       ← history turn N
            {past_question_N}</s>
            <|assistant|>
            {past_answer_N}</s>
            <|user|>                       ← current question
            {question}</s>
            <|assistant|>
        """
        history_block = memory.to_chatml_history() if (memory and not memory.is_empty) else ""
        return (
            f"<|system|>\n{self._system_prompt}</s>\n"
            f"{history_block}"
            f"<|user|>\n{question}</s>\n"
            f"<|assistant|>\n"
        )

    # ------------------------------------------------------------------
    # Internal — tokenisation
    # ------------------------------------------------------------------

    def _tokenize(self, prompt: str) -> torch.Tensor:
        """
        Tokenise the formatted prompt and move the tensor to the model's device.

        Returns
        -------
        torch.Tensor
            Input IDs shaped ``(1, seq_len)`` on the model's device.
        """
        device = next(self._model.parameters()).device
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,   # Special tokens are in the ChatML template
        )
        input_ids = inputs["input_ids"]
        self.last_prompt_tokens = input_ids.shape[-1]
        return input_ids.to(device)

    # ------------------------------------------------------------------
    # Internal — inference
    # ------------------------------------------------------------------

    def _generate(self, input_ids: torch.Tensor) -> str:
        """
        Run the model's forward pass inside ``torch.inference_mode()`` and
        decode only the newly generated tokens (prompt tokens are sliced off).

        Thread safety:
            The ``inference_lock`` from ``models.loader`` is acquired before
            ``model.generate()`` to serialize concurrent calls from the
            CollaborationEngine thread pool. All tokenisation and decoding
            happen outside the lock and run in true parallel.

        Returns
        -------
        str
            Decoded answer string, stripped of special tokens and whitespace.
        """
        cfg           = self.generation_config
        prompt_length = input_ids.shape[-1]

        # Acquire lock to serialize GPU inference across threads.
        # Pre/post processing (tokenisation, decoding) runs in parallel.
        with inference_lock:
            with torch.inference_mode():
                output_ids = self._model.generate(
                    input_ids,
                    max_new_tokens    = cfg.max_new_tokens,
                    do_sample         = True,
                    temperature       = cfg.temperature,
                    top_p             = cfg.top_p,
                    repetition_penalty= cfg.repetition_penalty,
                    pad_token_id      = self._tokenizer.eos_token_id,
                    eos_token_id      = self._tokenizer.eos_token_id,
                )

        # Slice off the prompt tokens — return only the model's new tokens.
        new_tokens = output_ids[0][prompt_length:]
        self.last_tokens_generated = len(new_tokens)
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Internal — input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_question(question: str) -> str:
        """
        Strip and validate the question string.

        Raises
        ------
        ValueError
            If the stripped question is empty.
        """
        question = question.strip()
        if not question:
            raise ValueError(
                "Question must not be empty. Please provide a non-blank string."
            )
        return question
