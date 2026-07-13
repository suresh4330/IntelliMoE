"""
router/llm_router.py
--------------------
LLMRoutingStrategy — intent classification via TinyLlama.

How it works:
─────────────
  1. A system prompt (prompts/routing.txt) describes all available experts
     and instructs TinyLlama to return ONLY a JSON array of expert names.
  2. The user query is appended as the user turn in ChatML format.
  3. TinyLlama runs under torch.inference_mode() with greedy decoding
     (temperature=0, do_sample=False) for deterministic, structured output.
  4. The response is parsed with a regex JSON extractor.  If parsing fails,
     validation fails, or the model returns garbage, the strategy falls back
     to KeywordRoutingStrategy — so the system is always available.

Extensibility:
──────────────
  Adding a new expert requires only two changes:
    1. Add its name to the ExpertName enum (router.py).
    2. Add a line describing it in prompts/routing.txt.
  No code changes to this file are ever needed.

SOLID:
──────
  - Implements RoutingStrategy (Liskov, Interface Segregation).
  - Depends on load_model_and_tokenizer() abstraction (Dependency Inversion).
  - Fallback strategy is injected via constructor (Open/Closed).
"""

import json
import logging
import re
from typing import Optional

from utils.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generation config for the routing task
# ---------------------------------------------------------------------------
# We want DETERMINISTIC output → greedy decoding (do_sample=False).
# The response is a short JSON array, so max_new_tokens=64 is generous.
_ROUTING_MAX_NEW_TOKENS: int = 64
_ROUTING_TEMPERATURE: float  = 0.1   # Near-zero for near-greedy sampling
_ROUTING_TOP_P: float         = 1.0
_ROUTING_REPETITION_PENALTY  = 1.05

# Name of the routing system-prompt file (in prompts/).
_ROUTING_PROMPT_NAME: str = "routing"


class LLMRoutingStrategy:
    """
    Routing strategy that uses TinyLlama to classify query intent.

    The model is prompted to output a strict JSON array of expert names.
    If the model output cannot be parsed or validated, the strategy
    transparently falls back to ``fallback_strategy``.

    Parameters
    ----------
    fallback_strategy : RoutingStrategy | None
        Strategy to use when LLM output is unparseable. If None, a
        KeywordRoutingStrategy is used. Pass a custom strategy for testing.

    Examples
    --------
    >>> strategy = LLMRoutingStrategy()
    >>> experts = strategy.select_experts("Build an AI hospital system")
    >>> print([e.value for e in experts])
    ['coding', 'system_design', 'ml', 'genai']
    """

    def __init__(self, fallback_strategy=None) -> None:
        # Lazy-loaded model resources.
        self._tokenizer = None
        self._model     = None

        # System prompt loaded once from prompts/routing.txt.
        self._system_prompt: Optional[str] = None
        self._prompt_manager = PromptManager()

        # Fallback strategy — imported lazily to avoid circular imports.
        self._fallback_strategy = fallback_strategy  # resolved in _get_fallback()

    # ------------------------------------------------------------------
    # Public API — implements RoutingStrategy interface
    # ------------------------------------------------------------------

    def select_expert(self, query: str):
        """Return the single top-ranked expert (first element of select_experts)."""
        return self.select_experts(query)[0]

    def select_experts(self, query: str) -> list:
        """
        Use TinyLlama to classify the query and return all relevant experts.

        Parameters
        ----------
        query : str
            The raw user query string.

        Returns
        -------
        list[ExpertName]
            Ordered list of relevant ExpertName values (at least one).
            Falls back to KeywordRoutingStrategy on any failure.
        """
        try:
            self._ensure_loaded()
            prompt     = self._build_routing_prompt(query)
            raw_output = self._run_inference(prompt)
            expert_names = self._parse_and_validate(raw_output)

            logger.info(
                "LLM router selected: %s | query: '%.60s...'",
                [e.value for e in expert_names], query,
            )
            return expert_names

        except Exception as exc:
            logger.warning(
                "LLM routing failed (%s: %s) — falling back to keyword router.",
                type(exc).__name__, exc,
            )
            return self._get_fallback().select_experts(query)

    # ------------------------------------------------------------------
    # Internal — lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load routing system prompt on first call."""
        if self._system_prompt is None:
            logger.info("LLMRoutingStrategy: loading routing prompt …")
            self._system_prompt = self._prompt_manager.get_prompt(_ROUTING_PROMPT_NAME)

    # ------------------------------------------------------------------
    # Internal — prompt construction
    # ------------------------------------------------------------------

    def _build_routing_prompt(self, query: str) -> str:
        """
        Build a prompt that asks the model to classify ``query``.
        """
        return f"Query: {query}\nResponse:"

    # ------------------------------------------------------------------
    # Internal — inference
    # ------------------------------------------------------------------

    def _run_inference(self, prompt: str) -> str:
        """
        Run Gemini API and return the raw output.
        """
        from config.settings import GEMINI_MODEL_ID  # noqa: PLC0415
        from services.gemini_client import generate_response  # noqa: PLC0415
        
        raw = generate_response(
            prompt=prompt,
            system_prompt=self._system_prompt,
            model=GEMINI_MODEL_ID,
            temperature=0.1,  # Near-zero for deterministic JSON output
        )
        logger.debug("LLM raw routing output: %r", raw)
        return raw

    # ------------------------------------------------------------------
    # Internal — JSON parsing and validation
    # ------------------------------------------------------------------

    def _parse_and_validate(self, raw_output: str) -> list:
        """
        Extract a JSON array from the model's raw output and validate each
        element against the ExpertName enum.

        Strategy:
          1. Find the first ``[...]`` block in the output with regex.
          2. Parse it as JSON.
          3. Validate each string is a known ExpertName value.
          4. Raise ValueError on any failure → triggers fallback.

        Parameters
        ----------
        raw_output : str
            The raw decoded text from the model.

        Returns
        -------
        list[ExpertName]
            Validated list of ExpertName values.

        Raises
        ------
        ValueError
            If no JSON array is found, JSON is malformed, or any returned
            expert name is not a member of ExpertName.
        """
        # Lazy import avoids circular dependency at module load time.
        from router.router import ExpertName  # noqa: PLC0415

        # Step 1: Extract the first [...] block from the output.
        match = re.search(r"\[.*?\]", raw_output, re.DOTALL)
        if not match:
            raise ValueError(
                f"No JSON array found in model output: {raw_output!r}"
            )

        json_str = match.group(0)

        # Step 2: Parse as JSON.
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON in model output: {json_str!r}. Error: {exc}"
            ) from exc

        # Step 3: Validate type and contents.
        if not isinstance(parsed, list) or not parsed:
            raise ValueError(
                f"Expected a non-empty JSON array, got: {parsed!r}"
            )

        valid_values = {e.value for e in ExpertName}
        expert_names = []

        for item in parsed:
            if not isinstance(item, str):
                raise ValueError(f"Expected string expert name, got: {item!r}")

            item = item.strip().lower()
            if item not in valid_values:
                raise ValueError(
                    f"Unknown expert name '{item}'. "
                    f"Valid names: {sorted(valid_values)}"
                )
            expert_names.append(ExpertName(item))

        # Deduplicate while preserving order.
        seen = set()
        unique = []
        for name in expert_names:
            if name not in seen:
                seen.add(name)
                unique.append(name)

        return unique

    # ------------------------------------------------------------------
    # Internal — fallback
    # ------------------------------------------------------------------

    def _get_fallback(self):
        """
        Return the fallback routing strategy, initialising it lazily.

        Lazy import of SemanticRoutingStrategy breaks the circular import
        that would occur if router.py is imported at module level here.
        """
        if self._fallback_strategy is None:
            from router.router import SemanticRoutingStrategy  # noqa: PLC0415
            self._fallback_strategy = SemanticRoutingStrategy()
            logger.info("LLMRoutingStrategy: fallback initialised (SemanticRoutingStrategy).")
        return self._fallback_strategy
