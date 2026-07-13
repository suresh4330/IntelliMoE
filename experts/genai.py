"""
experts/genai.py
----------------
GenAIExpert — generative AI and large language model domain.

Updated to use the Google Gemini API instead of the local TinyLlama model
to provide more creative and complete explanations of GenAI concepts, RAG, and agents.
"""

import logging
from typing import Optional, TYPE_CHECKING

from config.settings import EXPERT_CONFIGS, GEMINI_MODEL_ID, GenerationConfig
from experts.base import BaseExpert
from services.gemini_client import generate_response

if TYPE_CHECKING:
    from utils.memory import ConversationMemory

logger = logging.getLogger(__name__)


class GenAIExpert(BaseExpert):
    """Expert for LLMs, prompt engineering, RAG, agents, and the GenAI ecosystem."""

    @property
    def prompt_name(self) -> str:
        return "genai"

    @property
    def generation_config(self) -> GenerationConfig:
        return EXPERT_CONFIGS["genai"]

    def answer(self, question: str, memory: "Optional[ConversationMemory]" = None) -> str:
        """
        Generate a GenAI expert answer using the Gemini API.

        This overrides the parent class method to bypass local model loading and
        use Gemini client for inference while preserving the same interface.
        """
        question = self._validate_question(question)

        # Lazy-load system prompt using inherited PromptManager
        if self._system_prompt is None:
            self._system_prompt = self._prompt_manager.get_prompt(self.prompt_name)

        # Build prompt using memory (if present)
        if memory and not memory.is_empty:
            history_text = ""
            for turn in memory.get_turns():
                history_text += f"User: {turn.question}\nAssistant: {turn.answer}\n\n"
            full_prompt = f"Conversation history:\n{history_text}User: {question}"
        else:
            full_prompt = question

        cfg = self.generation_config
        logger.info("GenAIExpert: sending request to Gemini API...")

        try:
            # Generate the response using Gemini client
            response = generate_response(
                prompt=full_prompt,
                system_prompt=self._system_prompt,
                model=GEMINI_MODEL_ID,
                temperature=cfg.temperature,
            )

            # Update token metrics for telemetry/cost tracking
            # 1 token is roughly 4 characters
            system_len = len(self._system_prompt) if self._system_prompt else 0
            self.last_prompt_tokens = (len(full_prompt) + system_len) // 4
            self.last_tokens_generated = len(response) // 4

            logger.info("GenAIExpert: successfully generated response from Gemini API.")
            return response

        except Exception as exc:
            logger.exception("GenAIExpert failed to generate answer using Gemini API.")
            raise RuntimeError(
                f"GenAIExpert failed to generate an answer. "
                f"Cause: {type(exc).__name__}: {exc}"
            ) from exc
