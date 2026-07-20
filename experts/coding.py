"""
experts/coding.py
-----------------
CodingExpert — software engineering and programming domain.

Dual-API Strategy: OpenAI (primary) → Gemini (fallback)
  - Primary  : OpenAI gpt-4o-mini  — excellent reasoning and code quality.
  - Fallback : Google Gemini       — activates automatically if OpenAI fails
                                     (quota, network error, etc.)
"""

import logging
from typing import Optional, TYPE_CHECKING

from config.settings import EXPERT_CONFIGS, OPENAI_MODEL_ID, GEMINI_MODEL_ID, GenerationConfig
from experts.base import BaseExpert

if TYPE_CHECKING:
    from utils.memory import ConversationMemory

logger = logging.getLogger(__name__)


class CodingExpert(BaseExpert):
    """Expert for software engineering, algorithms, and programming queries."""

    @property
    def prompt_name(self) -> str:
        return "coding"

    @property
    def generation_config(self) -> GenerationConfig:
        return EXPERT_CONFIGS["coding"]

    def answer(self, question: str, memory: "Optional[ConversationMemory]" = None) -> str:
        """
        Generate a coding-expert answer using OpenAI (primary) with Gemini as fallback.

        Tries OpenAI first. If it fails for any reason (quota, timeout, error),
        automatically falls back to the Gemini API transparently.
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
        system_len = len(self._system_prompt) if self._system_prompt else 0

        # ------------------------------------------------------------------
        # Primary: OpenAI API
        # ------------------------------------------------------------------
        try:
            from services.openai_client import generate_response as openai_generate  # noqa: PLC0415

            logger.info("CodingExpert: sending request to OpenAI API (primary)...")
            response = openai_generate(
                prompt=full_prompt,
                system_prompt=self._system_prompt,
                model=OPENAI_MODEL_ID,
                temperature=cfg.temperature,
                max_tokens=cfg.max_new_tokens,
            )

            self.last_prompt_tokens = (len(full_prompt) + system_len) // 4
            self.last_tokens_generated = len(response) // 4

            logger.info("CodingExpert: successfully generated response from OpenAI API.")
            return response

        except Exception as openai_exc:
            logger.warning(
                "CodingExpert: OpenAI API failed (%s). Falling back to Gemini API...",
                openai_exc,
            )

        # ------------------------------------------------------------------
        # Fallback: Gemini API
        # ------------------------------------------------------------------
        try:
            from services.gemini_client import generate_response as gemini_generate  # noqa: PLC0415

            logger.info("CodingExpert: sending request to Gemini API (fallback)...")
            response = gemini_generate(
                prompt=full_prompt,
                system_prompt=self._system_prompt,
                model=GEMINI_MODEL_ID,
                temperature=cfg.temperature,
            )

            self.last_prompt_tokens = (len(full_prompt) + system_len) // 4
            self.last_tokens_generated = len(response) // 4

            logger.info("CodingExpert: successfully generated response from Gemini API (fallback).")
            return response

        except Exception as gemini_exc:
            logger.exception("CodingExpert: both OpenAI and Gemini APIs failed.")
            raise RuntimeError(
                f"CodingExpert failed to generate an answer. "
                f"OpenAI error: {openai_exc} | Gemini error: {gemini_exc}"
            ) from gemini_exc
