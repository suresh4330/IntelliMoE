"""
experts/math.py
---------------
MathExpert — mathematics and quantitative reasoning domain.

Updated to use the OpenAI API (gpt-4o-mini) instead of the Groq API
to leverage OpenAI's excellent mathematical reasoning capabilities.
"""

import logging
from typing import Optional, TYPE_CHECKING

from config.settings import EXPERT_CONFIGS, OPENAI_MODEL_ID, GenerationConfig
from experts.base import BaseExpert
from services.openai_client import generate_response

if TYPE_CHECKING:
    from utils.memory import ConversationMemory

logger = logging.getLogger(__name__)


class MathExpert(BaseExpert):
    """Expert for mathematics, calculus, algebra, and quantitative reasoning."""

    @property
    def prompt_name(self) -> str:
        return "math"

    @property
    def generation_config(self) -> GenerationConfig:
        return EXPERT_CONFIGS["math"]

    def answer(self, question: str, memory: "Optional[ConversationMemory]" = None) -> str:
        """
        Generate a mathematical-expert answer using the OpenAI API.

        This overrides the parent class method to bypass local model loading and
        use OpenAI client for inference while preserving the same interface.
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
        logger.info("MathExpert: sending request to OpenAI API...")

        try:
            # Generate the response using OpenAI client
            response = generate_response(
                prompt=full_prompt,
                system_prompt=self._system_prompt,
                model=OPENAI_MODEL_ID,
                temperature=cfg.temperature,
                max_tokens=cfg.max_new_tokens,
            )

            # Update token metrics for telemetry/cost tracking
            # 1 token is roughly 4 characters
            system_len = len(self._system_prompt) if self._system_prompt else 0
            self.last_prompt_tokens = (len(full_prompt) + system_len) // 4
            self.last_tokens_generated = len(response) // 4

            logger.info("MathExpert: successfully generated response from OpenAI API.")
            return response

        except Exception as exc:
            logger.exception("MathExpert failed to generate answer using OpenAI API.")
            raise RuntimeError(
                f"MathExpert failed to generate an answer. "
                f"Cause: {type(exc).__name__}: {exc}"
            ) from exc
