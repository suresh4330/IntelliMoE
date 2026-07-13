"""
conversation_ai/responder.py
-----------------------------
ConversationalResponder — generates warm, context-aware conversational replies
using the Groq LLM with full conversation memory context.

Design principles:
  - Dynamic, non-templated responses (each call generates a unique reply).
  - Context-aware: reads ALL prior turns from ConversationMemory, including
    which experts were activated and what they said.
  - Natural personality: friendly, curious, helpful; asks follow-up questions
    when appropriate; never repeats the same greeting.
  - For follow-ups / clarifications: references the last expert answer.
  - Uses llama-3.1-8b-instant via services/groq_client for speed.

The system persona is injected as a rich system prompt that instructs the
model to behave like a knowledgeable, friendly assistant—not a bot listing
its capabilities or repeating templates.
"""

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from utils.memory import ConversationMemory
    from conversation_ai.detector import IntentResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System persona prompt
# ---------------------------------------------------------------------------

_SYSTEM_PERSONA = """You are IntelliMoE, a smart, conversational, and genuinely helpful AI assistant.

Your personality:
- Human-like, natural, and warm. Speak like a friendly colleague or friend, not a stiff computer program.
- Conversational and fluid — use varied sentence structures, transition words, and natural phrasing.
- Empathetic and engaging — acknowledge the user's feelings, questions, or updates. If they share good news, be happy for them! If they share a problem, be supportive.
- Brief but descriptive — keep casual chat to 1-3 sentences, but provide complete and useful answers when they ask for information.
- Multilingual — if the user writes in Telugu, Hindi, Tamil, or any other language, respond in that same language naturally. If they write in mixed English (e.g. Hinglish or Telugish), reply in a similar friendly, natural mixed style!

Strict rules:
- NEVER say "As an AI" or refer to yourself as a language model.
- Avoid repetitive structural templates. Do not start every sentence with the same words.
- Be supportive, friendly, and helpful. Use emojis naturally where they fit.
- If the user's message is in a local language (e.g., Telugu "ela unnav"), reply warmly in that language (e.g. "Nenu chala baagunnanu, thank you! Meeru ela unnaru? Em help kavali?").
"""


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_conversation_context(memory: "ConversationMemory", max_turns: int = 6) -> str:
    """
    Build a rich conversation history string from ConversationMemory.

    Includes user questions, assistant answers, and which experts were activated,
    so the LLM can reference prior expert responses in follow-up replies.

    Parameters
    ----------
    memory : ConversationMemory
        The active session memory.
    max_turns : int
        Maximum number of prior turns to include (most recent N).

    Returns
    -------
    str
        A formatted conversation history string, or "" if memory is empty.
    """
    turns = memory.get_turns()
    if not turns:
        return ""

    # Use the most recent N turns
    recent = turns[-max_turns:]

    lines: list[str] = ["[CONVERSATION HISTORY]"]
    for i, turn in enumerate(recent, 1):
        lines.append(f"User: {turn.question}")

        # Include expert context for richer follow-up responses
        expert_label = ""
        if turn.experts:
            expert_names = ", ".join(
                e.replace("_", " ").title() for e in turn.experts
                if e != "conversational"
            )
            if expert_names:
                expert_label = f" [via {expert_names} Expert]"

        # Truncate very long expert answers to keep context manageable
        answer = turn.answer
        if len(answer) > 600:
            answer = answer[:580].rstrip() + "…"

        lines.append(f"Assistant{expert_label}: {answer}")
        if i < len(recent):
            lines.append("")  # blank line between turns

    lines.append("[END HISTORY]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Intent-specific prompt builders
# ---------------------------------------------------------------------------

def _build_prompt_for_intent(
    query: str,
    intent_type: str,
    context: str,
    is_first_message: bool,
) -> str:
    """
    Build a targeted prompt based on the detected intent so the LLM can
    generate the most appropriate response style.
    """
    context_block = f"\n\n{context}\n\n" if context else "\n\n"

    base = f"{context_block}User's current message: {query}\n\n"

    intent_instructions = {
        "greeting": (
            "Generate a warm, friendly greeting response. "
            + ("This is the FIRST message — welcome them genuinely but briefly." if is_first_message
               else "This is NOT the first message — don't say 'welcome' again. Acknowledge them naturally.")
            + " Ask what they're working on or how you can help. Keep it to 1–2 sentences. Vary your wording."
        ),
        "farewell": (
            "The user is saying goodbye. Respond warmly and briefly. "
            "Wish them well. Keep it to 1 sentence. Be natural."
        ),
        "small_talk": (
            "Respond to this small talk naturally and briefly (2–3 sentences). "
            "Be warm and genuine. If they asked how you are, answer naturally then redirect to them. "
            "Optionally ask what they're working on."
        ),
        "general_knowledge": (
            "Answer this general knowledge question conversationally and accurately. "
            "Be helpful and clear. Keep it concise. If relevant, mention you can dive deeper."
        ),
        "follow_up": (
            "The user wants to continue or expand on the previous topic. "
            "Reference the prior conversation naturally. Expand thoughtfully. "
            "Keep the flow of conversation going."
        ),
        "clarification": (
            "The user wants you to clarify or re-explain something from earlier. "
            "Look at the conversation history and re-explain the relevant point "
            "more simply or from a different angle. Be patient and clear."
        ),
    }

    instruction = intent_instructions.get(
        intent_type,
        "Respond naturally and helpfully to the user's message."
    )

    return base + f"Task: {instruction}\n\nGenerate a natural, conversational response:"


# ---------------------------------------------------------------------------
# ConversationalResponder
# ---------------------------------------------------------------------------

class ConversationalResponder:
    """
    Generates natural, context-aware conversational responses using the Groq LLM.

    Uses the full ConversationMemory (prior turns, expert context, conversation
    thread) to produce replies that feel like a continuous, coherent conversation.

    Parameters
    ----------
    model : str
        Groq model to use. Defaults to "llama-3.1-8b-instant" for speed.
    temperature : float
        Sampling temperature. Higher = more varied responses.
    max_tokens : int
        Maximum tokens to generate per conversational reply.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.72,
        max_tokens: int = 200,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def respond(
        self,
        query: str,
        intent_result: "IntentResult",
        memory: "ConversationMemory",
    ) -> str:
        """
        Generate a conversational reply for the given query.

        Parameters
        ----------
        query : str
            The user's current message.
        intent_result : IntentResult
            Classification result from IntentDetector.
        memory : ConversationMemory
            Active session memory containing full conversation history.

        Returns
        -------
        str
            A natural, context-aware conversational response.
        """
        is_first_message = memory.is_empty
        context = _build_conversation_context(memory, max_turns=6)

        prompt = _build_prompt_for_intent(
            query=query,
            intent_type=intent_result.intent.value,
            context=context,
            is_first_message=is_first_message,
        )

        logger.info(
            "ConversationalResponder: generating reply for intent='%s' "
            "(is_first=%s, context_turns=%d)",
            intent_result.intent.value,
            is_first_message,
            len(memory.get_turns()),
        )

        try:
            from services.groq_client import generate_response  # noqa: PLC0415

            response = generate_response(
                prompt=prompt,
                system_prompt=_SYSTEM_PERSONA,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

            response = response.strip()

            if not response:
                logger.warning("ConversationalResponder: received empty response from Groq.")
                return self._safe_fallback(intent_result.intent.value, is_first_message)

            logger.info(
                "ConversationalResponder: generated %d-char reply.",
                len(response)
            )
            return response

        except Exception as exc:
            logger.error("ConversationalResponder: Groq call failed — %s", exc)
            return self._safe_fallback(intent_result.intent.value, is_first_message)

    def _safe_fallback(self, intent: str, is_first: bool) -> str:
        """
        Minimal non-template fallbacks for when the LLM call fails.
        These are last-resort only — production responses come from the LLM.
        """
        fallbacks = {
            "greeting": (
                "Hey there! 👋 Great to have you here. What are you working on today?"
                if is_first else
                "Good to hear from you! What can I help you with?"
            ),
            "farewell": "Take care! Feel free to come back anytime. 👋",
            "small_talk": "Doing well, thanks! 😊 What are you exploring today?",
            "follow_up": "Let me expand on that — happy to go deeper on any part of it.",
            "clarification": "Of course! Let me explain that differently.",
            "general_knowledge": "That's a great question! Let me think through that for you.",
        }
        return fallbacks.get(intent, "I'm here to help — what would you like to explore?")
