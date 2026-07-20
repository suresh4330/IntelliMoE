"""
utils/memory.py
---------------
ConversationMemory — stores and manages multi-turn conversation history.

How it plugs in:
  UI → router.route(query, memory) → expert.answer(query, memory)
       → BaseExpert._build_prompt(question, memory)
       → multi-turn ChatML prompt sent to TinyLlama

TinyLlama ChatML multi-turn format:
  <|system|>
  {system_prompt}</s>
  <|user|>
  {turn_1_question}</s>
  <|assistant|>
  {turn_1_answer}</s>
  <|user|>
  {turn_2_question}</s>
  <|assistant|>
  {turn_2_answer}</s>
  <|user|>
  {current_question}</s>
  <|assistant|>

Design decisions:
  - History is stored as a list of Message objects (not raw strings).
  - max_turns caps the history to prevent the context window from overflowing.
  - When the cap is reached, the OLDEST turn is dropped (sliding window).
  - max_chars_per_message truncates excessively long stored messages.
  - Memory is NOT thread-safe by design — one instance per user session.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Message — a single utterance in the conversation
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """
    One message in the conversation.

    Attributes
    ----------
    role : str
        Either ``"user"`` or ``"assistant"``.
    content : str
        The text of the message.
    timestamp : datetime
        Wall-clock time when the message was recorded.
    expert : str | None
        The expert that produced this message (only set for assistant messages).
    experts : list[str]
        A list of all expert names activated to generate this message.
    """
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    expert: Optional[str] = None
    experts: list[str] = field(default_factory=list)
    image_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got: {self.role!r}")
        if not self.content.strip():
            raise ValueError("Message content must not be empty.")


# ---------------------------------------------------------------------------
# Turn — one full exchange (user question + assistant answer)
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """A matched pair of user question and assistant answer."""
    user_message:      Message
    assistant_message: Message

    @property
    def question(self) -> str:
        return self.user_message.content

    @property
    def answer(self) -> str:
        return self.assistant_message.content

    @property
    def expert(self) -> Optional[str]:
        return self.assistant_message.expert

    @property
    def experts(self) -> list[str]:
        return self.assistant_message.experts

    @property
    def image_path(self) -> Optional[str]:
        return self.user_message.image_path


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------

class ConversationMemory:
    """
    Stores and manages multi-turn conversation history for an IntelliMoE session.

    Responsibilities:
      - Record question/answer pairs as ``Turn`` objects.
      - Enforce a sliding-window cap (``max_turns``) on history depth.
      - Format stored history as the ChatML turns string for model input.
      - Provide clear(), statistics, and iteration support.

    Parameters
    ----------
    max_turns : int
        Maximum number of past turns to retain. When the cap is reached,
        the oldest turn is dropped (sliding window). Default: 10.
    max_chars_per_message : int
        Truncate stored message text to this length to prevent runaway
        context growth from very long answers. Default: 1500.

    Examples
    --------
    >>> memory = ConversationMemory(max_turns=5)
    >>> memory.add_turn("What is Python?", "Python is ...", expert="coding")
    >>> memory.add_turn("Give an example.", "Sure, here is ...", expert="coding")
    >>> print(memory.to_chatml_history())
    >>> memory.clear()
    """

    def __init__(
        self,
        max_turns: int = 10,
        max_chars_per_message: int = 100000,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")

        self._max_turns = max_turns
        self._max_chars = max_chars_per_message
        self._turns: list[Turn] = []

        logger.debug(
            "ConversationMemory initialised (max_turns=%d, max_chars=%d).",
            max_turns, max_chars_per_message,
        )

    # ------------------------------------------------------------------
    # Public API — writing
    # ------------------------------------------------------------------

    def add_turn(
        self,
        question: str,
        answer: str,
        expert: Optional[str] = None,
        experts: Optional[list[str]] = None,
        image_path: Optional[str] = None,
    ) -> None:
        """
        Record a completed question/answer exchange.

        If the history is at capacity, the oldest turn is automatically
        dropped to make room (sliding window).

        Parameters
        ----------
        question : str
            The user's question for this turn.
        answer : str
            The expert's answer for this turn.
        expert : str | None
            Name of the primary expert that generated the answer (for backward compatibility).
        experts : list[str] | None
            Names of all active experts contributing to the turn.
        image_path : str | None
            Path of the uploaded diagram/image if present.
        """
        user_msg = Message(
            role="user",
            content=self._truncate(question),
            image_path=image_path
        )
        
        # Default list of experts
        active_experts = experts or []
        if not active_experts and expert:
            active_experts = [expert]

        assistant_msg = Message(
            role="assistant",
            content=self._truncate(answer),
            expert=expert,
            experts=active_experts,
        )

        turn = Turn(user_message=user_msg, assistant_message=assistant_msg)
        self._turns.append(turn)

        # Sliding window: drop oldest turn when cap is exceeded.
        if len(self._turns) > self._max_turns:
            dropped = self._turns.pop(0)
            logger.debug(
                "Memory cap reached — dropped oldest turn: '%.40s...'",
                dropped.question,
            )

        logger.info(
            "Turn recorded (expert=%s, experts=%s, depth=%d/%d) | Q: '%.50s...'",
            expert or "unknown", active_experts, len(self._turns), self._max_turns, question,
        )

    def clear(self) -> None:
        """
        Erase all stored conversation history.

        After calling this, the next ``answer()`` call will have no prior context.
        """
        count = len(self._turns)
        self._turns.clear()
        logger.info("ConversationMemory cleared (%d turns removed).", count)

    # ------------------------------------------------------------------
    # Public API — reading
    # ------------------------------------------------------------------

    def to_chatml_history(self) -> str:
        """
        Format stored history as the ChatML turns block to inject into the prompt.

        The returned string contains all past turns in TinyLlama's ChatML format,
        ready to be inserted between the system prompt and the current user turn::

            <|user|>
            {turn_1_question}</s>
            <|assistant|>
            {turn_1_answer}</s>
            <|user|>
            {turn_2_question}</s>
            <|assistant|>
            {turn_2_answer}</s>

        Returns an empty string if no turns have been recorded.
        """
        if not self._turns:
            return ""

        parts: list[str] = []
        for turn in self._turns:
            parts.append(f"<|user|>\n{turn.question}</s>")
            parts.append(f"<|assistant|>\n{turn.answer}</s>")

        return "\n".join(parts) + "\n"

    def get_turns(self) -> list[Turn]:
        """
        Return a shallow copy of the stored turns list.

        Returns
        -------
        list[Turn]
            Chronologically ordered list of past turns (oldest first).
        """
        return list(self._turns)

    def get_messages(self) -> list[Message]:
        """
        Return all stored messages as a flat list (user and assistant interleaved).

        Returns
        -------
        list[Message]
            Chronologically ordered: user, assistant, user, assistant, ...
        """
        messages: list[Message] = []
        for turn in self._turns:
            messages.append(turn.user_message)
            messages.append(turn.assistant_message)
        return messages

    @property
    def is_empty(self) -> bool:
        """True if no turns have been recorded."""
        return len(self._turns) == 0

    @property
    def depth(self) -> int:
        """Number of stored turns."""
        return len(self._turns)

    @property
    def max_turns(self) -> int:
        """The configured maximum number of turns."""
        return self._max_turns

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._turns)

    def __iter__(self):
        return iter(self._turns)

    def __repr__(self) -> str:
        return (
            f"ConversationMemory("
            f"depth={self.depth}, "
            f"max_turns={self._max_turns})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _truncate(self, text: str) -> str:
        """
        Truncate ``text`` to ``max_chars_per_message`` characters.

        A truncation marker is appended when text is actually shortened
        so the model knows the context is incomplete.
        """
        text = text.strip()
        if len(text) <= self._max_chars:
            return text
        truncated = text[: self._max_chars].rstrip()
        logger.debug("Message truncated from %d to %d chars.", len(text), len(truncated))
        return truncated + " [...]"
