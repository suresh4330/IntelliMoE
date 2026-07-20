"""
conversation_ai/layer.py
-------------------------
ConversationLayer — the single public entry point of the Conversational AI Layer.

This module is called from ui/app.py's _handle_query() BEFORE any router or
expert system is invoked. It acts as the intelligent gatekeeper that decides
whether to:

  1. Handle the query directly (conversational response via LLM)  ← is_conversational=True
  2. Forward to the Hybrid Router pipeline                        ← is_conversational=False

Workflow
--------
  User query
      │
      ▼
  ConversationLayer.process()
      │
      ├── IntentDetector.detect()          ← 3-tier: Rule → ML → LLM
      │       │
      │   Conversational intent?
      │       │
      │   YES ├── ConversationalResponder.respond()  ← LLM with full memory context
      │       │       └── returns ConversationResult(is_conversational=True, response=...)
      │       │
      │   NO  └── returns ConversationResult(is_conversational=False)
      │                   └── Caller (app.py) routes to Hybrid Router as before
      │
      ▼
  ConversationResult returned to app.py

Integration
-----------
In ui/app.py._handle_query(), the layer is invoked as the very first step:

    from conversation_ai.layer import ConversationLayer
    layer = ConversationLayer()
    result = layer.process(query, memory)

    if result.is_conversational:
        # Record turn and return — never touches Hybrid Router
        memory.add_turn(query, result.response, expert="conversational", experts=["conversational"])
        ...
        return

    # Everything below is unchanged — Hybrid Router, Decision Engine, Planner, etc.
"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from utils.memory import ConversationMemory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConversationResult:
    """
    Outcome of ConversationLayer.process().

    Attributes
    ----------
    is_conversational : bool
        True → caller should use ``response`` directly.
        False → caller should route to Hybrid Router.
    response : str | None
        The generated conversational reply. Only set when is_conversational=True.
    intent : str
        Detected intent label (e.g. 'greeting', 'coding', 'follow_up').
    confidence : float
        Detection confidence in [0.0, 1.0].
    tier_used : str
        Which detection tier produced the result ('rule', 'ml', 'llm', 'heuristic').
    reasoning : str
        Human-readable explanation of the routing decision.
    response_time_s : float
        Wall-clock time taken by the Conversation AI Layer in seconds.
    """
    is_conversational: bool
    response: Optional[str]
    intent: str
    confidence: float
    tier_used: str
    reasoning: str
    response_time_s: float = 0.0
    original_query: str = ""
    cleaned_query: str = ""
    greeting_removed: bool = False
    routing_decision: str = ""


# ---------------------------------------------------------------------------
# ConversationLayer
# ---------------------------------------------------------------------------

class ConversationLayer:
    """
    Intelligent Conversational AI Layer — the gatekeeper between user input
    and the Multi-Expert routing pipeline.

    Singleton-friendly: safe to instantiate once and reuse across requests
    (all state is in the caller's ConversationMemory, not here).

    Parameters
    ----------
    ml_confidence_threshold : float | None
        Override for the ML confidence threshold. Uses the project default
        (ML_ROUTING_CONFIDENCE_THRESHOLD from config) when None.
    """

    def __init__(self, ml_confidence_threshold: Optional[float] = None) -> None:
        from conversation_ai.detector import IntentDetector    # noqa: PLC0415
        from conversation_ai.responder import ConversationalResponder  # noqa: PLC0415

        self._detector = IntentDetector(ml_confidence_threshold=ml_confidence_threshold)
        self._responder = ConversationalResponder()

        logger.info("ConversationLayer initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        query: str,
        memory: "ConversationMemory",
    ) -> ConversationResult:
        """
        Classify the query and either generate a conversational reply or
        signal that expert routing is needed.

        Parameters
        ----------
        query : str
            Raw user input.
        memory : ConversationMemory
            Active session memory. Used for context-aware replies and
            follow-up / clarification detection.

        Returns
        -------
        ConversationResult
            If is_conversational=True → use response field directly.
            If is_conversational=False → pass query to Hybrid Router.
        """
        t_start = time.perf_counter()
        query = query.strip()

        has_prior_context = not memory.is_empty

        # Step 1: Normalize query — strip leading greeting words only
        from conversation_ai.detector import clean_query, IntentResult, IntentType  # noqa: PLC0415
        cleaned_query, greeting_removed = clean_query(query)

        # If the cleaned query is empty (e.g. user typed ONLY "Hi" or "Hey"),
        # it is a pure greeting — handle conversationally.
        if not cleaned_query:
            intent_result = IntentResult(
                intent=IntentType.GREETING,
                is_conversational=True,
                confidence=1.0,
                reasoning="Query contained only a greeting/filler with no actionable content.",
                tier_used="rule",
            )
        else:
            # Step 2: Detect intent on the ORIGINAL full query so the complete
            # sentence context is available (greeting words are NOT stripped here).
            # This ensures "Hey who won today's ODI?" is never mistaken for a greeting.
            intent_result = self._detector.detect(query, has_prior_context=has_prior_context)

        # Step 3: Add confidence score check for conversation detection
        # If conversation confidence is below threshold, automatically forward to the Hybrid Router
        from config.settings import ML_ROUTING_CONFIDENCE_THRESHOLD  # noqa: PLC0415
        
        is_conversational = intent_result.is_conversational
        if is_conversational and intent_result.confidence < ML_ROUTING_CONFIDENCE_THRESHOLD:
            logger.info(
                "Conversational confidence (%.2f) below threshold (%.2f) - forwarding to Hybrid Router.",
                intent_result.confidence, ML_ROUTING_CONFIDENCE_THRESHOLD
            )
            is_conversational = False

        # Step 4: Route decision
        if is_conversational:
            # Generate conversational reply with full memory context
            response = self._responder.respond(
                query=cleaned_query if cleaned_query else query,
                intent_result=intent_result,
                memory=memory,
            )
            elapsed = time.perf_counter() - t_start
            logger.info(
                "ConversationLayer: conversational reply generated in %.3fs for intent='%s'.",
                elapsed, intent_result.intent.value
            )
            return ConversationResult(
                is_conversational=True,
                response=response,
                intent=intent_result.intent.value,
                confidence=intent_result.confidence,
                tier_used=intent_result.tier_used,
                reasoning=intent_result.reasoning,
                response_time_s=elapsed,
                original_query=query,
                cleaned_query=cleaned_query,
                greeting_removed=greeting_removed,
                routing_decision="Pure Conversation",
            )

        # Technical intent or low-confidence conversational intent → signal caller to use Hybrid Router
        elapsed = time.perf_counter() - t_start
        logger.info(
            "ConversationLayer: forwarding to Hybrid Router (intent='%s', %.3fs).",
            intent_result.intent.value, elapsed
        )
        return ConversationResult(
            is_conversational=False,
            response=None,
            intent=intent_result.intent.value,
            confidence=intent_result.confidence,
            tier_used=intent_result.tier_used,
            reasoning=intent_result.reasoning,
            response_time_s=elapsed,
            original_query=query,
            cleaned_query=cleaned_query,
            greeting_removed=greeting_removed,
            routing_decision="Forward to Hybrid Router",
        )
