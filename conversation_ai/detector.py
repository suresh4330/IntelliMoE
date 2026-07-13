"""
conversation_ai/detector.py
----------------------------
IntentDetector — classifies user input into 13 intent types using a
three-tier hybrid approach that mirrors the existing Hybrid Router design:

  Tier 1 — Rule-based fast pass (greetings / farewells)
            Zero-latency; catches the most unambiguous cases instantly.

  Tier 2 — ML Intent Classifier (primary)
            Re-uses the same trained scikit-learn model + TF-IDF vectorizer
            that drives the Hybrid Router.  If the model predicts a
            TECHNICAL expert class with confidence ≥ threshold, the query is
            marked as technical.  If it predicts low confidence OR the intent
            is outside the technical expert taxonomy (greetings, small-talk,
            etc.) it remains as "general conversational".

  Tier 3 — LLM tiebreaker (Groq)
            Called only when ML confidence < ML_ROUTING_CONFIDENCE_THRESHOLD.
            Asks Groq to return a single intent label from the canonical list.
            This matches the HybridRouter's fallback strategy.

Conversational intents (handled directly by ConversationalResponder):
  greeting, farewell, small_talk, general_knowledge, follow_up, clarification

Technical intents (forwarded to Hybrid Router):
  coding, math, machine_learning, deep_learning, research,
  system_design, technical_general
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------

class IntentType(str, Enum):
    # ── Conversational ──────────────────────────────────────────────────────
    GREETING          = "greeting"
    FAREWELL          = "farewell"
    SMALL_TALK        = "small_talk"
    GENERAL_KNOWLEDGE = "general_knowledge"
    FOLLOW_UP         = "follow_up"
    CLARIFICATION     = "clarification"
    # ── Technical (route to Hybrid Router) ──────────────────────────────────
    CODING            = "coding"
    MATH              = "math"
    MACHINE_LEARNING  = "machine_learning"
    DEEP_LEARNING     = "deep_learning"
    RESEARCH          = "research"
    SYSTEM_DESIGN     = "system_design"
    TECHNICAL_GENERAL = "technical_general"


# Intents that should be handled conversationally (no expert routing)
CONVERSATIONAL_INTENTS: frozenset[IntentType] = frozenset({
    IntentType.GREETING,
    IntentType.FAREWELL,
    IntentType.SMALL_TALK,
    IntentType.GENERAL_KNOWLEDGE,
    IntentType.FOLLOW_UP,
    IntentType.CLARIFICATION,
})

# Map from ML classifier expert labels → IntentType
_ML_EXPERT_TO_INTENT: dict[str, IntentType] = {
    "coding":        IntentType.CODING,
    "math":          IntentType.MATH,
    "ml":            IntentType.MACHINE_LEARNING,
    "deep_learning": IntentType.DEEP_LEARNING,
    "research":      IntentType.RESEARCH,
    "system_design": IntentType.SYSTEM_DESIGN,
    "genai":         IntentType.TECHNICAL_GENERAL,
    "vision":        IntentType.TECHNICAL_GENERAL,
}

# Canonical intent label list exposed to the LLM tiebreaker
_LLM_INTENT_LABELS: list[str] = [i.value for i in IntentType]


# ---------------------------------------------------------------------------
# Rule-based patterns (Tier 1) — greetings & farewells only
# ---------------------------------------------------------------------------

_GREETING_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening|day)|what'?s?\s+up|yo|sup|namaste|namaskar|ram\s*ram|hey\s*bro|oi)\b", re.IGNORECASE),
    re.compile(r"^\s*hiya\b", re.IGNORECASE),
    re.compile(r"^\s*salut\b", re.IGNORECASE),
]

_FAREWELL_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*(bye|goodbye|see\s+you|cya|farewell|take\s+care|good\s+night|catch\s+you\s+later|later|ttyl|gtg|gotta\s+go|alvida|tata)\b", re.IGNORECASE),
]

_SMALL_TALK_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(how\s+are\s+you|how\s+do\s+you\s+do|how'?s?\s+it\s+going|how\s+have\s+you\s+been|what'?s?\s+new|how\s+are\s+things|ela\s+unnav|ela\s+unnaru|baagunnara|kaise\s+ho|kya\s+kar\s+rahe\s+ho|kya\s+chal\s+raha\s+hai|sab\s+thik|kem\s+cho|kya\s+samachar)\b", re.IGNORECASE),
    re.compile(r"\b(em\s+chestunnav|em\s+chestunnaru|kya\s+kar\s+rahe\s+ho|what\s+are\s+you\s+doing|anything\s+new)\b", re.IGNORECASE),
    re.compile(r"\b(are\s+you\s+(ok|okay|good|well|fine|alright))\b", re.IGNORECASE),
    re.compile(r"\b(what('?s|\s+is)\s+your\s+name|who\s+are\s+you|what\s+can\s+you\s+do|tell\s+me\s+about\s+yourself)\b", re.IGNORECASE),
    re.compile(r"\b(thank\s+(you|u)|thanks|thx|appreciate\s+it|cheers|dhanyavad|shukriya)\b", re.IGNORECASE),
]

_FOLLOW_UP_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*(tell\s+me\s+more|can\s+you\s+elaborate|elaborate|go\s+on|continue|expand\s+on\s+that|more\s+details|what\s+do\s+you\s+mean|i\s+see[,.]?\s*but|interesting[,.]?\s*(tell|can)|what\s+about|aur\s+batao|inka\s+batao)\b", re.IGNORECASE),
    re.compile(r"^\s*(and\s+(what|how|why|when|where|who)\b)", re.IGNORECASE),
]

_CLARIFICATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(can\s+you\s+(re-?explain|clarify|rephrase|simplify)|i\s+(don'?t|didn'?t)\s+understand|what\s+do\s+you\s+mean|could\s+you\s+(clarify|explain\s+that\s+again)|say\s+that\s+again|repeat\s+that|fir\s+se|malli\s+cheppu)\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class IntentResult:
    """
    Outcome of intent classification.

    Attributes
    ----------
    intent : IntentType
        The detected intent category.
    is_conversational : bool
        True when the query should be answered directly (no expert routing).
    confidence : float
        Detection confidence in [0.0, 1.0].
    reasoning : str
        Human-readable explanation of why this classification was made.
    tier_used : str
        Which detection tier produced the result ('rule', 'ml', 'llm').
    """
    intent: IntentType
    is_conversational: bool
    confidence: float
    reasoning: str
    tier_used: str


# ---------------------------------------------------------------------------
# IntentDetector
# ---------------------------------------------------------------------------

class IntentDetector:
    """
    Three-tier hybrid intent classifier.

    Mirrors the HybridRouter design: ML primary, rule-based fast-pass
    for greetings/farewells, LLM tiebreaker when ML confidence is low.

    Parameters
    ----------
    ml_confidence_threshold : float
        ML predictions below this value trigger the LLM tiebreaker.
        Defaults to the same threshold used by HybridRouter.
    """

    def __init__(self, ml_confidence_threshold: Optional[float] = None) -> None:
        if ml_confidence_threshold is None:
            from config.settings import ML_ROUTING_CONFIDENCE_THRESHOLD
            ml_confidence_threshold = ML_ROUTING_CONFIDENCE_THRESHOLD
        self._threshold = ml_confidence_threshold
        self._ml_router = None   # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, query: str, has_prior_context: bool = False) -> IntentResult:
        """
        Classify the intent of ``query``.

        Parameters
        ----------
        query : str
            Raw user input.
        has_prior_context : bool
            True when the conversation already has prior turns.
            Influences follow-up / clarification scoring.

        Returns
        -------
        IntentResult
        """
        q = query.strip()

        # ── Tier 1: Rule-based fast pass ──────────────────────────────
        rule_result = self._rule_detect(q, has_prior_context)
        if rule_result is not None:
            logger.info(
                "IntentDetector [Tier-1/Rule]: %s (confidence=%.2f)",
                rule_result.intent.value, rule_result.confidence
            )
            return rule_result

        # ── Tier 2: ML Classifier ─────────────────────────────────────
        ml_result = self._ml_detect(q)
        if ml_result is not None and ml_result.confidence >= self._threshold:
            logger.info(
                "IntentDetector [Tier-2/ML]: %s (confidence=%.2f)",
                ml_result.intent.value, ml_result.confidence
            )
            return ml_result

        # ── Tier 3: LLM tiebreaker ────────────────────────────────────
        logger.info(
            "IntentDetector [Tier-3/LLM]: ML confidence %.2f below threshold %.2f — calling LLM.",
            ml_result.confidence if ml_result else 0.0, self._threshold
        )
        llm_result = self._llm_detect(q, has_prior_context)
        if llm_result is not None:
            return llm_result

        # ── Final fallback: use ML result even if below threshold ──────
        if ml_result is not None:
            logger.warning(
                "IntentDetector: LLM tiebreaker failed; using low-confidence ML result: %s",
                ml_result.intent.value
            )
            return ml_result

        # Absolute fallback
        return IntentResult(
            intent=IntentType.TECHNICAL_GENERAL,
            is_conversational=False,
            confidence=0.0,
            reasoning="All detection tiers failed; defaulting to technical routing.",
            tier_used="fallback",
        )

    # ------------------------------------------------------------------
    # Tier 1 — Rule-based
    # ------------------------------------------------------------------

    def _rule_detect(self, query: str, has_prior_context: bool) -> Optional[IntentResult]:
        """
        Fast regex scan for unmistakable conversational patterns.
        Only triggers for greetings, farewells, small-talk, follow-ups, clarifications.
        """
        # Greeting
        for pat in _GREETING_PATTERNS:
            if pat.search(query):
                return IntentResult(
                    intent=IntentType.GREETING,
                    is_conversational=True,
                    confidence=0.97,
                    reasoning="Matched greeting pattern.",
                    tier_used="rule",
                )

        # Farewell
        for pat in _FAREWELL_PATTERNS:
            if pat.search(query):
                return IntentResult(
                    intent=IntentType.FAREWELL,
                    is_conversational=True,
                    confidence=0.97,
                    reasoning="Matched farewell pattern.",
                    tier_used="rule",
                )

        # Small talk
        for pat in _SMALL_TALK_PATTERNS:
            if pat.search(query):
                return IntentResult(
                    intent=IntentType.SMALL_TALK,
                    is_conversational=True,
                    confidence=0.90,
                    reasoning="Matched small-talk pattern.",
                    tier_used="rule",
                )

        # Follow-up / Clarification — only meaningful when prior context exists
        if has_prior_context:
            for pat in _FOLLOW_UP_PATTERNS:
                if pat.search(query):
                    return IntentResult(
                        intent=IntentType.FOLLOW_UP,
                        is_conversational=True,
                        confidence=0.85,
                        reasoning="Matched follow-up pattern with prior conversation context.",
                        tier_used="rule",
                    )
            for pat in _CLARIFICATION_PATTERNS:
                if pat.search(query):
                    return IntentResult(
                        intent=IntentType.CLARIFICATION,
                        is_conversational=True,
                        confidence=0.85,
                        reasoning="Matched clarification request pattern.",
                        tier_used="rule",
                    )

        return None

    # ------------------------------------------------------------------
    # Tier 2 — ML Classifier
    # ------------------------------------------------------------------

    def _get_ml_router(self):
        """Lazy-load MLClassifierRouter (same instance the HybridRouter uses)."""
        if self._ml_router is None:
            from router.ml_classifier_router import MLClassifierRouter  # noqa: PLC0415
            self._ml_router = MLClassifierRouter()
        return self._ml_router

    def _ml_detect(self, query: str) -> Optional[IntentResult]:
        """
        Use the trained ML classifier to detect intent.

        If the model predicts a known technical expert with sufficient
        confidence → TECHNICAL.  Low confidence → pass to LLM tiebreaker.
        """
        try:
            ml = self._get_ml_router()
            predicted_expert, confidence, prob_dict = ml.predict_expert(query)

            # Map expert label to intent type
            intent = _ML_EXPERT_TO_INTENT.get(predicted_expert, IntentType.TECHNICAL_GENERAL)
            is_conversational = intent in CONVERSATIONAL_INTENTS

            reasoning = (
                f"ML Classifier predicted expert='{predicted_expert}' "
                f"→ intent='{intent.value}' (confidence={confidence:.2%}). "
                f"Top probabilities: {self._top3(prob_dict)}"
            )

            return IntentResult(
                intent=intent,
                is_conversational=is_conversational,
                confidence=confidence,
                reasoning=reasoning,
                tier_used="ml",
            )

        except Exception as exc:
            logger.warning("IntentDetector ML tier failed: %s", exc)
            return None

    @staticmethod
    def _top3(prob_dict: dict) -> str:
        top = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        return ", ".join(f"{k}={v:.2%}" for k, v in top)

    # ------------------------------------------------------------------
    # Tier 3 — LLM tiebreaker
    # ------------------------------------------------------------------

    def _llm_detect(self, query: str, has_prior_context: bool) -> Optional[IntentResult]:
        """
        Ask Groq to classify the intent when ML confidence is too low.

        Mirrors HybridRouter's LLM fallback design but for intent classification
        rather than expert selection.
        """
        try:
            from services.groq_client import generate_response  # noqa: PLC0415

            context_hint = (
                "Note: The user is continuing an existing conversation."
                if has_prior_context else ""
            )
            system_prompt = f"""You are an intent classifier for an AI assistant.
Classify the user's message into EXACTLY ONE of these intent labels:

{chr(10).join(f"- {label}" for label in _LLM_INTENT_LABELS)}

CONVERSATIONAL intents (no technical expertise needed):
  greeting, farewell, small_talk, general_knowledge, follow_up, clarification

TECHNICAL intents (require expert routing):
  coding, math, machine_learning, deep_learning, research, system_design, technical_general

{context_hint}

Reply with ONLY the intent label. No explanation. No punctuation.
Example outputs: greeting | coding | small_talk | follow_up"""

            raw = generate_response(
                prompt=f"Classify this message: {query}",
                system_prompt=system_prompt,
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=10,
            )

            detected_label = raw.strip().lower().split()[0].rstrip(".,!;:")
            intent = IntentType(detected_label)
            is_conversational = intent in CONVERSATIONAL_INTENTS

            return IntentResult(
                intent=intent,
                is_conversational=is_conversational,
                confidence=0.75,  # LLM tiebreaker baseline confidence
                reasoning=f"LLM tiebreaker classified as '{intent.value}' (raw: '{raw.strip()}').",
                tier_used="llm",
            )

        except (ValueError, KeyError) as parse_err:
            logger.warning("IntentDetector LLM tier: unrecognised label — %s", parse_err)
            # Try to infer from the raw text whether it's technical
            return self._heuristic_fallback(query)
        except Exception as exc:
            logger.warning("IntentDetector LLM tier failed: %s", exc)
            return None

    def _heuristic_fallback(self, query: str) -> IntentResult:
        """Last-resort: short queries are probably conversational; longer ones technical."""
        word_count = len(query.split())
        is_technical = word_count > 8
        intent = IntentType.TECHNICAL_GENERAL if is_technical else IntentType.GENERAL_KNOWLEDGE
        return IntentResult(
            intent=intent,
            is_conversational=not is_technical,
            confidence=0.50,
            reasoning=f"Heuristic fallback: word_count={word_count}, is_technical={is_technical}.",
            tier_used="heuristic",
        )
