"""
router/hybrid_router.py
-----------------------
Hybrid Router combining the ML Intent Classifier with the LLM Router.
Conforms to RoutingStrategy to ensure clean design pattern alignment.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from config.settings import ML_ROUTING_CONFIDENCE_THRESHOLD
from router.ml_classifier_router import MLClassifierRouter
from router.router import ExpertName, RoutingStrategy

logger = logging.getLogger(__name__)


class HybridRouter(RoutingStrategy):
    """
    Hybrid Router Strategy.
    
    Combines the fast, low-cost ML Intent Classifier with the LLM Router.
    If prediction confidence falls below the threshold, it falls back to LLM routing.
    """

    def __init__(
        self,
        ml_router: Optional[MLClassifierRouter] = None,
        llm_router: Optional[RoutingStrategy] = None,
    ) -> None:
        self._ml_router: MLClassifierRouter = ml_router or MLClassifierRouter()
        
        # Lazy load LLM strategy to prevent unnecessary circular imports
        self._llm_router: Optional[RoutingStrategy] = llm_router
        
        # Last decision metadata to populate diagnostics telemetry
        self.last_decision: Dict = {}

    def _get_llm_router(self) -> RoutingStrategy:
        """Lazily initialize the fallback LLM routing strategy."""
        if self._llm_router is None:
            from router.llm_router import LLMRoutingStrategy  # noqa: PLC0415
            self._llm_router = LLMRoutingStrategy()
        return self._llm_router

    def select_expert(self, query: str) -> ExpertName:
        """
        Select a single expert using ML classifier if confidence is high,
        otherwise fall back to the LLM Router.
        """
        experts = self.select_experts(query)
        if experts:
            return experts[0]
        return ExpertName.CODING

    def select_experts(self, query: str) -> List[ExpertName]:
        """
        Select experts using ML classifier with LLM fallback.
        """
        query = query.strip()
        if not query:
            return []

        ml_predicted = None
        ml_confidence = 0.0
        ml_success = False
        fallback_used = True
        strategy_name = "LLM Router"

        # 1. Query the ML Intent Classifier
        try:
            predicted_str, confidence, _ = self._ml_router.predict_expert(query)
            ml_predicted = predicted_str
            ml_confidence = confidence
            ml_success = True
        except Exception as exc:
            logger.warning("ML routing classification failed: %s. Falling back to LLM Router.", exc)

        # 2. Apply threshold routing decision
        if ml_success and ml_confidence >= ML_ROUTING_CONFIDENCE_THRESHOLD:
            fallback_used = False
            strategy_name = "ML Intent Classifier"
            selected_experts = [ExpertName(ml_predicted)]
            logger.info(
                "HybridRouter: routed via ML Classifier to '%s' (confidence: %.2f%%).",
                ml_predicted,
                ml_confidence * 100,
            )
        else:
            # Fall back to LLM Router
            logger.info("HybridRouter: falling back to LLM Router (confidence below threshold or classification error).")
            llm_strategy = self._get_llm_router()
            selected_experts = llm_strategy.select_experts(query)
            strategy_name = "LLM Router"

        # 3. Store routing metadata
        self.last_decision = {
            "predicted_expert": ml_predicted,
            "confidence": ml_confidence,
            "routing_strategy": strategy_name,
            "fallback_used": fallback_used,
            "timestamp": datetime.now().isoformat()
        }

        return selected_experts
