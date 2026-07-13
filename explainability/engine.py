"""
explainability/engine.py
------------------------
Explainable AI (XAI) Engine for IntelliMoE.
Aggregates routing, decision engine, planning, API, and performance metadata
to compile a unified JSON explanation.
"""

import logging
from typing import Any, Dict, List

from config.settings import GEMINI_MODEL_ID

logger = logging.getLogger(__name__)


class ExplainableEngine:
    """
    Explainable AI Engine coordinator.
    
    Compiles query routing decisions and model executions into structured inspectable JSON.
    """

    def generate_explanation(
        self,
        query: str,
        router_decision: Dict[str, Any],
        execution_plan: Dict[str, Any],
        response_time_s: float,
        tokens_estimated: int = 0
    ) -> Dict[str, Any]:
        """
        Generate structured explainability JSON details.
        """
        # 1. Gather router attributes
        ml_predicted = router_decision.get("predicted_expert", "N/A")
        confidence = float(router_decision.get("confidence", 0.0))
        fallback_used = bool(router_decision.get("fallback_used", True))

        # 2. Gather Decision Engine attributes
        primary_expert = router_decision.get("primary_expert", ml_predicted)
        additional_experts = router_decision.get("additional_experts", [])
        decision_reason = router_decision.get("reason", "Decision completed.")
        all_experts = [primary_expert] + additional_experts

        # 3. Gather Planner attributes
        steps = execution_plan.get("steps", [])
        execution_order = [s.get("expert") for s in steps] if steps else all_experts

        # 4. Map API providers and target selection justifications
        api_providers = []
        api_reasons = []
        for exp in execution_order:
            if exp:
                if str(exp).lower() in ["coding", "math"]:
                    api_providers.append("Groq (llama3-8b-8192)")
                    api_reasons.append(f"'{str(exp).upper()}' query executes faster on Groq's high-speed Llama3 inference engine.")
                else:
                    api_providers.append(f"Gemini ({GEMINI_MODEL_ID})")
                    api_reasons.append(f"'{str(exp).upper()}' query leverages Gemini's deep reasoning capabilities.")

        provider_repr = ", ".join(list(set(api_providers)))
        reason_repr = " | ".join(list(set(api_reasons)))

        # 5. Build structured explanation JSON
        explanation = {
            "query": query,
            "router": {
                "prediction": ml_predicted,
                "confidence": round(confidence, 4),
                "fallback": fallback_used
            },
            "decision_engine": {
                "primary_expert": primary_expert,
                "additional_experts": additional_experts,
                "reason": decision_reason,
                "experts": all_experts
            },
            "planner": {
                "execution_order": execution_order
            },
            "api": {
                "provider": provider_repr,
                "reason": reason_repr
            },
            "answer_quality": {
                "plans": router_decision.get("answer_quality_plans", {}),
                "reviews": router_decision.get("answer_quality_reviews", {})
            },
            "performance": {
                "response_time_ms": round(response_time_s * 1000, 2),
                "tokens": tokens_estimated
            }
        }

        logger.info("Explainable AI explanation constructed successfully.")
        return explanation
