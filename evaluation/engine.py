"""
evaluation/engine.py
--------------------
Orchestration engine for the AI Evaluation Framework.
Aggregates quality metrics, computes composite Overall AI and System scores.
"""

import logging
import time
from typing import Any, Dict

from evaluation.metrics import evaluate_response_quality
from evaluation.history import log_evaluation_run

logger = logging.getLogger(__name__)


class AIEvaluator:
    """
    AI Evaluator.
    
    Coordinates the execution, scoring, logging, and reporting of query transactions.
    """

    def evaluate(
        self,
        query: str,
        response: str,
        router_decision: Dict[str, Any],
        response_time_s: float
    ) -> Dict[str, Any]:
        """
        Evaluate a query transaction and log the metrics to the CSV database.
        """
        logger.info("AIEvaluator: starting transaction evaluation...")
        
        # 1. Fetch routing details
        selected_experts = router_decision.get("selected_experts", [])
        provider = router_decision.get("router_used", "LLM Router")
        
        # 2. Get prompt/completion estimated tokens
        p_toks = int(len(query.split()) * 1.33)
        c_toks = int(len(response.split()) * 1.33)
        total_tokens = p_toks + c_toks

        # 3. Call LLM-as-a-Judge for semantic quality evaluation
        scores = evaluate_response_quality(query, response, selected_experts)
        
        faithfulness = float(scores.get("faithfulness", 0.90))
        relevance = float(scores.get("relevance", 0.90))
        completeness = float(scores.get("completeness", 0.85))
        hallucination_risk = float(scores.get("hallucination_risk", 0.05))
        response_quality = float(scores.get("response_quality", 0.88))
        routing_accuracy = float(scores.get("routing_accuracy", 0.90))
        expert_selection_accuracy = float(scores.get("expert_selection_accuracy", 0.90))
        multi_agent_accuracy = float(scores.get("multi_agent_accuracy", 0.85))
        reasoning = scores.get("reasoning", "Completed successfully.")

        # 4. Calculate Overall AI Score (0-100)
        # Weighted metric combining relevance, completeness, faithfulness and quality, penalizing hallucinations
        ai_raw = (relevance * 0.3 + completeness * 0.3 + faithfulness * 0.2 + response_quality * 0.2 - hallucination_risk * 0.2)
        overall_ai_score = min(max(float(ai_raw * 100), 0.0), 100.0)

        # 5. Calculate Overall System Score (0-100)
        # Combines routing correctness, expert collaboration steps and execution efficiency
        latency_penalty = min(response_time_s * 2, 10.0)  # Max 10 pts penalty for slow queries (>5s)
        sys_raw = (routing_accuracy * 0.4 + expert_selection_accuracy * 0.3 + multi_agent_accuracy * 0.3) * 100 - latency_penalty
        overall_system_score = min(max(float(sys_raw), 0.0), 100.0)

        # 6. Log results to CSV history database
        try:
            log_evaluation_run(
                provider=provider,
                response_time=response_time_s,
                token_usage=total_tokens,
                faithfulness=faithfulness,
                relevance=relevance,
                completeness=completeness,
                hallucination_risk=hallucination_risk,
                routing_accuracy=routing_accuracy,
                expert_selection_accuracy=expert_selection_accuracy,
                multi_agent_accuracy=multi_agent_accuracy,
                response_quality=response_quality,
                overall_ai_score=overall_ai_score,
                overall_system_score=overall_system_score,
                reasoning=reasoning
            )
        except Exception as e:
            logger.error("Failed to append evaluation history: %s", e)

        # 7. Package metrics dictionary
        eval_dict = {
            "query": query,
            "response": response,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "completeness": completeness,
            "hallucination_risk": hallucination_risk,
            "response_quality": response_quality,
            "routing_accuracy": routing_accuracy,
            "expert_selection_accuracy": expert_selection_accuracy,
            "multi_agent_accuracy": multi_agent_accuracy,
            "response_time": response_time_s,
            "token_usage": total_tokens,
            "overall_ai_score": overall_ai_score,
            "overall_system_score": overall_system_score,
            "provider": provider,
            "reasoning": reasoning
        }

        # Dynamically compile the markdown report
        try:
            from evaluation.report import generate_evaluation_report  # noqa: PLC0415
            generate_evaluation_report(eval_dict)
        except Exception as e:
            logger.warning("Failed to generate evaluation report: %s", e)

        return eval_dict
