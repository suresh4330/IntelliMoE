"""
router/decision_engine.py
-------------------------
AI Decision Engine for IntelliMoE.
Intelligently decides if a user query requires one expert or multiple experts,
generating detailed reasoning and a recommended execution order.
"""

import json
import logging
import re
from typing import Any, Dict

from router.router import ExpertName

logger = logging.getLogger(__name__)


class AIDecisionEngine:
    """
    AI Decision Engine.
    
    Runs after the Hybrid Router to decide if additional experts are needed.
    Returns structured JSON with reasoning and execution sequence.
    """

    def decide(self, query: str, primary_expert: ExpertName) -> Dict[str, Any]:
        """
        Determine if the query requires a single expert or multiple experts.

        Parameters
        ----------
        query : str
            The user question.
        primary_expert : ExpertName
            The primary expert predicted by the Hybrid Router.

        Returns
        -------
        dict
            Structured decision JSON dictionary.
        """
        from config.settings import GEMINI_MODEL_ID  # noqa: PLC0415
        from services.gemini_client import generate_response  # noqa: PLC0415

        system_prompt = (
            "You are the AI Decision Engine for IntelliMoE.\n"
            "Given a user query and the primary expert selected by the router, you must decide "
            "if other experts are required to fully address the query, or if the primary expert alone is sufficient.\n"
            "Available experts:\n"
            "  - coding (implementing algorithms, scripts, programming)\n"
            "  - math (equations, calculus, numerical computation)\n"
            "  - ml (machine learning, data engineering, tabular models)\n"
            "  - deep_learning (neural networks, computer vision, transformers)\n"
            "  - genai (prompt engineering, LLMs, AI agents)\n"
            "  - research (paper searches, RAG, academic citations)\n"
            "  - system_design (architecture design, scalability, databases)\n\n"
            "Respond ONLY with a JSON object of this structure:\n"
            "{\n"
            "  \"primary_expert\": \"coding\",\n"
            "  \"additional_experts\": [\"system_design\"],\n"
            "  \"reason\": \"Detailed reason explaining why these experts are needed for the query.\",\n"
            "  \"execution_order\": [\"system_design\", \"coding\"]\n"
            "}\n"
            "Rules:\n"
            "1. Output ONLY valid JSON. No explanations, no markdown fences.\n"
            "2. Keep the selection minimal. Only add additional experts if they are strictly necessary.\n"
            "3. The execution_order must list the sequence of experts starting from the foundation."
        )

        user_content = (
            f"Query: {query}\n"
            f"Primary Expert Selected by Router: {primary_expert.value}"
        )

        try:
            logger.info("AI Decision Engine: analyzing query and primary expert '%s'...", primary_expert.value)
            raw_output = generate_response(
                prompt=user_content,
                system_prompt=system_prompt,
                model=GEMINI_MODEL_ID,
                temperature=0.1
            )

            # Clean markdown fences if they are outputted
            clean_str = raw_output.strip()
            if "```json" in clean_str:
                match = re.search(r"```json\s*(.*?)\s*```", clean_str, re.DOTALL)
                if match:
                    clean_str = match.group(1)
            elif "```" in clean_str:
                match = re.search(r"```\s*(.*?)\s*```", clean_str, re.DOTALL)
                if match:
                    clean_str = match.group(1)

            decision = json.loads(clean_str.strip())
            
            # Post-validation checks
            if "primary_expert" not in decision:
                decision["primary_expert"] = primary_expert.value
            if "additional_experts" not in decision:
                decision["additional_experts"] = []
            if "reason" not in decision:
                decision["reason"] = "Processed successfully."
            if "execution_order" not in decision:
                decision["execution_order"] = [primary_expert.value]

            logger.info("AI Decision Engine: decision successfully generated.")
            return decision

        except Exception as e:
            logger.warning("AI Decision Engine failed: %s. Falling back to single expert.", e)
            return {
                "primary_expert": primary_expert.value,
                "additional_experts": [],
                "reason": f"Fallback triggered due to engine processing error: {e}",
                "execution_order": [primary_expert.value]
            }
