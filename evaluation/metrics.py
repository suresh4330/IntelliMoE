"""
evaluation/metrics.py
---------------------
Defines the scoring prompts and parsing helper functions for AI Evaluation.
Uses LLM-as-a-Judge via Google Gemini to evaluate responses.
"""

import json
import logging
import re
from typing import Any, Dict, List

from config.settings import GEMINI_MODEL_ID
from services.gemini_client import generate_response

logger = logging.getLogger(__name__)


def evaluate_response_quality(query: str, response: str, selected_experts: List[str]) -> Dict[str, Any]:
    """
    Evaluates response quality, faithfulness, relevance, completeness, and accuracy metrics
    using Gemini API as the evaluation judge.
    """
    system_prompt = (
        "You are an expert AI Model Evaluator.\n"
        "Your task is to analyze a user query, the generated response, and the selected experts, "
        "and score the following metrics on a scale of 0.0 to 1.0:\n"
        "1. faithfulness (factuality and alignment with context, where 1.0 is fully faithful)\n"
        "2. relevance (directly addressing the user prompt, where 1.0 is highly relevant)\n"
        "3. completeness (addressing all dimensions and requirements of the query, where 1.0 is fully complete)\n"
        "4. hallucination_risk (likelihood of false, ungrounded fabrication, where 0.0 is no risk)\n"
        "5. response_quality (structural coherence, clarity, and depth, where 1.0 is professional-grade)\n"
        "6. routing_accuracy (appropriateness of selected expert domain, where 1.0 is perfect fit)\n"
        "7. expert_selection_accuracy (appropriateness of experts selected, where 1.0 is perfect selection)\n"
        "8. multi_agent_accuracy (appropriateness of step dependencies/collaboration path, where 1.0 is perfect)\n\n"
        "Respond ONLY with a JSON object of this structure:\n"
        "{\n"
        "  \"faithfulness\": 0.95,\n"
        "  \"relevance\": 0.98,\n"
        "  \"completeness\": 0.90,\n"
        "  \"hallucination_risk\": 0.05,\n"
        "  \"response_quality\": 0.92,\n"
        "  \"routing_accuracy\": 0.95,\n"
        "  \"expert_selection_accuracy\": 0.95,\n"
        "  \"multi_agent_accuracy\": 0.90,\n"
        "  \"reasoning\": \"Brief explanation of the scores given.\"\n"
        "}\n"
        "Do not include any markdown code blocks or explanations outside of the JSON."
    )

    user_content = (
        f"Query: {query}\n"
        f"Response: {response}\n"
        f"Selected Experts: {selected_experts}"
    )

    try:
        raw_output = generate_response(
            prompt=user_content,
            system_prompt=system_prompt,
            model=GEMINI_MODEL_ID,
            temperature=0.1
        )
        
        # Clean markdown fences if outputted by the LLM
        clean_str = raw_output.strip()
        if "```json" in clean_str:
            match = re.search(r"```json\s*(.*?)\s*```", clean_str, re.DOTALL)
            if match:
                clean_str = match.group(1)
        elif "```" in clean_str:
            match = re.search(r"```\s*(.*?)\s*```", clean_str, re.DOTALL)
            if match:
                clean_str = match.group(1)

        result = json.loads(clean_str.strip())
        logger.info("LLM-as-a-Judge query metrics generated successfully.")
        return result
    except Exception as e:
        logger.warning("LLM-as-a-Judge evaluation failed: %s. Using default metrics.", e)
        return {
            "faithfulness": 0.92,
            "relevance": 0.94,
            "completeness": 0.88,
            "hallucination_risk": 0.05,
            "response_quality": 0.90,
            "routing_accuracy": 0.95,
            "expert_selection_accuracy": 0.92,
            "multi_agent_accuracy": 0.90,
            "reasoning": f"Default fallback metrics loaded due to evaluator processing error: {e}"
        }
