"""
router/planner.py
-----------------
Planner Agent for IntelliMoE's Multi-Agent Collaboration System.

The Planner Agent:
  1. Receives the user query.
  2. Uses TinyLlama to generate a structured dependency execution plan (JSON format).
  3. Falls back to a deterministic heuristic-based plan if the LLM output fails
     to parse or contains invalid expert names.
  4. Models the plan as a Directed Acyclic Graph (DAG) of execution steps.

Plan Step Structure:
  - step_id: Unique integer index (1-based).
  - expert_name: The target IntelliMoE expert to query.
  - dependencies: List of step_ids whose outputs must be passed to this expert.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from router.router import ExpertName

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses representing the Execution Plan
# ---------------------------------------------------------------------------

@dataclass
class ExecutionStep:
    """A single step in the multi-agent execution pipeline."""
    step_id: int
    expert_name: ExpertName
    dependencies: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "expert": self.expert_name.value,
            "dependencies": self.dependencies,
        }


@dataclass
class ExecutionPlan:
    """A DAG representing the ordered execution plan for a query."""
    steps: list[ExecutionStep] = field(default_factory=list)
    query: str = ""
    strategy: str = "llm"  # "llm" or "heuristic"

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "query": self.query,
            "steps": [s.to_dict() for s in self.steps],
        }

    @property
    def is_empty(self) -> bool:
        return len(self.steps) == 0


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------

class PlannerAgent:
    """
    Analyzes user queries and builds execution plans using TinyLlama or heuristics.
    """

    def __init__(self, model_loader=None) -> None:
        self._model_loader = model_loader  # If none, imports lazily from models.loader

    def plan(self, query: str) -> ExecutionPlan:
        """
        Analyze ``query`` and return a dependency-aware ExecutionPlan.
        """
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        logger.info("PlannerAgent: analyzing query: '%.60s...'", query)
        
        # 1. Attempt LLM-based planning
        try:
            plan = self._plan_with_llm(query)
            if plan and not plan.is_empty:
                logger.info("PlannerAgent: LLM planning succeeded with %d steps.", len(plan.steps))
                return plan
        except Exception as exc:
            logger.warning("PlannerAgent: LLM planning failed (%s) — falling back to heuristics.", exc)

        # 2. Fallback to heuristic-based planning
        plan = self._plan_with_heuristics(query)
        logger.info("PlannerAgent: heuristic planning created %d steps.", len(plan.steps))
        return plan

    # ------------------------------------------------------------------
    # Internal: LLM Planning
    # ------------------------------------------------------------------

    def _plan_with_llm(self, query: str) -> Optional[ExecutionPlan]:
        from config.settings import GEMINI_MODEL_ID  # noqa: PLC0415
        from services.gemini_client import generate_response  # noqa: PLC0415

        # Build prompt that requests strict JSON mapping
        system_prompt = (
            "You are the Agentic Planner for IntelliMoE.\n"
            "Given a user query, you must create a structured execution plan utilizing the available experts:\n"
            "  - coding (implements algorithms, scripts, programming)\n"
            "  - math (solves equations, calculus, numerical tasks)\n"
            "  - ml (machine learning, data engineering, tabular models)\n"
            "  - deep_learning (neural networks, computer vision, transformers)\n"
            "  - genai (prompt engineering, llm integrations, agents)\n"
            "  - research (rag paper search, literature reviews, academic citations)\n"
            "  - system_design (scaling, system architecture, database design)\n\n"
            "Respond ONLY with a JSON object of this structure:\n"
            "{\n"
            "  \"steps\": [\n"
            "    {\"step_id\": 1, \"expert\": \"system_design\", \"dependencies\": []},\n"
            "    {\"step_id\": 2, \"expert\": \"coding\", \"dependencies\": [1]}\n"
            "  ]\n"
            "}\n"
            "Rules:\n"
            "1. Output ONLY valid JSON. No explanations, no markdown formatting.\n"
            "2. Keep the plan minimal (only select experts that directly help with the question).\n"
            "3. Dependencies must only reference step_ids from previous steps."
        )

        user_prompt = f"Query: {query}"

        # Generate response using Gemini client
        raw_output = generate_response(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model=GEMINI_MODEL_ID,
            temperature=0.1,  # Near-greedy for deterministic JSON
        )

        # Clean JSON markdown fences if the model included them
        clean_json_str = raw_output
        if "```json" in clean_json_str:
            match = re.search(r"```json\s*(.*?)\s*```", clean_json_str, re.DOTALL)
            if match:
                clean_json_str = match.group(1)
        elif "```" in clean_json_str:
            match = re.search(r"```\s*(.*?)\s*```", clean_json_str, re.DOTALL)
            if match:
                clean_json_str = match.group(1)

        # Parse JSON
        parsed = json.loads(clean_json_str.strip())
        steps_data = parsed.get("steps", [])
        
        steps = []
        for step_data in steps_data:
            step_id = int(step_data["step_id"])
            expert_str = step_data["expert"].strip().lower()
            
            # Map alternate strings if generated
            if expert_str == "systemdesign":
                expert_str = "system_design"
            elif expert_str == "deeplearning":
                expert_str = "deep_learning"
                
            expert_name = ExpertName(expert_str)
            deps = [int(d) for d in step_data.get("dependencies", [])]
            steps.append(ExecutionStep(step_id=step_id, expert_name=expert_name, dependencies=deps))

        # Check for circular/invalid dependencies
        for step in steps:
            for dep in step.dependencies:
                if dep >= step.step_id:
                    raise ValueError(f"Forward dependency detected in step {step.step_id} pointing to {dep}.")

        return ExecutionPlan(steps=steps, query=query, strategy="llm")

    # ------------------------------------------------------------------
    # Internal: Heuristic Planning Fallback
    # ------------------------------------------------------------------

    def _plan_with_heuristics(self, query: str) -> ExecutionPlan:
        """
        Builds a reasonable execution plan using semantic routing.
        """
        # 1. Determine activated experts based on semantic similarity
        from router.router import SemanticRoutingStrategy  # noqa: PLC0415
        strategy = SemanticRoutingStrategy(min_score_threshold=0.30)
        selected = strategy.select_experts(query)

        # 2. Sort experts based on a logical progression pipeline:
        # System Design -> Coding -> ML -> Deep Learning -> GenAI -> Research
        # This guarantees standard workflow chaining when multiple are active.
        execution_order = [
            ExpertName.SYSTEM_DESIGN,
            ExpertName.CODING,
            ExpertName.ML,
            ExpertName.DEEP_LEARNING,
            ExpertName.GENAI,
            ExpertName.RESEARCH,
            ExpertName.MATH
        ]
        
        sorted_selected = [e for e in execution_order if e in selected]
        
        # 3. Build linear steps where each step depends on the previous step
        steps = []
        for idx, expert in enumerate(sorted_selected, 1):
            deps = [idx - 1] if idx > 1 else []
            steps.append(ExecutionStep(step_id=idx, expert_name=expert, dependencies=deps))
            
        return ExecutionPlan(steps=steps, query=query, strategy="heuristic")
