"""
router/orchestrator.py
----------------------
Agent Orchestrator for IntelliMoE's Multi-Agent Collaboration System.

The Orchestrator:
  1. Receives the user query and an ExecutionPlan from the Planner.
  2. Resolves execution stages based on topological dependency levels.
     Steps that do not depend on each other are executed in parallel.
  3. Context Chaining: For steps with dependencies, the orchestrator retrieves
     the outputs of those dependency steps and prepends them to the query
     passed to the downstream expert.
  4. Records a detailed Collaboration Timeline for display in the UI.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from router.router import ExpertName
from router.planner import ExecutionPlan, ExecutionStep
from router.aggregator import ExpertResponse

if TYPE_CHECKING:
    from utils.memory import ConversationMemory

logger = logging.getLogger(__name__)


@dataclass
class TimelineEvent:
    """A log event in the multi-agent collaboration timeline."""
    timestamp: float          # Time elapsed since query start
    message: str              # Description of the event (e.g. "Step 1 started")
    status: str = "info"      # "info", "success", "warning", "error"
    expert: Optional[str] = None


class AgentOrchestrator:
    """
    Executes a multi-agent ExecutionPlan, handles dependency context chaining,
    runs independent steps in parallel, and logs timeline events.
    """

    def __init__(self, max_workers: int = 4, timeout: float = 180.0) -> None:
        self._max_workers = max_workers
        self._timeout     = timeout

    def execute(
        self,
        plan: ExecutionPlan,
        experts_registry: dict[ExpertName, object],
        memory: "Optional[ConversationMemory]" = None,
        image_path: Optional[str] = None,
    ) -> tuple[list[ExpertResponse], list[TimelineEvent]]:
        """
        Execute the plan steps, chaining dependencies and returning outputs + timeline.
        """
        if plan.is_empty:
            logger.warning("AgentOrchestrator: executed empty plan.")
            return [], []

        t_start = time.perf_counter()
        timeline: list[TimelineEvent] = []

        def log_event(msg: str, status: str = "info", expert: Optional[str] = None):
            elapsed = time.perf_counter() - t_start
            event = TimelineEvent(timestamp=elapsed, message=msg, status=status, expert=expert)
            timeline.append(event)
            logger.info("Orchestrator [%.2fs]: %s", elapsed, msg)

        log_event(f"Starting Multi-Agent Collaboration Plan (Strategy: {plan.strategy})")

        # Map step_id -> ExecutionStep
        id_to_step = {s.step_id: s for s in plan.steps}

        # Store text outputs and raw response objects
        step_outputs: dict[int, str] = {}
        expert_responses: dict[int, ExpertResponse] = {}

        # Resolve steps into execution levels based on dependency layers
        # Layer 0: Steps with no dependencies (can all run in parallel)
        # Layer N: Steps depending only on layers < N (can run in parallel once dependencies finish)
        resolved_steps = set()
        layers: list[list[ExecutionStep]] = []

        remaining = list(plan.steps)
        while remaining:
            current_layer = []
            for step in list(remaining):
                # If all dependencies are already resolved in previous layers
                if all(dep in resolved_steps for dep in step.dependencies):
                    current_layer.append(step)
                    remaining.remove(step)
            
            if not current_layer:
                # Cycle fallback: if there is a dependency cycle, resolve linearly
                logger.warning("AgentOrchestrator: Cycle or unresolved dependency detected. Resolving linearly.")
                layers.append([remaining.pop(0)])
                continue
                
            layers.append(current_layer)
            for step in current_layer:
                resolved_steps.add(step.step_id)

        log_event(f"Resolved execution plan into {len(layers)} sequential layer(s).")

        # Execute layer by layer
        for layer_idx, layer in enumerate(layers, 1):
            layer_experts = [s.expert_name.value.replace('_', ' ').title() for s in layer]
            log_event(f"Executing Layer {layer_idx}/{len(layers)}: {', '.join(layer_experts)}")

            # If layer has multiple steps, run them in parallel
            with ThreadPoolExecutor(
                max_workers=min(self._max_workers, len(layer)),
                thread_name_prefix="intellimoe-orchestrator",
            ) as pool:
                futures_to_step: dict[Future, ExecutionStep] = {}

                for step in layer:
                    # 1. Build context from dependencies
                    context_chain = ""
                    if step.dependencies:
                        log_event(f"Chaining context from steps {step.dependencies} to expert '{step.expert_name.value}'")
                        for dep_id in step.dependencies:
                            dep_step = id_to_step[dep_id]
                            dep_out  = step_outputs.get(dep_id, "")
                            context_chain += (
                                f"\n=== Context from {dep_step.expert_name.value.upper()} (Step {dep_id}) ===\n"
                                f"{dep_out}\n"
                            )

                    # Augment user query with dependent context
                    augmented_query = plan.query
                    if context_chain:
                        augmented_query = (
                            f"{context_chain.strip()}\n\n"
                            f"Using the context above, address the user request:\n"
                            f"{plan.query}"
                        )

                    expert_instance = experts_registry.get(step.expert_name)
                    if not expert_instance:
                        raise NotImplementedError(f"Expert '{step.expert_name.value}' not found in registry.")

                    # Submit execution task
                    log_event(f"Launching expert '{step.expert_name.value}' ...", expert=step.expert_name.value)
                    future = pool.submit(
                        self._run_expert_step,
                        step.step_id,
                        expert_instance,
                        step.expert_name,
                        augmented_query,
                        memory,
                        image_path
                    )
                    futures_to_step[future] = step

                # Gather results for current layer
                for future in futures_to_step:
                    step = futures_to_step[future]
                    try:
                        resp: ExpertResponse = future.result(timeout=self._timeout)
                        expert_responses[step.step_id] = resp
                        step_outputs[step.step_id] = resp.answer if resp.success else ""
                        
                        if resp.success:
                            log_event(
                                f"Expert '{step.expert_name.value}' completed successfully in {resp.elapsed:.1f}s.",
                                status="success",
                                expert=step.expert_name.value
                            )
                        else:
                            log_event(
                                f"Expert '{step.expert_name.value}' failed: {resp.error}",
                                status="error",
                                expert=step.expert_name.value
                            )
                    except Exception as exc:
                        log_event(
                            f"Expert '{step.expert_name.value}' raised critical error: {exc}",
                            status="error",
                            expert=step.expert_name.value
                        )
                        expert_responses[step.step_id] = ExpertResponse(
                            expert_name=step.expert_name,
                            answer="",
                            elapsed=0.0,
                            success=False,
                            error=exc
                        )
                        step_outputs[step.step_id] = ""

        total_elapsed = time.perf_counter() - t_start
        log_event(f"Multi-Agent Plan execution completed in {total_elapsed:.1f}s.", status="success")

        # Re-sort responses in order of step_id
        ordered_responses = [expert_responses[s.step_id] for s in plan.steps]
        return ordered_responses, timeline

    @staticmethod
    def _run_expert_step(
        step_id: int,
        expert,
        expert_name: ExpertName,
        query: str,
        memory: "Optional[ConversationMemory]",
        image_path: Optional[str] = None
    ) -> ExpertResponse:
        """Runs a single expert execution task, measuring latency and VRAM/RAM metrics."""
        from config.settings import PRICE_PER_1K_PROMPT_TOKENS, PRICE_PER_1K_COMPLETION_TOKENS  # noqa: PLC0415
        from models.loader import get_memory_usage_mb  # noqa: PLC0415
        from utils.evaluation import EvaluationEngine, ExpertRunMetric  # noqa: PLC0415
        import psutil  # noqa: PLC0415

        t0 = time.perf_counter()
        proc = psutil.Process()
        try:
            proc.cpu_percent(interval=None)
        except Exception:
            pass

        try:
            from router.quality_engine import AnswerQualityEngine  # noqa: PLC0415
            quality_engine = AnswerQualityEngine()
            
            logger.info("AgentOrchestrator: executing Answer Quality Engine for expert '%s'", expert_name.value)
            quality_res = quality_engine.generate_quality_response(
                expert=expert,
                expert_name=expert_name.value,
                query=query,
                memory=memory,
                image_path=image_path if expert_name.value == "vision" else None
            )
            answer = quality_res["answer"]
            answer_plan = quality_res["plan"]
            review_feedback = quality_res["review"]
            
            elapsed = time.perf_counter() - t0
            
            prompt_tokens = getattr(expert, "last_prompt_tokens", 0)
            tokens_generated = getattr(expert, "last_tokens_generated", 0)
            cost = (
                (prompt_tokens * PRICE_PER_1K_PROMPT_TOKENS / 1000.0) +
                (tokens_generated * PRICE_PER_1K_COMPLETION_TOKENS / 1000.0)
            )
            mem_mb = get_memory_usage_mb()
            
            try:
                cpu_pct = proc.cpu_percent(interval=None)
            except Exception:
                cpu_pct = 0.0
 
            # Log to SQLite
            db = EvaluationEngine()
            db.log_run(ExpertRunMetric(
                expert_name=expert_name.value,
                query=query,
                response_time=elapsed,
                prompt_tokens=prompt_tokens,
                completion_tokens=tokens_generated,
                memory_usage_mb=mem_mb,
                cpu_usage_pct=cpu_pct,
                success=True
            ))
 
            return ExpertResponse(
                expert_name=expert_name,
                answer=answer,
                elapsed=elapsed,
                success=True,
                prompt_tokens=prompt_tokens,
                tokens_generated=tokens_generated,
                estimated_cost=cost,
                memory_usage_mb=mem_mb,
                answer_plan=answer_plan,
                review_feedback=review_feedback
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            mem_mb = get_memory_usage_mb()
            
            try:
                cpu_pct = proc.cpu_percent(interval=None)
            except Exception:
                cpu_pct = 0.0

            # Log to SQLite
            db = EvaluationEngine()
            db.log_run(ExpertRunMetric(
                expert_name=expert_name.value,
                query=query,
                response_time=elapsed,
                prompt_tokens=0,
                completion_tokens=0,
                memory_usage_mb=mem_mb,
                cpu_usage_pct=cpu_pct,
                success=False,
                error=str(exc)
            ))

            return ExpertResponse(
                expert_name=expert_name,
                answer="",
                elapsed=elapsed,
                success=False,
                error=exc,
                memory_usage_mb=mem_mb
            )
