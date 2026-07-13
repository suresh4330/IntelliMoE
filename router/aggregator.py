"""
router/aggregator.py
--------------------
Multi-Expert Collaboration Engine and Response Aggregation for IntelliMoE.

Components
──────────
  ExpertResponse        — dataclass wrapping one expert's answer + metadata.
  AggregationStrategy   — abstract base; implement to add new merge strategies.
  SectionedAggregator   — production aggregator: labelled sections + summary header.
  CollaborationEngine   — runs multiple experts in parallel (ThreadPoolExecutor)
                          and hands ExpertResponse objects to an AggregationStrategy.

Parallelism note
────────────────
  IntelliMoE uses a single shared TinyLlama model. PyTorch's GIL and CUDA
  serialization mean two concurrent model.generate() calls on the same device
  would interleave — not truly parallel for the GPU kernel, but:

    • All pre/post processing (tokenisation, decoding, prompt building) CAN
      overlap across threads.
    • A threading.Lock in models/loader.py (inference_lock) serializes the
      model.generate() calls safely while keeping the thread-pool architecture
      intact.
    • When future experts use SEPARATE models, they will run truly in parallel
      with zero code changes here.

SOLID:
  S — CollaborationEngine (execution) and Aggregator (merging) are separate.
  O — New merge strategies extend AggregationStrategy without touching Engine.
  L — Any AggregationStrategy is substitutable for another.
  I — ExpertRouter depends only on CollaborationEngine.run() and Aggregator.aggregate().
  D — CollaborationEngine depends on the BaseExpert abstraction, not concrete classes.
"""

import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from router.router import ExpertName
    from utils.memory import ConversationMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maximum worker threads in the collaboration pool.
# With a shared model, this caps concurrent inference calls.
DEFAULT_MAX_WORKERS: int = 4

# Per-expert inference timeout in seconds.
# Experts exceeding this are cancelled and their error is surfaced.
DEFAULT_TIMEOUT_SECONDS: float = 180.0


# ---------------------------------------------------------------------------
# ExpertResponse — result from one expert
# ---------------------------------------------------------------------------

@dataclass
class ExpertResponse:
    """
    The result of querying one expert.

    Attributes
    ----------
    expert_name : ExpertName
        The expert domain that produced this response.
    answer : str
        The generated answer text. Empty string on failure.
    elapsed : float
        Wall-clock seconds taken by this expert.
    success : bool
        True if the expert answered without error.
    error : Exception | None
        The exception raised if ``success`` is False.
    """
    expert_name: "ExpertName"
    answer: str
    elapsed: float
    success: bool
    error: Optional[Exception] = field(default=None, repr=False)
    prompt_tokens: int = 0
    tokens_generated: int = 0
    estimated_cost: float = 0.0
    memory_usage_mb: float = 0.0
    answer_plan: str = ""
    review_feedback: str = ""

    @property
    def expert_label(self) -> str:
        """Human-readable label, e.g. ``"System Design"``."""
        return self.expert_name.value.replace("_", " ").title()

    @property
    def elapsed_str(self) -> str:
        """Formatted elapsed time string, e.g. ``"3.2s"``."""
        return f"{self.elapsed:.1f}s"


# ---------------------------------------------------------------------------
# AggregationStrategy — abstract base (Open/Closed principle)
# ---------------------------------------------------------------------------

class AggregationStrategy(ABC):
    """
    Abstract base class for multi-expert response aggregation.

    Implement ``aggregate()`` to define how a list of ExpertResponses is
    merged into a single final answer string.

    To add a new aggregation strategy:
      1. Subclass AggregationStrategy.
      2. Implement aggregate().
      3. Pass an instance to CollaborationEngine or ExpertRouter.
    No other code changes required.
    """

    @abstractmethod
    def aggregate(
        self,
        responses: list[ExpertResponse],
        query: str,
    ) -> str:
        """
        Merge expert responses into one final answer.

        Parameters
        ----------
        responses : list[ExpertResponse]
            All ExpertResponse objects (successful and failed).
        query : str
            The original user query (may be used for context/headers).

        Returns
        -------
        str
            The merged final answer string.
        """


# ---------------------------------------------------------------------------
# SectionedAggregator — production implementation
# ---------------------------------------------------------------------------

# Expert display icons (mirrors ui/app.py EXPERT_META for consistency)
_EXPERT_ICONS: dict[str, str] = {
    "coding":        "💻",
    "math":          "📐",
    "ml":            "⚙️ ",
    "deep_learning": "🧬",
    "genai":         "✨",
    "research":      "🔬",
    "system_design": "🏗️ ",
}

_DIVIDER_HEAVY = "═" * 52
_DIVIDER_LIGHT = "─" * 52


class SectionedAggregator(AggregationStrategy):
    """
    Aggregates multi-expert responses into clearly labelled sections.

    Output format::

        ┌─────────────────────────────────────────────────────┐
        │  IntelliMoE — Multi-Expert Collaboration Report      │
        │  Query     : <original query>                        │
        │  Experts   : 💻 Coding · 🏗️  System Design · ✨ GenAI │
        │  Total time: 12.4s (parallel execution)              │
        └─────────────────────────────────────────────────────┘

        ════════════════════════════════════════════════════════
        💻 CODING EXPERT  [3.2s]
        ════════════════════════════════════════════════════════
        <answer>

        ════════════════════════════════════════════════════════
        🏗️  SYSTEM DESIGN EXPERT  [4.1s]
        ════════════════════════════════════════════════════════
        <answer>

    Parameters
    ----------
    show_errors : bool
        If True, failed expert sections display the error message.
        If False, failed experts are silently omitted. Default: True.
    show_timing : bool
        If True, each section header includes the elapsed time.
        Default: True.
    """

    def __init__(
        self,
        show_errors: bool = True,
        show_timing: bool = True,
    ) -> None:
        self._show_errors = show_errors
        self._show_timing = show_timing

    def aggregate(self, responses: list[ExpertResponse], query: str) -> str:
        """
        Merge all expert responses into a structured sectioned document.

        Single successful response → returned directly with no wrapping.
        Multiple responses → full collaboration report with sections.
        """
        successful = [r for r in responses if r.success]
        failed     = [r for r in responses if not r.success]

        # Single expert — return answer directly, no report wrapper needed.
        if len(responses) == 1 and successful:
            return successful[0].answer

        parts: list[str] = []

        # ── Summary header ──────────────────────────────────────────────
        parts.append(self._build_header(responses, query))

        # ── One section per successful expert ───────────────────────────
        for resp in successful:
            parts.append(self._build_section(resp))

        # ── Error sections (if any) ─────────────────────────────────────
        if failed and self._show_errors:
            for resp in failed:
                parts.append(self._build_error_section(resp))

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _build_header(self, responses: list[ExpertResponse], query: str) -> str:
        """Build the top-level summary header block."""
        successful  = [r for r in responses if r.success]
        total_time  = sum(r.elapsed for r in responses)
        # Wall-clock time is the longest single expert (parallel execution).
        wall_time   = max((r.elapsed for r in responses), default=0.0)

        expert_pills = " · ".join(
            f"{_EXPERT_ICONS.get(r.expert_name.value, '🤖')} {r.expert_label}"
            + (f" [{r.elapsed_str}]" if self._show_timing else "")
            for r in successful
        )

        short_query = query if len(query) <= 80 else query[:77] + "..."

        header = (
            f"{_DIVIDER_LIGHT}\n"
            f"🧠 IntelliMoE — Multi-Expert Collaboration\n"
            f"{_DIVIDER_LIGHT}\n"
            f"Query   : {short_query}\n"
            f"Experts : {expert_pills}\n"
            f"Timing  : {wall_time:.1f}s wall · {total_time:.1f}s total compute\n"
            f"{_DIVIDER_LIGHT}"
        )
        return header

    def _build_section(self, resp: ExpertResponse) -> str:
        """Build one expert answer section."""
        icon  = _EXPERT_ICONS.get(resp.expert_name.value, "🤖")
        label = resp.expert_label.upper()
        timing = f"  [{resp.elapsed_str}]" if self._show_timing else ""

        return (
            f"\n{_DIVIDER_HEAVY}\n"
            f"{icon} {label} EXPERT{timing}\n"
            f"{_DIVIDER_HEAVY}\n"
            f"{resp.answer}"
        )

    def _build_error_section(self, resp: ExpertResponse) -> str:
        """Build a failure notice section for a failed expert."""
        icon  = _EXPERT_ICONS.get(resp.expert_name.value, "🤖")
        label = resp.expert_label.upper()
        error_msg = str(resp.error) if resp.error else "Unknown error"

        return (
            f"\n{_DIVIDER_HEAVY}\n"
            f"{icon} {label} EXPERT  [FAILED]\n"
            f"{_DIVIDER_HEAVY}\n"
            f"⚠️  This expert encountered an error and could not respond.\n"
            f"   Error: {error_msg}"
        )


# ---------------------------------------------------------------------------
# CollaborationEngine — parallel expert execution
# ---------------------------------------------------------------------------

class CollaborationEngine:
    """
    Runs multiple experts concurrently and collects their ExpertResponse objects.

    Execution model:
      - A ``ThreadPoolExecutor`` submits one task per expert simultaneously.
      - Each task calls ``expert.answer(query, memory)``.
      - Results are collected as they complete (``as_completed``).
      - Experts that exceed ``timeout`` are treated as failures with a
        ``TimeoutError`` recorded in their ExpertResponse.
      - Experts that raise any exception are also recorded as failures;
        other experts continue unaffected.

    Parameters
    ----------
    max_workers : int
        Thread pool size. With a shared model, threads queue on the
        inference lock; with separate models they run truly in parallel.
    timeout : float
        Per-expert timeout in seconds.
    aggregator : AggregationStrategy | None
        Strategy used to merge responses. Defaults to SectionedAggregator.
    """

    def __init__(
        self,
        max_workers: int = DEFAULT_MAX_WORKERS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        aggregator: Optional[AggregationStrategy] = None,
    ) -> None:
        self._max_workers = max_workers
        self._timeout     = timeout
        self._aggregator  = aggregator or SectionedAggregator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        experts: dict["ExpertName", object],
        query: str,
        memory: "Optional[ConversationMemory]" = None,
    ) -> list[ExpertResponse]:
        """
        Query all given experts concurrently and return their responses.

        Parameters
        ----------
        experts : dict[ExpertName, BaseExpert]
            Mapping of expert name → expert instance to query.
        query : str
            The user query to pass to each expert.
        memory : ConversationMemory | None
            Optional conversation history for contextual responses.

        Returns
        -------
        list[ExpertResponse]
            One ExpertResponse per expert, preserving the original order
            from ``experts``. Failed experts have ``success=False``.
        """
        if not experts:
            logger.warning("CollaborationEngine.run() called with no experts.")
            return []

        expert_count = len(experts)
        logger.info(
            "CollaborationEngine: launching %d expert(s) in parallel pool "
            "(max_workers=%d, timeout=%.0fs) …",
            expert_count, self._max_workers, self._timeout,
        )

        # Map future → ExpertName so we can label results.
        future_to_name: dict[Future, "ExpertName"] = {}

        with ThreadPoolExecutor(
            max_workers=min(self._max_workers, expert_count),
            thread_name_prefix="intellimoe-expert",
        ) as pool:
            for name, expert in experts.items():
                future = pool.submit(self._call_expert, expert, query, memory)
                future_to_name[future] = name

            responses_by_name: dict["ExpertName", ExpertResponse] = {}

            try:
                for future in as_completed(future_to_name, timeout=self._timeout):
                    name = future_to_name[future]
                    try:
                        resp = future.result()
                        resp_obj = resp   # already an ExpertResponse
                    except Exception as exc:
                        logger.error(
                            "Expert '%s' raised an exception: %s: %s",
                            name.value, type(exc).__name__, exc,
                        )
                        resp_obj = ExpertResponse(
                            expert_name=name,
                            answer="",
                            elapsed=0.0,
                            success=False,
                            error=exc,
                        )
                    responses_by_name[name] = resp_obj

            except TimeoutError:
                # Some futures did not complete within the global timeout.
                for future, name in future_to_name.items():
                    if name not in responses_by_name:
                        future.cancel()
                        logger.warning(
                            "Expert '%s' timed out after %.0fs — cancelled.",
                            name.value, self._timeout,
                        )
                        responses_by_name[name] = ExpertResponse(
                            expert_name=name,
                            answer="",
                            elapsed=self._timeout,
                            success=False,
                            error=TimeoutError(
                                f"Expert '{name.value}' exceeded "
                                f"{self._timeout:.0f}s timeout."
                            ),
                        )

        # Rebuild in original submission order.
        ordered = [responses_by_name[name] for name in experts if name in responses_by_name]

        success_count = sum(1 for r in ordered if r.success)
        wall_time     = max((r.elapsed for r in ordered), default=0.0)
        total_compute = sum(r.elapsed for r in ordered)

        logger.info(
            "CollaborationEngine: %d/%d experts succeeded | "
            "wall=%.1fs, total_compute=%.1fs",
            success_count, expert_count, wall_time, total_compute,
        )
        return ordered

    def aggregate(self, responses: list[ExpertResponse], query: str) -> str:
        """
        Delegate to the configured AggregationStrategy.

        Parameters
        ----------
        responses : list[ExpertResponse]
            Responses collected by ``run()``.
        query : str
            Original user query (passed to aggregator for context).

        Returns
        -------
        str
            The merged final answer.
        """
        return self._aggregator.aggregate(responses, query)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _call_expert(
        expert,
        query: str,
        memory: "Optional[ConversationMemory]",
    ) -> ExpertResponse:
        """
        Call one expert's answer() method and wrap the result in ExpertResponse.

        This is the function submitted to the thread pool. It catches all
        exceptions so the pool can continue with other experts on failure.
        """
        from router.router import ExpertName  # noqa: PLC0415

        # Determine the expert's name from its class's prompt_name attribute.
        expert_name_str: str = getattr(expert, "prompt_name", "unknown")

        # Map prompt_name back to ExpertName enum value.
        # prompt_name "deeplearning" → ExpertName.DEEP_LEARNING
        _PROMPT_TO_ENUM = {
            "coding":        "coding",
            "math":          "math",
            "ml":            "ml",
            "deeplearning":  "deep_learning",
            "genai":         "genai",
            "research":      "research",
            "systemdesign":  "system_design",
        }
        enum_val = _PROMPT_TO_ENUM.get(expert_name_str, expert_name_str)

        try:
            expert_name = ExpertName(enum_val)
        except ValueError:
            expert_name = list(ExpertName)[0]  # fallback

        t0 = time.perf_counter()
        try:
            answer = expert.answer(query, memory=memory)
            elapsed = time.perf_counter() - t0
            
            # Extract tracking metrics from the expert instance
            prompt_tokens = getattr(expert, "last_prompt_tokens", 0)
            tokens_generated = getattr(expert, "last_tokens_generated", 0)
            
            from config.settings import PRICE_PER_1K_PROMPT_TOKENS, PRICE_PER_1K_COMPLETION_TOKENS
            from models.loader import get_memory_usage_mb
            
            cost = (
                (prompt_tokens * PRICE_PER_1K_PROMPT_TOKENS / 1000.0) +
                (tokens_generated * PRICE_PER_1K_COMPLETION_TOKENS / 1000.0)
            )
            mem_mb = get_memory_usage_mb()

            logger.info(
                "Expert '%s' completed in %.1fs (%d chars).",
                expert_name.value, elapsed, len(answer),
            )
            return ExpertResponse(
                expert_name=expert_name,
                answer=answer,
                elapsed=elapsed,
                success=True,
                prompt_tokens=prompt_tokens,
                tokens_generated=tokens_generated,
                estimated_cost=cost,
                memory_usage_mb=mem_mb,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error(
                "Expert '%s' failed after %.1fs: %s: %s",
                expert_name.value, elapsed, type(exc).__name__, exc,
            )
            from models.loader import get_memory_usage_mb
            return ExpertResponse(
                expert_name=expert_name,
                answer="",
                elapsed=elapsed,
                success=False,
                error=exc,
                memory_usage_mb=get_memory_usage_mb(),
            )
