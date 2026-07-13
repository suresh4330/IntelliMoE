"""
router/router.py
----------------
Expert Router for IntelliMoE — directs user queries to one or more experts.

Architecture (three-layer design):
───────────────────────────────────
  1. RoutingStrategy (abstract)
       └── KeywordRoutingStrategy   ← keyword scoring
       └── LLMRoutingStrategy       ← TinyLlama intent classification (default)

  2. CollaborationEngine (aggregator.py)
       └── Runs selected experts in a ThreadPoolExecutor.
       └── Collects ExpertResponse objects (with timing + error handling).
       └── Delegates to AggregationStrategy for final answer merging.

  3. ExpertRouter (this module)
       └── Public entry point: route(query, memory) → str.
       └── Owns the expert registry (lazy-loaded on first call).
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Optional

from config.settings import MIN_SCORE_THRESHOLD, MAX_EXPERTS

if TYPE_CHECKING:
    from utils.memory import ConversationMemory

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ExpertName — canonical identifiers for every expert
# ---------------------------------------------------------------------------

class ExpertName(str, Enum):
    """Canonical names for all supported expert domains."""
    CODING         = "coding"
    MATH           = "math"
    ML             = "ml"
    DEEP_LEARNING  = "deep_learning"
    GENAI          = "genai"
    RESEARCH       = "research"
    SYSTEM_DESIGN  = "system_design"
    VISION         = "vision"


# ---------------------------------------------------------------------------
# Keyword map
# ---------------------------------------------------------------------------

KEYWORD_MAP: dict[ExpertName, list[str]] = {
    ExpertName.CODING: [
        "code", "coding", "program", "programming", "function", "class",
        "debug", "bug", "error", "syntax", "algorithm", "data structure",
        "python", "javascript", "typescript", "java", "c++", "rust", "go",
        "api", "rest", "graphql", "git", "refactor", "test", "unit test",
        "loop", "recursion", "array", "string", "list", "dict", "object",
        "compiler", "interpreter", "ide", "library", "framework", "package",
        "implement", "build", "develop", "write", "create", "application",
    ],
    ExpertName.MATH: [
        "math", "mathematics", "equation", "formula", "calculus", "algebra",
        "geometry", "trigonometry", "probability", "statistics", "matrix",
        "vector", "integral", "derivative", "theorem", "proof", "number theory",
        "combinatorics", "set theory", "linear algebra", "differential",
        "polynomial", "prime", "factorial", "logarithm", "exponential",
        "optimise", "optimize", "gradient", "loss",
    ],
    ExpertName.ML: [
        "machine learning", "ml", "supervised", "unsupervised", "classification",
        "regression", "clustering", "decision tree", "random forest", "svm",
        "support vector", "knn", "k-nearest", "naive bayes", "feature",
        "training", "validation", "overfitting", "underfitting", "cross-validation",
        "gradient descent", "loss function", "accuracy", "precision", "recall",
        "f1", "scikit", "sklearn", "xgboost", "lightgbm", "catboost",
        "predict", "prediction", "model", "dataset", "data pipeline",
        "anomaly detection", "recommendation", "patient", "diagnosis",
    ],
    ExpertName.DEEP_LEARNING: [
        "deep learning", "neural network", "cnn", "rnn", "lstm", "gru",
        "transformer", "attention", "backpropagation", "activation function",
        "relu", "sigmoid", "softmax", "dropout", "batch norm", "layer",
        "epoch", "batch size", "pytorch", "tensorflow", "keras", "autoencoder",
        "gan", "diffusion", "embedding", "token", "fine-tune", "pretrained",
        "computer vision", "image recognition", "nlp",
    ],
    ExpertName.GENAI: [
        "generative ai", "genai", "llm", "large language model", "gpt",
        "chatgpt", "claude", "gemini", "llama", "mistral", "falcon", "bloom",
        "prompt engineering", "prompt", "rag", "retrieval augmented",
        "vector database", "langchain", "llamaindex", "agent", "tool use",
        "chain of thought", "few-shot", "zero-shot", "instruction tuning",
        "rlhf", "hallucination", "inference", "quantization", "gguf",
        "ai assistant", "chatbot", "conversational", "intelligent",
    ],
    ExpertName.RESEARCH: [
        "research", "paper", "arxiv", "journal", "publication", "study",
        "survey", "literature", "review", "hypothesis", "experiment",
        "dataset", "benchmark", "sota", "state of the art", "baseline",
        "ablation", "citation", "methodology", "findings", "results",
        "academic", "conference", "workshop", "peer review",
    ],
    ExpertName.SYSTEM_DESIGN: [
        "system design", "architecture", "scalability", "distributed",
        "microservice", "monolith", "load balancer", "cache", "caching",
        "database", "sql", "nosql", "message queue", "kafka", "rabbitmq",
        "cdn", "api gateway", "rate limiting", "sharding", "replication",
        "cap theorem", "consistency", "availability", "latency", "throughput",
        "design pattern", "solid", "event driven", "pub sub", "grpc",
        "system", "platform", "infrastructure", "hospital", "management",
        "enterprise", "service", "deploy", "cloud", "aws", "azure", "gcp",
    ],
}

# Fallback when no expert scores above the threshold.
DEFAULT_EXPERT: ExpertName = ExpertName.CODING

# MIN_SCORE_THRESHOLD and MAX_EXPERTS are imported from config.settings.


# ---------------------------------------------------------------------------
# Abstract routing strategy
# ---------------------------------------------------------------------------

class RoutingStrategy(ABC):
    """
    Abstract base class for query routing strategies.

    Subclass this to add new routing logic (e.g. LLM-based).
    Implement both ``select_expert`` (single) and ``select_experts`` (multi).
    """

    @abstractmethod
    def select_expert(self, query: str) -> ExpertName:
        """Return the single best expert for the query (simple queries)."""

    @abstractmethod
    def select_experts(self, query: str) -> list[ExpertName]:
        """
        Return all qualifying experts for the query (complex queries).
        Returns a list of one or more ExpertName values, ordered by score.
        """


# ---------------------------------------------------------------------------
# Keyword-based routing strategy
# ---------------------------------------------------------------------------

class KeywordRoutingStrategy(RoutingStrategy):
    """
    Legacy keyword-based routing strategy.
    Counts occurrence of expert keywords in the query to score.
    """

    def __init__(
        self,
        default_expert: ExpertName = DEFAULT_EXPERT,
        min_score_threshold: float = 0.30,
        max_experts: int = MAX_EXPERTS,
    ) -> None:
        self._default_expert = default_expert
        self._min_score_threshold = min_score_threshold
        self._max_experts = max_experts

    def select_expert(self, query: str) -> ExpertName:
        """Return the single top-ranked expert based on keyword match count."""
        experts = self.select_experts(query)
        return experts[0] if experts else self._default_expert

    def select_experts(self, query: str) -> list[ExpertName]:
        """
        Count occurrences of keywords for each expert domain in the query,
        and return matching experts above the threshold.
        """
        query_lower = query.lower()
        scores = {}
        for name, keywords in KEYWORD_MAP.items():
            score = 0
            for kw in keywords:
                score += query_lower.count(kw)
            if score > 0:
                scores[name] = float(score)

        if not scores:
            return [self._default_expert]

        # Sort by score descending
        sorted_experts = sorted(scores.keys(), key=lambda n: scores[n], reverse=True)
        return sorted_experts[:self._max_experts]


# ---------------------------------------------------------------------------
# Semantic Descriptions Map
# ---------------------------------------------------------------------------

DEFAULT_DESCRIPTIONS: dict[ExpertName, str] = {
    ExpertName.CODING: "Expert in writing software programs, code, debugging, refactoring, compiling, scripting, and implementing algorithms in Python, C++, Java, Rust, Javascript, Go, and other programming languages.",
    ExpertName.MATH: "Expert in solving mathematical equations, formulas, calculus, integration, derivation, algebra, probability, geometry, prime numbers, linear algebra, proofs, and numeric optimization.",
    ExpertName.ML: "Expert in machine learning workflows, training classification and regression models, feature engineering, validation, tabular data, scikit-learn, XGBoost, anomaly detection, predictive analysis, and medical patient diagnosis models.",
    ExpertName.DEEP_LEARNING: "Expert in deep neural networks, computer vision, image classification, transformers, self-attention mechanisms, PyTorch, TensorFlow, backpropagation, CNNs, RNNs, training loops, and activation functions.",
    ExpertName.GENAI: "Expert in generative AI, large language models (LLMs), prompt engineering, retrieval-augmented generation (RAG), vector databases, AI agents, few-shot prompts, LangChain, LlamaIndex, conversational chatbots, and hallucination reduction.",
    ExpertName.RESEARCH: "Expert in academic research paper analysis, citation reviews, literature survey, finding state-of-the-art baselines, ablation studies, scientific benchmarks, and arXiv peer review studies.",
    ExpertName.SYSTEM_DESIGN: "Expert in software architecture, system design, scalability, distributed microservices, message queues like Kafka or RabbitMQ, database schema replication, sharding, load balancers, rate limiting, and CAP theorem consistency.",
    ExpertName.VISION: "Expert in computer vision, image processing, reading screenshots, explaining charts and plotted graphs, understanding UML diagrams, sequence diagrams, flowcharts, schemas, layouts, UI designs, and drawing structures from visual image assets.",
}

# Fallback when no expert scores above the threshold.
DEFAULT_EXPERT: ExpertName = ExpertName.CODING


# ---------------------------------------------------------------------------
# Semantic routing strategy
# ---------------------------------------------------------------------------

class SemanticRoutingStrategy(RoutingStrategy):
    """
    Routes queries by computing cosine similarity between user query embeddings
    and semantic descriptions of each expert's domain.

    Supports future experts dynamically without code modification by loading
    their system prompt files or class docstrings to build the semantic description.
    """

    def __init__(
        self,
        default_expert: ExpertName = DEFAULT_EXPERT,
        min_score_threshold: float = 0.30,  # Cosine similarity threshold
        max_experts: int = MAX_EXPERTS,
    ) -> None:
        self._default_expert = default_expert
        self._min_score_threshold = min_score_threshold
        self._max_experts = max_experts
        self._cached_description_embeddings: dict[ExpertName, list[float]] = {}

    def select_expert(self, query: str) -> ExpertName:
        """Return the single highest-matching expert (confidence) for the query."""
        scores = self._score_all(query)
        best = max(scores, key=lambda n: scores[n])
        if scores[best] < self._min_score_threshold:
            logger.info("No semantic match above threshold — defaulting to '%s'.", self._default_expert)
            return self._default_expert
        logger.info("Semantic expert selected: '%s' (similarity=%.4f).", best, scores[best])
        return best

    def select_experts(self, query: str) -> list[ExpertName]:
        """
        Return all experts meeting the similarity threshold, sorted by confidence.
        """
        scores = self._score_all(query)
        
        # Sort experts by similarity descending
        sorted_experts = sorted(scores.keys(), key=lambda n: scores[n], reverse=True)
        
        # Filter by threshold
        qualified = [e for e in sorted_experts if scores[e] >= self._min_score_threshold]

        if not qualified:
            # Fallback to the single highest-matching expert
            best = sorted_experts[0]
            logger.info(
                "No expert met similarity threshold=%.2f — defaulting to top matching expert '%s' (similarity=%.4f).",
                self._min_score_threshold, best, scores[best]
            )
            return [best]

        selected = qualified[: self._max_experts]
        logger.info(
            "Semantic routing selection -> %s (similarities: %s)",
            [e.value for e in selected],
            {e.value: f"{scores[e]:.4f}" for e in selected}
        )
        return selected

    def _score_all(self, query: str) -> dict[ExpertName, float]:
        """Compute semantic similarities between query and all expert descriptions."""
        import numpy as np  # noqa: PLC0415
        from utils.vector_store import _get_embed_model  # noqa: PLC0415

        embed_model = _get_embed_model()
        query_embedding = embed_model.encode([query], normalize_embeddings=True)[0]

        scores = {}
        for name in ExpertName:
            desc = self._get_expert_description(name)
            
            # Embed and cache description if not already cached
            if name not in self._cached_description_embeddings:
                desc_emb = embed_model.encode([desc], normalize_embeddings=True)[0]
                self._cached_description_embeddings[name] = desc_emb.tolist()
            
            desc_emb = np.array(self._cached_description_embeddings[name])
            
            # Cosine similarity is dot product since embeddings are normalized
            sim = float(np.dot(query_embedding, desc_emb))
            scores[name] = sim

        return scores

    def _get_expert_description(self, name: ExpertName) -> str:
        """
        Retrieve description for an expert. Falls back to system prompt files
        to support any future registered experts automatically.
        """
        # 1. Use hardcoded defaults if available
        if name in DEFAULT_DESCRIPTIONS:
            return DEFAULT_DESCRIPTIONS[name]

        # 2. Fallback to system prompt text dynamically
        try:
            from utils.prompt_manager import PromptManager  # noqa: PLC0415
            pm = PromptManager()
            sys_prompt = pm.get_prompt(name.value)
            if sys_prompt:
                # Use first line or two of system prompt
                lines = [l.strip() for l in sys_prompt.split("\n") if l.strip()]
                if lines:
                    return " ".join(lines[:2])
        except Exception as exc:
            logger.warning("Could not load prompt for '%s' for semantic routing: %s", name.value, exc)
            
        return f"Expert specialized in {name.value.replace('_', ' ')}."


# ---------------------------------------------------------------------------
# Response combiner
# ---------------------------------------------------------------------------

class ResponseCombiner:
    """
    Merges answers from multiple experts into one structured final answer.

    Format
    ------
    Each expert's contribution is wrapped in a titled section::

        ════════════════════════════════════════
        💻 CODING EXPERT
        ════════════════════════════════════════
        <expert answer here>

        ════════════════════════════════════════
        🏗️  SYSTEM DESIGN EXPERT
        ════════════════════════════════════════
        <expert answer here>
    """

    # Icon mapping reused from the UI layer for consistency.
    _ICONS: dict[ExpertName, str] = {
        ExpertName.CODING:        "💻",
        ExpertName.MATH:          "📐",
        ExpertName.ML:            "⚙️ ",
        ExpertName.DEEP_LEARNING: "🧬",
        ExpertName.GENAI:         "✨",
        ExpertName.RESEARCH:      "🔬",
        ExpertName.SYSTEM_DESIGN: "🏗️ ",
    }

    _DIVIDER = "═" * 48

    def combine(
        self,
        responses: dict[ExpertName, str],
        query: str,
    ) -> str:
        """
        Combine expert responses into one structured string.

        Parameters
        ----------
        responses : dict[ExpertName, str]
            Mapping of each activated expert to their generated answer.
        query : str
            The original user query (used in the preamble).

        Returns
        -------
        str
            A single combined answer with labelled expert sections.
        """
        if len(responses) == 1:
            # Single expert — return its answer directly, no wrapping needed.
            return next(iter(responses.values()))

        parts: list[str] = []

        # Preamble
        expert_labels = ", ".join(
            f"{self._ICONS.get(name, '🤖')} {name.value.replace('_', ' ').title()}"
            for name in responses
        )
        preamble = (
            f"This query was analysed by {len(responses)} experts: "
            f"{expert_labels}.\n"
            f"Each section below contains that expert's perspective.\n"
        )
        parts.append(preamble)

        # One section per expert
        for expert_name, answer in responses.items():
            icon = self._ICONS.get(expert_name, "🤖")
            label = expert_name.value.replace("_", " ").upper()
            section = (
                f"\n{self._DIVIDER}\n"
                f"{icon} {label} EXPERT\n"
                f"{self._DIVIDER}\n"
                f"{answer}"
            )
            parts.append(section)

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Expert Registry
# ---------------------------------------------------------------------------

def _build_expert_registry() -> dict[ExpertName, object]:
    """
    Lazily import and instantiate all expert classes.
    Each expert manages its own model singleton internally.
    """
    from experts.coding        import CodingExpert        # noqa: PLC0415
    from experts.math          import MathExpert           # noqa: PLC0415
    from experts.ml            import MLExpert             # noqa: PLC0415
    from experts.deeplearning  import DeepLearningExpert   # noqa: PLC0415
    from experts.genai         import GenAIExpert          # noqa: PLC0415
    from experts.research      import ResearchExpert       # noqa: PLC0415
    from experts.system_design import SystemDesignExpert   # noqa: PLC0415
    from experts.vision        import VisionExpert         # noqa: PLC0415

    return {
        ExpertName.CODING:        CodingExpert(),
        ExpertName.MATH:          MathExpert(),
        ExpertName.ML:            MLExpert(),
        ExpertName.DEEP_LEARNING: DeepLearningExpert(),
        ExpertName.GENAI:         GenAIExpert(),
        ExpertName.RESEARCH:      ResearchExpert(),
        ExpertName.SYSTEM_DESIGN: SystemDesignExpert(),
        ExpertName.VISION:        VisionExpert(),
    }


# ---------------------------------------------------------------------------
# ExpertRouter — public interface
# ---------------------------------------------------------------------------

class ExpertRouter:
    """
    Central router that maps user queries to one or more experts and returns
    a (possibly combined) generated answer.

    Simple query   → one expert answers  → answer returned directly.
    Complex query  → multiple experts answer → answers combined into sections.

    Parameters
    ----------
    strategy : RoutingStrategy | None
        The routing strategy to use. Defaults to ``LLMRoutingStrategy``
        (TinyLlama intent classification with JSON output). Pass
        ``KeywordRoutingStrategy()`` to use the legacy keyword-based router.
    combiner : ResponseCombiner | None
        The combiner used to merge multi-expert responses.

    Examples
    --------
    >>> router = ExpertRouter()                             # LLM routing (default)
    >>> answer = router.route("How do I reverse a linked list in Python?")

    >>> # Complex query — multiple experts activated
    >>> answer = router.route("Build an AI Hospital Management System")

    >>> # Use keyword routing explicitly
    >>> from router.router import KeywordRoutingStrategy
    >>> router = ExpertRouter(strategy=KeywordRoutingStrategy())

    >>> # Inspect routing without running inference
    >>> experts = router.selected_experts("Build an AI Hospital Management System")
    >>> print([e.value for e in experts])
    ['coding', 'system_design', 'ml', 'genai']
    """

    def __init__(
        self,
        strategy: Optional[RoutingStrategy] = None,
        engine: "Optional[CollaborationEngine]" = None,
    ) -> None:
        # Default to Hybrid Router (ML Classifier + LLM Router fallback).
        if strategy is None:
            from router.hybrid_router import HybridRouter  # noqa: PLC0415
            strategy = HybridRouter()

        self._strategy: RoutingStrategy = strategy

        # CollaborationEngine handles parallel execution + aggregation.
        if engine is None:
            from router.aggregator import CollaborationEngine  # noqa: PLC0415
            engine = CollaborationEngine()
        self._engine: "CollaborationEngine" = engine

        # Expert registry populated lazily on first route() call.
        self._registry: Optional[dict[ExpertName, object]] = None

        # Tracking metrics for evaluation dashboard
        self.last_responses: list[object] = []
        self.last_router_decision: dict = {}

        # Multi-Agent Collaboration Components
        from router.planner import PlannerAgent  # noqa: PLC0415
        from router.orchestrator import AgentOrchestrator  # noqa: PLC0415
        self._planner: PlannerAgent = PlannerAgent()
        self._orchestrator: AgentOrchestrator = AgentOrchestrator()

        # Initialize AI Decision Engine
        from router.decision_engine import AIDecisionEngine  # noqa: PLC0415
        self._decision_engine: AIDecisionEngine = AIDecisionEngine()

        self.last_timeline: list[object] = []
        self.last_execution_plan: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        query: str,
        memory: "Optional[ConversationMemory]" = None,
        image_path: Optional[str] = None
    ) -> str:
        """
        Route ``query`` using ML classification or Agentic Planner, execute plan,
        chain contexts, and aggregate final outputs.
        """
        import time                                                  # noqa: PLC0415
        t_route_start = time.perf_counter()

        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        self._ensure_registry()

        from router.planner import ExecutionPlan, ExecutionStep      # noqa: PLC0415
        from datetime import datetime                                # noqa: PLC0415

        # 1. Route query using the Hybrid Router strategy to identify primary expert
        selected_experts = self._strategy.select_experts(query)
        primary_expert = selected_experts[0] if selected_experts else ExpertName.CODING

        # Extract metadata from the hybrid router
        decision_meta = getattr(self._strategy, "last_decision", {})
        ml_predicted = decision_meta.get("predicted_expert")
        ml_confidence = decision_meta.get("confidence", 0.0)
        router_used = decision_meta.get("routing_strategy", "LLM Router")
        fallback_used = decision_meta.get("fallback_used", True)

        # 2. Evaluate decision via AI Decision Engine
        if len(selected_experts) > 1:
            decision = self._decision_engine.decide(query, primary_expert)
            additional_experts_strs = decision.get("additional_experts", [])
            decision_reason = decision.get("reason", "")
            execution_order_strs = decision.get("execution_order", [])
        else:
            additional_experts_strs = []
            decision_reason = "Single expert routing sufficient. Decision Engine bypassed for latency optimization."
            execution_order_strs = [primary_expert.value]

        # Map strings to ExpertName enum values
        additional_experts = []
        for e in additional_experts_strs:
            try:
                additional_experts.append(ExpertName(e))
            except ValueError:
                pass

        all_selected = [primary_expert] + additional_experts

        # 3. Construct plan based on single vs multiple experts decision
        if not additional_experts:
            # Single Expert decision: bypass Planner Agent and execute directly
            step = ExecutionStep(step_id=1, expert_name=primary_expert, dependencies=[])
            plan = ExecutionPlan(steps=[step], query=query, strategy="single_expert")
            logger.info("AI Decision Engine routed query to Single Expert: %s", primary_expert.value)
        else:
            # Multiple Experts decision: Send selected experts to the Planner Agent via guided query
            guided_query = (
                f"Query: {query}\n"
                f"Plan using ONLY these experts: {', '.join(e.value for e in all_selected)}"
            )
            plan = self._planner.plan(guided_query)
            logger.info("AI Decision Engine routed query to Multiple Experts: %s", [e.value for e in all_selected])

        # If an image is uploaded, automatically activate the Vision Expert
        if image_path:
            has_vision = any(s.expert_name == ExpertName.VISION for s in plan.steps)
            if not has_vision:
                from router.planner import ExecutionStep  # noqa: PLC0415
                # Re-index existing steps to make room for step 1: Vision
                new_steps = [ExecutionStep(step_id=1, expert_name=ExpertName.VISION, dependencies=[])]
                for step in plan.steps:
                    # Shift IDs and dependencies
                    shifted_deps = [d + 1 for d in step.dependencies]
                    # If it had no dependencies, make it depend on the vision step
                    if not shifted_deps:
                        shifted_deps = [1]
                    new_steps.append(ExecutionStep(
                        step_id=step.step_id + 1,
                        expert_name=step.expert_name,
                        dependencies=shifted_deps
                    ))
                plan.steps = new_steps

        self.last_execution_plan = plan.to_dict()

        # 4. Extract activated experts list for the router decision
        expert_names = [step.expert_name for step in plan.steps]

        # Compute semantic confidence scores for all experts
        from router.router import SemanticRoutingStrategy  # noqa: PLC0415
        semantic_strategy = SemanticRoutingStrategy()
        confidence_scores = semantic_strategy._score_all(query)

        # Record router decisions metadata
        self.last_router_decision = {
            "strategy_used": f"{router_used} ({plan.strategy})" if not image_path else f"{router_used} (Image Forced)",
            "selected_experts": [e.value for e in expert_names],
            "query": query,
            "confidence_scores": {e.value: float(confidence_scores[e]) for e in confidence_scores},
            # ML Intent Classifier routing telemetry
            "predicted_expert": ml_predicted,
            "confidence": ml_confidence,
            "router_used": router_used,
            "fallback_used": fallback_used,
            "timestamp": datetime.now().isoformat(),
            # AI Decision Engine telemetry
            "primary_expert": primary_expert.value,
            "additional_experts": [e.value for e in additional_experts],
            "reason": decision_reason,
            "execution_order": execution_order_strs
        }

        # 5. Agent Orchestrator executes the DAG layer-by-layer
        responses, timeline = self._orchestrator.execute(
            plan,
            self._registry,
            memory=memory,
            image_path=image_path
        )

        self.last_responses = responses
        self.last_timeline = timeline

        # Record answer quality engine metadata for diagnostics
        if self.last_router_decision is None:
            self.last_router_decision = {}
        self.last_router_decision["answer_quality_plans"] = {
            r.expert_name.value: getattr(r, "answer_plan", "") for r in responses if getattr(r, "answer_plan", None)
        }
        self.last_router_decision["answer_quality_reviews"] = {
            r.expert_name.value: getattr(r, "review_feedback", "") for r in responses if getattr(r, "review_feedback", None)
        }

        # Measure query latency
        elapsed_s = time.perf_counter() - t_route_start

        # 6. Generate Explainable AI Explanation
        try:
            from explainability.engine import ExplainableEngine  # noqa: PLC0415
            from explainability.report import generate_explainability_report  # noqa: PLC0415
            
            xai_engine = ExplainableEngine()
            
            # Estimate tokens based on query and response sizes
            total_response_words = sum(len(str(getattr(r, "content", r.get("content", "") if isinstance(r, dict) else "")).split()) for r in responses)
            tokens_est = int((len(query.split()) + total_response_words) * 1.33)
            
            explanation = xai_engine.generate_explanation(
                query=query,
                router_decision=self.last_router_decision,
                execution_plan=plan.to_dict(),
                response_time_s=elapsed_s,
                tokens_estimated=tokens_est
            )
            
            self.last_router_decision["xai_explanation"] = explanation
            
            generate_explainability_report(explanation)
        except Exception as exc:
            logger.warning("Explainable AI Engine compilation failed: %s", exc)

        # 7. Response Aggregator merges all outputs into one document
        return self._engine.aggregate(responses, query)

    def selected_expert_name(self, query: str) -> ExpertName:
        """
        Return the single top-ranked expert for ``query`` (no inference).
        Useful for simple routing checks and backward compatibility.
        """
        return self._strategy.select_expert(query.strip())

    def selected_experts(self, query: str) -> list[ExpertName]:
        """
        Return all experts that would be activated for ``query`` (no inference).

        Parameters
        ----------
        query : str
            The raw user question.

        Returns
        -------
        list[ExpertName]
            Ordered list of activated experts (highest score first).
        """
        return self._strategy.select_experts(query.strip())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_registry(self) -> None:
        """Build the expert registry on the first call (lazy init)."""
        if self._registry is None:
            logger.info("ExpertRouter: building expert registry ...")
            self._registry = _build_expert_registry()

    def _get_expert(self, expert_name: ExpertName) -> object:
        """
        Retrieve the expert instance for the given name.

        Raises
        ------
        NotImplementedError
            If the expert is not registered.
        """
        expert = self._registry.get(expert_name)
        if expert is None:
            raise NotImplementedError(
                f"Expert '{expert_name.value}' is not registered. "
                f"Available: {[k.value for k in self._registry]}"
            )
        return expert
