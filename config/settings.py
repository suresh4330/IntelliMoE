"""
config/settings.py
------------------
Central configuration for the IntelliMoE application.

All tunable constants live here. Changing a value here propagates
to every module that imports it — no hunting through source files.

Design:
  - GenerationConfig is a frozen dataclass → immutable, hashable, inspectable.
  - EXPERT_CONFIGS maps expert prompt-names to their tuned generation params.
  - Logging, router, and model constants are plain module-level values so they
    can be overridden via environment variables in a future 12-factor extension.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment Variables & API Keys
# ---------------------------------------------------------------------------

# Load environment variables from a .env file if present
load_dotenv()

# Read required API keys for model services
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_ID: str = os.getenv("INTELLIMOE_GEMINI_MODEL", "gemini-3.1-flash-lite")

# Raise exception if either of the keys is missing
if not GROQ_API_KEY:
    raise ValueError(
        "Configuration Error: 'GROQ_API_KEY' environment variable is missing. "
        "Please set it in your environment or a .env file."
    )

if not GEMINI_API_KEY:
    raise ValueError(
        "Configuration Error: 'GEMINI_API_KEY' environment variable is missing. "
        "Please set it in your environment or a .env file."
    )

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_ID: str = os.getenv(
    "INTELLIMOE_MODEL_ID",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)


# ---------------------------------------------------------------------------
# Generation configuration (per expert)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerationConfig:
    """
    Immutable container for text-generation hyperparameters.

    Parameters
    ----------
    max_new_tokens : int
        Maximum number of tokens the model may generate per response.
    temperature : float
        Sampling temperature. Lower → more deterministic. Higher → more creative.
    top_p : float
        Nucleus sampling cumulative probability threshold.
    repetition_penalty : float
        Penalty applied to already-generated tokens (> 1.0 reduces repetition).
    """
    max_new_tokens: int   = 512
    temperature: float    = 0.7
    top_p: float          = 0.95
    repetition_penalty: float = 1.1


# Per-expert generation tuning.
# Temperature rationale:
#   - Math/Research: low (0.3–0.5) → deterministic, factual answers.
#   - Coding/ML/DL/SysDesign: mid (0.6–0.7) → accurate yet articulate.
#   - GenAI: higher (0.75) → richer, more expansive explanations.
EXPERT_CONFIGS: dict[str, GenerationConfig] = {
    "coding": GenerationConfig(
        max_new_tokens=512, temperature=0.70, top_p=0.95, repetition_penalty=1.10
    ),
    "math": GenerationConfig(
        max_new_tokens=512, temperature=0.30, top_p=0.90, repetition_penalty=1.10
    ),
    "ml": GenerationConfig(
        max_new_tokens=512, temperature=0.60, top_p=0.92, repetition_penalty=1.10
    ),
    "deeplearning": GenerationConfig(
        max_new_tokens=512, temperature=0.65, top_p=0.92, repetition_penalty=1.10
    ),
    "genai": GenerationConfig(
        max_new_tokens=512, temperature=0.75, top_p=0.95, repetition_penalty=1.15
    ),
    "research": GenerationConfig(
        max_new_tokens=600, temperature=0.50, top_p=0.90, repetition_penalty=1.15
    ),
    "systemdesign": GenerationConfig(
        max_new_tokens=600, temperature=0.60, top_p=0.93, repetition_penalty=1.10
    ),
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: int    = int(os.getenv("INTELLIMOE_LOG_LEVEL", str(logging.INFO)))
LOG_FORMAT: str   = "%(asctime)s | %(levelname)-8s | %(name)-32s | %(message)s"
LOG_DATE_FMT: str = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

# Minimum keyword score for an expert to qualify in multi-expert mode.
MIN_SCORE_THRESHOLD: int = int(os.getenv("INTELLIMOE_MIN_SCORE", "1"))

# Hard cap on the number of experts activated per query.
MAX_EXPERTS: int = int(os.getenv("INTELLIMOE_MAX_EXPERTS", "5"))


# ---------------------------------------------------------------------------
# Pricing for cost estimation (per 1,000 tokens)
# ---------------------------------------------------------------------------
# Estimated cloud-equivalent cost per 1,000 tokens (e.g. LLaMA-3-8B API equivalent)
PRICE_PER_1K_PROMPT_TOKENS: float = 0.0001      # $0.10 per million tokens
PRICE_PER_1K_COMPLETION_TOKENS: float = 0.0002  # $0.20 per million tokens


# ---------------------------------------------------------------------------
# Machine Learning Intent Classifier Settings
# ---------------------------------------------------------------------------
# Project paths
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path      = _PROJECT_ROOT / "data"

INTENT_MODEL_PATH: Path = DATA_DIR / "intent_classifier.joblib"
VECTORIZER_PATH: Path   = DATA_DIR / "vectorizer.joblib"

# Confidence threshold below which model falls back to the LLM/Planner router.
ML_ROUTING_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("INTELLIMOE_ML_CONFIDENCE", "0.60")
)

