"""
models/loader.py
----------------
Singleton model and tokenizer loader for IntelliMoE.

Design:
  - Singleton pattern: model and tokenizer are loaded exactly once and
    reused across all expert calls — no repeated disk I/O or memory overhead.
  - Device auto-detection: CUDA if available, otherwise CPU.
  - Evaluation mode: model is placed in eval() to disable dropout and
    ensure deterministic, inference-optimised behaviour.
  - Config-driven: MODEL_ID and dtype are sourced from config.settings.
"""

import logging
import threading
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from config.settings import MODEL_ID

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------
_tokenizer: Optional[PreTrainedTokenizerBase] = None
_model: Optional[PreTrainedModel]             = None

# ---------------------------------------------------------------------------
# Inference lock
# ---------------------------------------------------------------------------
# Since all experts share ONE TinyLlama model, concurrent calls to
# model.generate() from different threads must be serialized to prevent
# undefined behavior. This lock is acquired by BaseExpert._generate()
# (via get_inference_lock()) before every model.generate() call.
# Pre/post processing (tokenisation, decoding, prompt building) runs
# fully in parallel — only the generate() call is gated.
inference_lock: threading.Lock = threading.Lock()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model_and_tokenizer() -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """
    Return the shared tokenizer and model, loading them on the first call.

    On subsequent calls the cached instances are returned immediately —
    safe to call from multiple expert modules without any reload cost.

    Returns
    -------
    tokenizer : PreTrainedTokenizerBase
        The TinyLlama chat tokenizer (fast Rust-backed implementation).
    model : PreTrainedModel
        The TinyLlama causal-LM model in evaluation mode on the best
        available device (CUDA → CPU).

    Raises
    ------
    RuntimeError
        If the model or tokenizer cannot be loaded from Hugging Face.
    """
    global _tokenizer, _model

    # Return cached singletons if already initialised.
    if _tokenizer is not None and _model is not None:
        logger.debug("Returning cached model and tokenizer.")
        return _tokenizer, _model

    device = _detect_device()
    logger.info("Initialising model '%s' on device '%s' …", MODEL_ID, device)

    try:
        _tokenizer = _load_tokenizer()
        _model     = _load_model(device)
        _model.eval()   # Disable dropout; enable deterministic inference.
        logger.info("Model ready — device='%s', dtype='%s'.", device, _model.dtype)

    except Exception as exc:
        # Reset state so a subsequent call can retry cleanly.
        _tokenizer = None
        _model     = None
        logger.exception("Failed to load model '%s'.", MODEL_ID)
        raise RuntimeError(
            f"Could not load model or tokenizer for '{MODEL_ID}'. "
            f"Ensure Hugging Face Transformers is installed and the model "
            f"ID is correct. Cause: {type(exc).__name__}: {exc}"
        ) from exc

    return _tokenizer, _model


def get_device() -> str:
    """
    Return the device string used for inference.

    Returns
    -------
    str
        ``"cuda"`` if a CUDA GPU is available, otherwise ``"cpu"``.
    """
    return _detect_device()


def is_model_loaded() -> bool:
    """
    Return True if the model singleton has been initialised.

    Useful for health-check endpoints or conditional logging.
    """
    return _tokenizer is not None and _model is not None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    """Detect the best available compute device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug("Detected compute device: %s", device)
    return device


def _load_tokenizer() -> PreTrainedTokenizerBase:
    """
    Download/load the tokenizer from Hugging Face Hub.

    ``use_fast=True`` selects the Rust-backed fast tokenizer — faster
    encoding and decoding compared to the Python implementation.
    """
    logger.info("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    logger.info("Tokenizer loaded (vocab_size=%d).", tokenizer.vocab_size)
    return tokenizer


def _load_model(device: str) -> PreTrainedModel:
    """
    Download/load the causal LM from Hugging Face Hub.

    ``device_map="auto"`` lets Accelerate distribute model layers across
    all available devices (GPU(s) → CPU → disk) automatically.

    ``torch_dtype`` is set to float16 on CUDA (halves VRAM) and float32
    on CPU (float16 is not optimised for CPU inference).
    """
    logger.info("Loading model weights …")
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
    )
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model loaded (%.0fM parameters, dtype=%s).", param_count, dtype)
    return model


def get_memory_usage_mb() -> float:
    """
    Return the current memory usage of the system or GPU.
    If CUDA is used and available, returns the PyTorch VRAM allocated in MB.
    Otherwise, returns the RAM (Resident Set Size) used by this process in MB.
    """
    try:
        import psutil
        device = _detect_device()
        if device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
    except Exception as exc:
        logger.warning("Could not calculate memory usage: %s", exc)
        return 0.0

