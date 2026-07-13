"""
experts/vision.py
-----------------
Vision Expert for IntelliMoE.

Capable of explaining flowcharts, UML diagrams, charts, graphs, and reading screenshots.
Uses 'vikhyatk/moondream2' (1.6B parameter fast local VLM) or falls back to MockVisionModel
if offline or resource-constrained.
"""

import logging
import time
from typing import Optional
from PIL import Image

from experts.base import BaseExpert
from config.settings import EXPERT_CONFIGS, GenerationConfig
from router.router import ExpertName

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock Fallback Vision Model
# ---------------------------------------------------------------------------

class MockVisionModel:
    """Mock fallback vision model to prevent loading/network issues on constrained machines."""
    
    def encode_image(self, image: Image.Image) -> str:
        return "mock_encoded_image"

    def answer_question(self, enc_image: str, question: str, tokenizer: Optional[object] = None) -> str:
        q_lower = question.lower()
        if "uml" in q_lower or "class" in q_lower:
            return (
                "🔍 **[Mock Vision Analysis: UML Diagram]**\n\n"
                "Based on the UML structural class diagram:\n"
                "- **Class Components**: Contains `Controller`, `Service`, `Repository`, and `Model` modules.\n"
                "- **Attributes**: The model defines private ID, timestamp, and metadata properties.\n"
                "- **Relationships**: Follows a standard dependency injection flow where request handlers call services sequentially."
            )
        elif "flowchart" in q_lower or "flow" in q_lower or "sequence" in q_lower:
            return (
                "🔍 **[Mock Vision Analysis: Flowchart]**\n\n"
                "Analyzing the workflow flowchart diagram:\n"
                "- **Input/Start**: Process begins with a validation decision node.\n"
                "- **Branching**: If validation checks succeed (Yes), it progresses to DB execution. If they fail (No), it redirects to error handlers.\n"
                "- **End State**: The process concludes by logging metrics to SQLite and returning the payload response."
            )
        elif "graph" in q_lower or "chart" in q_lower or "plot" in q_lower:
            return (
                "🔍 **[Mock Vision Analysis: Data Chart]**\n\n"
                "Based on the plotted visual metrics:\n"
                "- **X-Axis**: Timeline progress / concurrent thread count.\n"
                "- **Y-Axis**: Processing throughput / memory footprint in MB.\n"
                "- **Observations**: Shows linear performance scaling up to peak loads, followed by flat stable performance."
            )
        elif "screenshot" in q_lower:
            return (
                "🔍 **[Mock Vision Analysis: Screenshot]**\n\n"
                "Reading the screen capture contents:\n"
                "- **Header**: A dark theme navbar with thread navigation tabs is visible.\n"
                "- **Main Panel**: Displays a structured log of active expert progress timelines.\n"
                "- **OCR Text**: Found headings: 'Diagnostics Telemetry' and 'System Logs'."
            )
        
        return (
            "🔍 **[Mock Vision Analysis]**\n\n"
            "This is a detailed analysis of the uploaded visual. The Vision Expert identified primary nodes, "
            "connecting arrows, and textual labels. The overall structure represents a sequential data process "
            "integrating relational flows."
        )


# ---------------------------------------------------------------------------
# Global Vision Singletons
# ---------------------------------------------------------------------------

_vision_model: Optional[object] = None
_vision_tokenizer: Optional[object] = None

def get_vision_resources():
    """Lazy-load moondream2 model/tokenizer, falling back to MockVisionModel on failures."""
    global _vision_model, _vision_tokenizer
    if _vision_model is None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
            logger.info("VisionExpert: loading model 'vikhyatk/moondream2' …")
            _vision_model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                trust_remote_code=True,
                revision="2024-08-26"
            )
            _vision_tokenizer = AutoTokenizer.from_pretrained(
                "vikhyatk/moondream2",
                trust_remote_code=True
            )
            logger.info("VisionExpert: moondream2 loaded successfully.")
        except Exception as e:
            logger.warning(
                "VisionExpert: failed to load moondream2 local VLM (%s). Falling back to MockVisionModel.", e
            )
            _vision_model = MockVisionModel()
            _vision_tokenizer = None
            
    return _vision_model, _vision_tokenizer


# ---------------------------------------------------------------------------
# VisionExpert Subclass
# ---------------------------------------------------------------------------

class VisionExpert(BaseExpert):
    """
    Expert for processing images, explaining flowcharts, screenshots, and graphs.
    """

    def __init__(self) -> None:
        super().__init__()
        # Custom tracker for image path
        self.last_image_path: Optional[str] = None

    @property
    def prompt_name(self) -> str:
        return "vision"

    @property
    def generation_config(self) -> GenerationConfig:
        # Reuses coding configuration parameters or standard settings
        return EXPERT_CONFIGS.get("coding", EXPERT_CONFIGS[list(EXPERT_CONFIGS.keys())[0]])

    def answer(self, question: str, memory: Optional[object] = None, image_path: Optional[str] = None) -> str:
        """
        Processes vision queries using moondream2 VLM or mock fallbacks.
        """
        question = self._validate_question(question)
        
        # If no image path was explicitly passed, try to fetch from instance state
        img_path = image_path or self.last_image_path
        
        if not img_path:
            logger.warning("VisionExpert called without an uploaded image. Running text-only fallback.")
            return "⚠️ **Vision Expert**: Please upload an image first using the sidebar uploader so I can analyze it for you!"

        t_start = time.perf_counter()
        try:
            model, tokenizer = get_vision_resources()
            
            logger.info("VisionExpert: opening image from '%s' ...", img_path)
            image = Image.open(img_path)
            
            # Run inference using the moondream2 API
            enc_image = model.encode_image(image)
            
            # Build query containing question
            result = model.answer_question(enc_image, question, tokenizer)
            elapsed = time.perf_counter() - t_start

            # Mock token counts for telemetry tracking
            self.last_prompt_tokens = len(question) // 4 + 150
            self.last_tokens_generated = len(result) // 4
            
            logger.info("VisionExpert: generation completed in %.2fs.", elapsed)
            return result

        except Exception as exc:
            logger.exception("VisionExpert failed to analyze image.")
            raise RuntimeError(f"VisionExpert failed: {exc}") from exc
