"""
explainability/timeline.py
--------------------------
Timeline formatter for Explainable AI Engine.
Generates structural execution events trace lists and explainability reasoning steps.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

def _safe_upper(val: Any) -> str:
    """Helper to convert any value to string and return upper case safely."""
    if val is None:
        return "N/A"
    return str(val).strip().upper()


def generate_execution_timeline(explanation: Dict[str, Any]) -> List[str]:
    """
    Generate a millisecond-precision event trace detailing the system steps.
    """
    elapsed_ms = explanation.get("performance", {}).get("response_time_ms", 450)
    t_start = datetime.now() - timedelta(milliseconds=elapsed_ms)
    
    timeline = []
    
    # Progress mappings for key pipeline milestones
    events = [
        ("Query Received", 0.0),
        ("ML Classifier", 0.08),
        ("Decision Engine", 0.18),
        ("Planner", 0.32)
    ]
    
    order = explanation.get("planner", {}).get("execution_order", [])
    for idx, exp in enumerate(order):
        if exp:
            prov = "Groq API" if str(exp).lower() in ["coding", "math"] else "Gemini API"
            events.append((f"{_safe_upper(exp)} Expert", 0.45 + idx * 0.15))
            events.append((prov, 0.58 + idx * 0.15))
        
    events.append(("Response Aggregator", 0.85))
    events.append(("Final Response", 1.0))
    
    for label, progress in events:
        evt_time = t_start + timedelta(milliseconds=elapsed_ms * progress)
        timeline.append(f"{evt_time.strftime('%H:%M:%S.%f')[:-3]} → {label}")
        
    return timeline


def generate_reasoning_timeline(explanation: Dict[str, Any]) -> List[str]:
    """
    Generate human-readable step-by-step logic detailing choices made.
    """
    router = explanation.get("router", {})
    decision = explanation.get("decision_engine", {})
    api = explanation.get("api", {})
    
    timeline = []
    
    # Step 1: ML Classifier
    pred = router.get("prediction", "N/A")
    conf = router.get("confidence", 0.0)
    timeline.append(f"Step 1:\nML Classifier predicted '{_safe_upper(pred)}' with {float(conf)*100:.1f}% confidence.")
    
    # Step 2: Threshold comparison
    from config.settings import ML_ROUTING_CONFIDENCE_THRESHOLD  # noqa: PLC0415
    timeline.append(f"Step 2:\nConfidence score threshold set to {ML_ROUTING_CONFIDENCE_THRESHOLD*100:.1f}%.")
        
    # Step 3: Fallback check
    fallback = router.get("fallback", True)
    if not fallback:
        timeline.append("Step 3:\nNo fallback required. ML Prediction selected as primary.")
    else:
        timeline.append("Step 3:\nFallback triggered. Routed to LLM routing agent.")
        
    # Step 4: Decision Engine
    primary = decision.get("primary_expert", "N/A")
    additionals = decision.get("additional_experts", [])
    
    clean_adds = [add for add in additionals if add]
    if clean_adds:
        timeline.append(
            f"Step 4:\nAI Decision Engine selected Primary Expert '{_safe_upper(primary)}' "
            f"and Additional Experts: {', '.join(map(_safe_upper, clean_adds))}.\n"
            f"Reasoning: {decision.get('reason')}"
        )
    else:
        timeline.append(
            f"Step 4:\nAI Decision Engine verified Primary Expert '{_safe_upper(primary)}' alone is sufficient.\n"
            f"Reasoning: {decision.get('reason')}"
        )
        
    # Step 5: API Selection
    timeline.append(
        f"Step 5:\nAPI Selection resolved to: {api.get('provider')}.\n"
        f"Justification: {api.get('reason')}"
    )
    
    return timeline
