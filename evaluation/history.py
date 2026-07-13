"""
evaluation/history.py
---------------------
Manages storage, loading, and CSV exporting of AI Evaluation metrics.
Saves data into data/evaluation_history.csv.
"""

import os
import csv
from datetime import datetime
from pathlib import Path
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = _PROJECT_ROOT / "data" / "evaluation_history.csv"


def log_evaluation_run(
    provider: str,
    response_time: float,
    token_usage: int,
    faithfulness: float,
    relevance: float,
    completeness: float,
    hallucination_risk: float,
    routing_accuracy: float,
    expert_selection_accuracy: float,
    multi_agent_accuracy: float,
    response_quality: float,
    overall_ai_score: float,
    overall_system_score: float,
    reasoning: str
) -> None:
    """
    Append a single evaluation run log to data/evaluation_history.csv.
    """
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = CSV_PATH.exists()
    headers = [
        "timestamp",
        "provider",
        "response_time",
        "token_usage",
        "faithfulness",
        "relevance",
        "completeness",
        "hallucination_risk",
        "routing_accuracy",
        "expert_selection_accuracy",
        "multi_agent_accuracy",
        "response_quality",
        "overall_ai_score",
        "overall_system_score",
        "reasoning"
    ]
    
    row = {
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "response_time": round(response_time, 4),
        "token_usage": int(token_usage),
        "faithfulness": round(faithfulness, 4),
        "relevance": round(relevance, 4),
        "completeness": round(completeness, 4),
        "hallucination_risk": round(hallucination_risk, 4),
        "routing_accuracy": round(routing_accuracy, 4),
        "expert_selection_accuracy": round(expert_selection_accuracy, 4),
        "multi_agent_accuracy": round(multi_agent_accuracy, 4),
        "response_quality": round(response_quality, 4),
        "overall_ai_score": round(overall_ai_score, 2),
        "overall_system_score": round(overall_system_score, 2),
        "reasoning": reasoning.replace("\n", " ").strip()
    }
    
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_evaluation_history() -> pd.DataFrame:
    """
    Read the historical evaluations as a Pandas DataFrame.
    """
    if not CSV_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(CSV_PATH)
    except Exception:
        return pd.DataFrame()
