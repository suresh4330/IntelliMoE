"""
benchmark/history.py
--------------------
Manages benchmark history logging and data exporting to CSV format.
"""

import os
import csv
from datetime import datetime
from pathlib import Path
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = _PROJECT_ROOT / "data" / "benchmark_history.csv"

def log_benchmark_run(
    provider: str,
    model_name: str,
    response_time: float,
    first_token_latency: float,
    total_response_time: float,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    success_rate: float,
    error_rate: float,
    avg_confidence: float,
    estimated_cost: float,
) -> None:
    """
    Append a single benchmark execution run to the data/benchmark_history.csv database.
    """
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = CSV_PATH.exists()
    headers = [
        "timestamp",
        "provider",
        "model_name",
        "response_time",
        "first_token_latency",
        "total_response_time",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "success_rate",
        "error_rate",
        "avg_confidence",
        "estimated_cost"
    ]
    
    row = {
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "model_name": model_name,
        "response_time": round(response_time, 4),
        "first_token_latency": round(first_token_latency, 4),
        "total_response_time": round(total_response_time, 4),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
        "success_rate": round(success_rate, 4),
        "error_rate": round(error_rate, 4),
        "avg_confidence": round(avg_confidence, 4),
        "estimated_cost": round(estimated_cost, 6)
    }
    
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def get_benchmark_history() -> pd.DataFrame:
    """
    Read the complete benchmark execution logs as a Pandas DataFrame.
    """
    if not CSV_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(CSV_PATH)
    except Exception:
        return pd.DataFrame()
