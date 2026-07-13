"""
utils/evaluation.py
-------------------
Expert Evaluation Engine for IntelliMoE.

Tracks, stores, and aggregates system diagnostics, compute resources,
and token metrics for all expert runs using an SQLite database backend.

Features:
  - SQLite database for persistent logging.
  - Automatically captures response time, token metrics, CPU/memory usage, and success states.
  - Calculates aggregated analytics: success/failure rates, average latency, and averages.
  - Extensible and reusable classes for logging telemetry.
"""

import os
import sqlite3
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Persistent SQLite Database path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = _PROJECT_ROOT / "data"
DB_PATH = DB_DIR / "evaluation.db"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExpertRunMetric:
    """Telemetry data representing a single expert execution run."""
    expert_name: str
    query: str
    response_time: float
    prompt_tokens: int
    completion_tokens: int
    memory_usage_mb: float
    cpu_usage_pct: float
    success: bool
    error: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class ExpertAggregateMetrics:
    """Aggregated historical analytics for a given expert domain."""
    expert_name: str
    total_runs: int
    average_response_time: float
    total_tokens_generated: int
    average_prompt_tokens: float
    average_completion_tokens: float
    average_memory_usage_mb: float
    average_cpu_usage_pct: float
    success_rate: float
    failure_rate: float


# ---------------------------------------------------------------------------
# EvaluationEngine
# ---------------------------------------------------------------------------

class EvaluationEngine:
    """
    Manages SQLite storage, querying, logging, and aggregation of expert telemetry.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create the database and required tables if they do not exist."""
        DB_DIR.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS expert_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expert_name TEXT NOT NULL,
                    query TEXT,
                    response_time REAL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    memory_usage_mb REAL,
                    cpu_usage_pct REAL,
                    success INTEGER,
                    error TEXT
                )
            """)
            conn.commit()

    def log_run(self, metric: ExpertRunMetric) -> None:
        """Log a single expert execution to the SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO expert_logs (
                        expert_name, query, response_time, prompt_tokens,
                        completion_tokens, memory_usage_mb, cpu_usage_pct,
                        success, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.expert_name,
                    metric.query,
                    metric.response_time,
                    metric.prompt_tokens,
                    metric.completion_tokens,
                    metric.memory_usage_mb,
                    metric.cpu_usage_pct,
                    1 if metric.success else 0,
                    metric.error
                ))
                conn.commit()
            logger.info("EvaluationEngine: logged execution run for expert '%s'", metric.expert_name)
        except Exception as e:
            logger.error("EvaluationEngine: failed to log run to database: %s", e)

    def get_expert_aggregates(self) -> list[ExpertAggregateMetrics]:
        """Compute aggregated metrics for all experts based on SQLite records."""
        aggregates = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Fetch group-by averages and sums
                cursor.execute("""
                    SELECT 
                        expert_name,
                        COUNT(*) as total_runs,
                        AVG(response_time) as avg_latency,
                        SUM(completion_tokens) as total_tokens,
                        AVG(prompt_tokens) as avg_prompt,
                        AVG(completion_tokens) as avg_completion,
                        AVG(memory_usage_mb) as avg_mem,
                        AVG(cpu_usage_pct) as avg_cpu,
                        SUM(success) as successful_runs
                    FROM expert_logs
                    GROUP BY expert_name
                """)
                
                rows = cursor.fetchall()
                for row in rows:
                    total = row["total_runs"]
                    successes = row["successful_runs"]
                    
                    success_rate = (successes / total) if total > 0 else 0.0
                    failure_rate = 1.0 - success_rate
                    
                    aggregates.append(ExpertAggregateMetrics(
                        expert_name=row["expert_name"],
                        total_runs=total,
                        average_response_time=row["avg_latency"],
                        total_tokens_generated=row["total_tokens"] or 0,
                        average_prompt_tokens=row["avg_prompt"],
                        average_completion_tokens=row["avg_completion"],
                        average_memory_usage_mb=row["avg_mem"],
                        average_cpu_usage_pct=row["avg_cpu"],
                        success_rate=success_rate,
                        failure_rate=failure_rate
                    ))
        except Exception as e:
            logger.error("EvaluationEngine: failed to compute aggregates: %s", e)
            
        return aggregates

    def get_raw_logs(self, limit: int = 100) -> list[ExpertRunMetric]:
        """Fetch the most recent raw log entries from SQLite."""
        logs = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT timestamp, expert_name, query, response_time, prompt_tokens,
                           completion_tokens, memory_usage_mb, cpu_usage_pct, success, error
                    FROM expert_logs
                    ORDER BY id DESC
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                for r in rows:
                    logs.append(ExpertRunMetric(
                        expert_name=r["expert_name"],
                        query=r["query"],
                        response_time=r["response_time"],
                        prompt_tokens=r["prompt_tokens"],
                        completion_tokens=r["completion_tokens"],
                        memory_usage_mb=r["memory_usage_mb"],
                        cpu_usage_pct=r["cpu_usage_pct"],
                        success=bool(r["success"]),
                        error=r["error"],
                        timestamp=r["timestamp"]
                    ))
        except Exception as e:
            logger.error("EvaluationEngine: failed to fetch raw logs: %s", e)
        return logs

    def clear_logs(self) -> None:
        """Truncate the expert logs table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM expert_logs")
                conn.commit()
            logger.info("EvaluationEngine: database logs cleared.")
        except Exception as e:
            logger.error("EvaluationEngine: failed to clear logs: %s", e)
