"""
utils/feedback.py
-----------------
Feedback Learning System for IntelliMoE.

Manages storing user feedback (likes/dislikes) in the SQLite database,
calculates net ratings for all experts, and recommends prompt optimization
instructions to improve expert outputs based on performance ratings.
"""

import sqlite3
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Reuses the same database file for consolidated metrics storage
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = _PROJECT_ROOT / "data" / "evaluation.db"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class FeedbackRecord:
    """Represents a feedback event logged by the user."""
    expert_name: str
    query: str
    response: str
    rating: int  # 1 for Like, -1 for Dislike
    timestamp: Optional[str] = None


# ---------------------------------------------------------------------------
# FeedbackSystem
# ---------------------------------------------------------------------------

class FeedbackSystem:
    """
    Manages SQLite storage, query telemetry, ratings calculations,
    and automatic prompt engineering recommendations.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the feedback schema inside SQLite."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS expert_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expert_name TEXT NOT NULL,
                    query TEXT,
                    response TEXT,
                    rating INTEGER NOT NULL
                )
            """)
            conn.commit()

    def add_feedback(self, expert_name: str, query: str, response: str, rating: int) -> None:
        """
        Record user feedback. Rating is +1 for Like and -1 for Dislike.
        Updates or creates a record in the database.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO expert_feedback (expert_name, query, response, rating)
                    VALUES (?, ?, ?, ?)
                """, (expert_name, query, response, rating))
                conn.commit()
            logger.info("FeedbackSystem: logged rating %d for expert '%s'", rating, expert_name)
        except Exception as e:
            logger.error("FeedbackSystem: failed to log feedback: %s", e)

    def get_ratings_summary(self) -> dict[str, dict]:
        """
        Compute rating statistics (Likes, Dislikes, Net Rating, Like Ratio)
        for every expert.
        """
        summary = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        expert_name,
                        SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as likes,
                        SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) as dislikes,
                        COUNT(*) as total
                    FROM expert_feedback
                    GROUP BY expert_name
                """)
                
                rows = cursor.fetchall()
                for r in rows:
                    name = r["expert_name"]
                    likes = r["likes"]
                    dislikes = r["dislikes"]
                    total = r["total"]
                    
                    net_rating = likes - dislikes
                    ratio = (likes / total) if total > 0 else 0.0
                    
                    summary[name] = {
                        "likes": likes,
                        "dislikes": dislikes,
                        "total": total,
                        "net_rating": net_rating,
                        "like_ratio": ratio
                    }
        except Exception as e:
            logger.error("FeedbackSystem: failed to fetch ratings summary: %s", e)
            
        return summary

    def get_prompt_recommendations(self, expert_name: str) -> list[str]:
        """
        Recommends optimization advice based on the feedback history
        of the specified expert.
        """
        ratings = self.get_ratings_summary()
        expert_stats = ratings.get(expert_name, {"likes": 0, "dislikes": 0, "total": 0, "like_ratio": 1.0})
        
        # High-quality prompt guidelines based on domain experts
        prompt_templates = {
            "coding": [
                "💡 **Enforce Step-by-Step Thinking**: Prepend your query with 'Explain your architectural logic in 3 bullet points before writing any code.'",
                "💡 **Request Unit Tests**: Add 'Include pytest or unittest assertions verifying edge cases for the written functions.'",
                "💡 **Add Type Annotations**: Append 'Use Python type hints and strict docstrings describing function arguments and returns.'"
            ],
            "math": [
                "💡 **Enforce Verification Step**: Add 'Double-check all calculation sign transitions before outputting the final equation value.'",
                "💡 **Chain of Thought**: Query with 'Break the derivative or proof down into individual algebraic operations step-by-step.'"
            ],
            "ml": [
                "💡 **Explain Assumptions**: Append 'Clarify the evaluation metric choice (e.g. F1-score vs ROC-AUC) based on class balance.'",
                "💡 **Outline Preprocessing**: Append 'Explicitly state feature scaling, imputation, and column encoding strategies.'"
            ],
            "deep_learning": [
                "💡 **Define Layer Dimensions**: Add 'State input, hidden, and output tensor shape transformations throughout the network layers.'",
                "💡 **Regularization Advice**: Include 'Specify dropout rates, batch normalization placements, and weights decay parameters.'"
            ],
            "genai": [
                "💡 **Reduce Hallucination**: Query with 'Provide citations or say \"I do not know\" if you are not sure of a fact.'",
                "💡 **System Prompts**: Append 'Establish a clear persona (e.g., \"You are a senior RAG agent engineer\").'"
            ],
            "research": [
                "💡 **Context Grounding**: Instruct with 'Focus strictly on comparing baseline results and SOTA methodology metrics.'",
                "💡 **Ablation Studies**: Add 'Highlight which components of the paper's framework contributed most to performance gain.'"
            ],
            "system_design": [
                "💡 **Draw Data Flow**: Query with 'Use ascii diagrams or mermaid charts representing client-server data synchronization pathways.'",
                "💡 **Enforce SLA Bounds**: Append 'Identify single points of failure (SPOF) and outline failover/replication procedures.'"
            ]
        }

        # Recommendations based on rating state
        recommendations = []
        
        # Default fallback tips if no history exists yet
        default_tips = prompt_templates.get(expert_name, [
            "💡 Include explicit instructions requesting structured bullet points.",
            "💡 Tell the assistant to review its reasoning step-by-step."
        ])

        # If expert has negative net feedback or dislikes, serve specific suggestions
        if expert_stats["dislikes"] > 0 or expert_stats["like_ratio"] < 0.8:
            recommendations.append(f"⚠️ **Expert '{expert_name.title()}' has received negative feedback.** Apply these prompt optimizations:")
            recommendations.extend(prompt_templates.get(expert_name, default_tips))
        else:
            # Good ratings suggestions
            recommendations.append(f"✅ **Expert '{expert_name.title()}' is highly rated ({expert_stats['like_ratio']:.1%} liked).** Tips to maintain high quality:")
            recommendations.append(prompt_templates.get(expert_name, default_tips)[0])
            
        return recommendations

    def clear_feedback(self) -> None:
        """Truncate the feedback table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM expert_feedback")
                conn.commit()
            logger.info("FeedbackSystem: feedback logs cleared.")
        except Exception as e:
            logger.error("FeedbackSystem: failed to clear feedback: %s", e)
