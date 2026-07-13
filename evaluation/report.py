"""
evaluation/report.py
--------------------
Report generator for the AI Evaluation Framework.
Writes evaluation_report.md in the project root summarizing latest metrics.
"""

from pathlib import Path
from typing import Any, Dict
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = _PROJECT_ROOT / "evaluation_report.md"
CSV_PATH = _PROJECT_ROOT / "data" / "evaluation_history.csv"


def generate_evaluation_report(latest_eval: Dict[str, Any]) -> None:
    """
    Compile system evaluation scores and write to evaluation_report.md in project root.
    """
    # 1. Gather historical baseline statistics
    history_df = pd.read_csv(CSV_PATH) if CSV_PATH.exists() else pd.DataFrame([latest_eval])
    
    # 2. Compute averages per provider (Groq vs Gemini)
    summary_rows = ""
    if not history_df.empty and "provider" in history_df.columns:
        # Group by provider/router
        grouped = history_df.groupby("provider").agg(
            avg_ai_score=("overall_ai_score", "mean"),
            avg_sys_score=("overall_system_score", "mean"),
            avg_faithfulness=("faithfulness", "mean"),
            avg_relevance=("relevance", "mean"),
            avg_completeness=("completeness", "mean"),
            avg_latency=("response_time", "mean"),
            avg_tokens=("token_usage", "mean")
        ).reset_index()
        
        for _, r in grouped.iterrows():
            summary_rows += (
                f"| **{r['provider']}** "
                f"| {r['avg_ai_score']:.1f}/100 "
                f"| {r['avg_sys_score']:.1f}/100 "
                f"| {r['avg_faithfulness']*100:.1f}% "
                f"| {r['avg_relevance']*100:.1f}% "
                f"| {r['avg_completeness']*100:.1f}% "
                f"| {r['avg_latency']:.3f}s "
                f"| {int(r['avg_tokens']):,} |\n"
            )
    else:
        # Fallback single row presentation
        summary_rows = f"| **Latest Run** | {latest_eval['overall_ai_score']:.1f}/100 | {latest_eval['overall_system_score']:.1f}/100 | {latest_eval['faithfulness']*100:.1f}% | {latest_eval['relevance']*100:.1f}% | {latest_eval['completeness']*100:.1f}% | {latest_eval['response_time']:.3f}s | {latest_eval['token_usage']:,} |\n"

    # 3. Format markdown content
    report_content = f"""# AI Evaluation Report: IntelliMoE Diagnostic Assessment

This report provides a structural assessment of IntelliMoE's responses, routing accuracies, and operational efficiency across Groq and Gemini API endpoints.

---

## 👤 Latest Query Evaluation
> **{latest_eval.get('query', 'N/A')}**

### 🎯 Metric Dials (Latest Turn)
- 🧠 **Overall AI Score**: **{latest_eval.get('overall_ai_score', 0.0):.1f}/100**
- ⚙️ **Overall System Score**: **{latest_eval.get('overall_system_score', 0.0):.1f}/100**
- ⏱️ **Response Time**: **{latest_eval.get('response_time', 0.0):.3f} seconds**
- 🪙 **Estimated Tokens**: **{latest_eval.get('token_usage', 0):,} tokens**
- 🕵️ **Hallucination Risk**: **{latest_eval.get('hallucination_risk', 0.0)*100:.1f}%**
- 💡 **Evaluation Justification**: 
  *{latest_eval.get('reasoning', 'N/A')}*

---

## 📊 Comparative Performance Analytics (Groq vs Gemini)

| Provider / Model | Avg AI Score | Avg System Score | Faithfulness | Relevance | Completeness | Avg Latency | Avg Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{summary_rows}

---

## 🧬 Semantic Metric Definitions

1. **Faithfulness**: Evaluates the extent to which the response remains factually aligned with retrieved documents and context.
2. **Relevance**: Evaluates if the response addresses the core intent of the user's explicit question.
3. **Completeness**: Analyzes whether all dimensions and sub-tasks of the user query were addressed.
4. **Hallucination Risk**: Likelihood of the model fabricating details not supported by the context.
5. **Response Quality**: Assessment of tone, clarity, and structural formatting.

---

*Report generated automatically by the AI Evaluation Framework on {latest_eval.get('provider', 'IntelliMoE')}. Database stored at `data/evaluation_history.csv`.*
"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
