"""
explainability/report.py
------------------------
Report exporter for Explainable AI Engine.
Generates explainability_report.md in the project root detailing XAI decisions.
"""

from pathlib import Path
from typing import Any, Dict

from explainability.graph import generate_ascii_graph
from explainability.timeline import generate_reasoning_timeline

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = _PROJECT_ROOT / "explainability_report.md"


def generate_explainability_report(explanation: Dict[str, Any]) -> None:
    """
    Compile query logic results and write to explainability_report.md.
    """
    query = explanation.get("query", "User Query")
    router = explanation.get("router", {})
    api = explanation.get("api", {})
    perf = explanation.get("performance", {})
    
    ascii_flow = generate_ascii_graph(explanation)
    reasoning_steps = generate_reasoning_timeline(explanation)
    
    reasoning_md = ""
    for step in reasoning_steps:
        parts = step.split(":\n", 1)
        if len(parts) == 2:
            reasoning_md += f"### 💡 {parts[0]}\n{parts[1].strip()}\n\n"
        else:
            reasoning_md += f"### 💡 Step\n{step.strip()}\n\n"
            
    news_rewriter = explanation.get("news_rewriter", {})
    news_md = ""
    if news_rewriter and news_rewriter.get("rewritten_query"):
        news_md = f"""
---

## 📰 News Live Search & Rewriter Details

- **Original User Query**: `{news_rewriter.get('original_query', '')}`
- **Rewritten Search Query**: `{news_rewriter.get('rewritten_query', '')}`
- **Search Provider Used**: `{news_rewriter.get('search_provider', '')}`
- **Search Latency**: {news_rewriter.get('search_latency_s', 0.0):.2f}s
- **LLM Synthesis Latency**: {news_rewriter.get('llm_latency_s', 0.0):.2f}s
- **Sources Used**: {', '.join(news_rewriter.get('sources_used', []))}

### Retrieved Articles:
"""
        for idx, art in enumerate(news_rewriter.get("retrieved_articles", []), 1):
            news_md += f"\n{idx}. **[{art.get('title')}]({art.get('url')})** ({art.get('source')})\n   > {art.get('content')}\n"

    report_content = f"""# Explainable AI (XAI) Report: IntelliMoE Trace

This report documents the explainable decisions, routing logic, execution path, and API selections resolved for the latest query processed by IntelliMoE.

---

## 👤 User Request
> **{query}**

---

## ⛓️ Execution Flow Diagram

```text
{ascii_flow}
```

---

## 🧠 Reasoning Timeline Traces

{reasoning_md}

---

## ⏱️ Performance & Telemetry Summary

- **Total Execution Response Time**: {perf.get('response_time_ms', 0)} ms
- **Estimated Token Volume**: {perf.get('tokens', 0)} tokens
- **Fallback Triggered**: {str(router.get('fallback', True))}
- **API Provider Endpoint(s)**: {api.get('provider', 'Groq/Gemini')}
{news_md}
"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
