"""
explainability/graph.py
-----------------------
Execution graph compiler for Explainable AI Engine.
Generates Mermaid and ASCII visual query processing flow diagrams.
"""

from typing import Dict, Any

def _safe_upper(val: Any) -> str:
    """Helper to convert any value to string and return upper case safely."""
    if val is None:
        return "N/A"
    return str(val).strip().upper()

def generate_mermaid_graph(explanation: Dict[str, Any]) -> str:
    """
    Construct a visual Mermaid flowchart diagram representing execution steps.
    """
    query = explanation.get("query", "User Query")
    router_pred = explanation.get("router", {}).get("prediction", "N/A")
    router_used = "ML Router" if not explanation.get("router", {}).get("fallback", True) else "LLM Fallback Router"
    decision = explanation.get("decision_engine", {})
    primary = decision.get("primary_expert", "N/A")
    additionals = decision.get("additional_experts", [])
    order = explanation.get("planner", {}).get("execution_order", [])
    
    # Escape query text for safe inclusion in node labels
    escaped_query = str(query).replace('"', '\\"').replace("\n", " ")
    if len(escaped_query) > 55:
        escaped_query = escaped_query[:52] + "..."
        
    lines = [
        "graph TD",
        f'  User["👤 {escaped_query}"] --> Router["🤖 {router_used}"]',
        f'  Router --> Primary["💼 Primary: {_safe_upper(primary)}"]'
    ]
    
    prev_node = "Primary"
    if additionals:
        lines.append(f'  Router --> DecEngine["🧠 Decision Engine: Multi-Expert"]')
        for idx, add in enumerate(additionals):
            if add:
                lines.append(f'  DecEngine --> Add{idx}["💼 Additional: {_safe_upper(add)}"]')
                lines.append(f'  Add{idx} --> Exec["⛓️ Planner Scheduler"]')
        lines.append('  Primary --> Exec')
        prev_node = "Exec"
        
    # Compile step order nodes
    order_nodes = []
    for idx, exp in enumerate(order):
        if exp:
            node_id = f"Step{idx}"
            order_nodes.append(node_id)
            prov = "Groq" if str(exp).lower() in ["coding", "math"] else "Gemini"
            lines.append(f'  {node_id}["💼 {_safe_upper(exp)} ({prov})"]')
        
    if len(order_nodes) > 1:
        lines.append(f"  {prev_node} --> {order_nodes[0]}")
        for i in range(len(order_nodes) - 1):
            lines.append(f"  {order_nodes[i]} --> {order_nodes[i+1]}")
        last_step = order_nodes[-1]
    elif order_nodes:
        lines.append(f"  {prev_node} --> Step0")
        last_step = "Step0"
    else:
        last_step = prev_node
        
    lines.append(f'  {last_step} --> Agg["🏁 Aggregator"]')
    lines.append('  Agg --> Out["📥 Output Answer"]')
    
    return "\n".join(lines)


def generate_ascii_graph(explanation: Dict[str, Any]) -> str:
    """
    Construct an ASCII flowchart mapping the execution stages.
    """
    query = explanation.get("query", "User Query")
    router_pred = explanation.get("router", {}).get("prediction", "N/A")
    decision = explanation.get("decision_engine", {})
    primary = decision.get("primary_expert", "N/A")
    additionals = decision.get("additional_experts", [])
    order = explanation.get("planner", {}).get("execution_order", [])
    
    escaped_query = str(query).replace("\n", " ")
    if len(escaped_query) > 40:
        escaped_query = escaped_query[:37] + "..."
        
    conf = explanation.get("router", {}).get("confidence", 0.0)
    
    flow = f"User Query: \"{escaped_query}\"\n"
    flow += "   ↓\n"
    flow += f"ML Classifier Prediction: {_safe_upper(router_pred)} (Confidence: {float(conf)*100:.1f}%)\n"
    flow += "   ↓\n"
    
    if additionals:
        clean_adds = [add for add in additionals if add]
        flow += f"Decision Engine (Multi-Expert): Primary: {_safe_upper(primary)} + Additionals: {', '.join(map(_safe_upper, clean_adds))}\n"
        flow += "   ↓\n"
        flow += f"Planner Scheduler (Sequence: {' ➡️ '.join(map(_safe_upper, order))})\n"
    else:
        flow += f"Decision Engine (Single Expert): {_safe_upper(primary)}\n"
        
    flow += "   ↓\n"
    
    # Map APIs
    api_steps = []
    for exp in order:
        if exp:
            prov = "Groq" if str(exp).lower() in ["coding", "math"] else "Gemini"
            api_steps.append(f"{_safe_upper(exp)} via {prov}")
        
    flow += f"API Inference: {' + '.join(api_steps)}\n"
    flow += "   ↓\n"
    flow += "Response Aggregator\n"
    flow += "   ↓\n"
    flow += "Answer Output\n"
    
    return flow
