# conversation_ai.md — Intelligent Conversational AI Layer

**Phase 30 — IntelliMoE Architecture Documentation**

---

## 1. Overview

The **Conversation AI Layer** transforms IntelliMoE from a pure expert-routing assistant into a natural conversational AI — similar to ChatGPT or Claude — while **preserving the entire Multi-Expert architecture untouched**.

It sits between the user's input and the Hybrid Router, acting as an intelligent gatekeeper:
- **Conversational queries** (greetings, small talk, follow-ups, general knowledge) → answered directly by the LLM with full context awareness.
- **Technical queries** (coding, math, ML, deep learning, research, system design) → forwarded to the existing Hybrid Router pipeline unchanged.

---

## 2. Architecture

```
User Input
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│               Conversation AI Layer                        │
│               conversation_ai/                             │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              IntentDetector (detector.py)            │  │
│  │                                                      │  │
│  │  Tier 1 ── Rule-based (regex)                        │  │
│  │             Greetings / Farewells / Small-talk        │  │
│  │             → Zero latency, instant classification   │  │
│  │                                                      │  │
│  │  Tier 2 ── ML Intent Classifier (primary)            │  │
│  │             Same scikit-learn model as Hybrid Router  │  │
│  │             confidence ≥ threshold → use result       │  │
│  │             confidence < threshold → call Tier 3     │  │
│  │                                                      │  │
│  │  Tier 3 ── LLM Tiebreaker (Groq)                    │  │
│  │             Called only when ML confidence is low    │  │
│  │             Mirrors HybridRouter's LLM fallback      │  │
│  └──────────────────────────┬──────────────────────────┘  │
│                             │                              │
│              IntentResult (intent, is_conversational,      │
│                             confidence, tier_used)         │
│                             │                              │
│                    ┌────────┴─────────┐                   │
│                    │                  │                    │
│              Conversational?       Technical?              │
│                    │                  │                    │
│  ┌─────────────────▼──────────────┐  │                    │
│  │  ConversationalResponder        │  │                    │
│  │  (responder.py)                 │  │                    │
│  │                                 │  │                    │
│  │  • Reads full ConversationMemory│  │                    │
│  │  • Includes prior expert answers│  │                    │
│  │  • Dynamic LLM personality      │  │                    │
│  │  • Never repeats templates      │  │                    │
│  └──────────────────────┬──────────┘  │                    │
│                         │             │                    │
└─────────────────────────┼─────────────┼────────────────────┘
                          │             │
                          ▼             ▼
                   Direct Reply    Hybrid Router
                   (💬 badge)      ──────────────────────────
                                   ML Classifier
                                   → Decision Engine
                                   → Planner
                                   → Orchestrator
                                   → Experts (parallel)
```

---

## 3. Module Structure

```
conversation_ai/
├── __init__.py       Public API exports
├── detector.py       IntentDetector — 3-tier hybrid classification
├── responder.py      ConversationalResponder — LLM reply generator
└── layer.py          ConversationLayer — main entry point
```

---

## 4. Intent Detection Flow

### 13 Intent Types

| Intent | Category | Routing |
|--------|----------|---------|
| `greeting` | Conversational | Direct LLM reply |
| `farewell` | Conversational | Direct LLM reply |
| `small_talk` | Conversational | Direct LLM reply |
| `general_knowledge` | Conversational | Direct LLM reply |
| `follow_up` | Conversational | Direct LLM reply (with prior context) |
| `clarification` | Conversational | Direct LLM reply (re-explains prior expert answer) |
| `coding` | Technical | → Hybrid Router → Coding Expert |
| `math` | Technical | → Hybrid Router → Math Expert |
| `machine_learning` | Technical | → Hybrid Router → ML Expert |
| `deep_learning` | Technical | → Hybrid Router → Deep Learning Expert |
| `research` | Technical | → Hybrid Router → Research Expert |
| `system_design` | Technical | → Hybrid Router → System Design Expert |
| `technical_general` | Technical | → Hybrid Router → Best Expert |

### Detection Tiers

```
Query received
     │
     ▼
┌─────────────────────────────────────────────────┐
│ TIER 1 — Rule-based (regex, zero latency)        │
│                                                  │
│  Patterns for: greeting, farewell, small_talk,   │
│  follow_up*, clarification*  (*only if context   │
│  already exists in ConversationMemory)           │
│                                                  │
│  Match found? → Return immediately               │
│  No match?   → Fall through to Tier 2           │
└─────────────────────────┬───────────────────────┘
                          │ No match
                          ▼
┌─────────────────────────────────────────────────┐
│ TIER 2 — ML Intent Classifier (primary)          │
│                                                  │
│  • Same MLClassifierRouter used by HybridRouter  │
│  • Same scikit-learn model + TF-IDF vectorizer   │
│  • Returns (expert_label, confidence, probs)     │
│                                                  │
│  confidence ≥ ML_ROUTING_CONFIDENCE_THRESHOLD?   │
│     YES → Map expert_label to IntentType         │
│            Return result                         │
│     NO  → Fall through to Tier 3                │
└─────────────────────────┬───────────────────────┘
                          │ Low confidence
                          ▼
┌─────────────────────────────────────────────────┐
│ TIER 3 — LLM Tiebreaker (Groq)                  │
│                                                  │
│  • Asks llama-3.1-8b-instant to pick ONE label  │
│  • System prompt lists all 13 intent labels      │
│  • temperature=0.1 for deterministic output      │
│  • Parses single label from response             │
│                                                  │
│  Success → Return LLM result (confidence=0.75)  │
│  Failure → Heuristic fallback (word count)       │
└─────────────────────────────────────────────────┘
```

---

## 5. Conversation Flow

### Greeting Example
```
User: "Hi!"
  → Tier 1 rule matches GREETING pattern (0ms)
  → is_conversational=True, confidence=0.97
  → ConversationalResponder called with:
       - intent: "greeting"
       - memory: [] (empty)
       - is_first_message: True
  → LLM generates unique greeting (never repeated)
  → Response: "Hey there! 👋 Great to see you — what are you working on today?"
  → 💬 Conversational badge shown in UI
  → Memory updated: memory.add_turn(query, response, expert="conversational")
```

### Follow-up Example
```
[Context: User asked about Python decorators, Coding Expert answered]

User: "Can you explain that more simply?"
  → Tier 1: matches CLARIFICATION pattern (has_prior_context=True)
  → is_conversational=True, confidence=0.85
  → ConversationalResponder reads last 6 turns from ConversationMemory
  → Builds context string including prior Coding Expert answer
  → LLM re-explains decorators in simpler terms, referencing prior answer
  → 💬 Conversational badge shown
```

### Technical Routing Example
```
User: "Write a Python function to parse JSON files"
  → Tier 1: no pattern match
  → Tier 2: ML predicts "coding" (confidence=0.89 ≥ 0.60)
  → is_conversational=False
  → ConversationLayer returns immediately (no LLM call)
  → app.py routes to Hybrid Router → Decision Engine → Coding Expert
  → 💻 Coding badge shown (existing behavior, unchanged)
```

---

## 6. Memory Integration

The ConversationLayer reads from the **existing** `ConversationMemory` object:

```python
# ConversationalResponder._build_conversation_context()

turns = memory.get_turns()          # All prior turns
for turn in turns[-6:]:             # Last 6 for context
    lines.append(f"User: {turn.question}")
    
    # Include expert information for rich follow-up context
    expert_names = ", ".join(e.replace("_", " ").title()
                             for e in turn.experts
                             if e != "conversational")
    if expert_names:
        label = f" [via {expert_names} Expert]"
    
    lines.append(f"Assistant{label}: {turn.answer[:600]}")
```

The context includes:
- ✅ Previous user messages
- ✅ Previous assistant messages (both conversational and expert)
- ✅ Which experts were activated per turn
- ✅ Expert response content (truncated to 600 chars for context efficiency)

---

## 7. Integration with Hybrid Router

The integration point is **one place only** — `ui/app.py._handle_query()`:

```python
def _handle_query(query: str) -> None:
    # ...setup...
    
    # ── Phase 30: Conversation AI Layer ─────────────
    if not img_path:  # Images → Vision Expert always
        try:
            conv_result = ConversationLayer().process(query, memory)
            
            if conv_result.is_conversational:
                memory.add_turn(query, conv_result.response, expert="conversational")
                # ...populate analytics metadata...
                st.session_state.just_generated = True
                return  # ← Exits before Hybrid Router
                
        except Exception:
            pass  # Graceful fallback to expert routing
    # ── END Conversation AI Layer ────────────────────
    
    # Everything below is UNCHANGED:
    # Hybrid Router → Decision Engine → Planner → Orchestrator → Experts
```

### Untouched Modules (Zero Modifications)

| Module | Status |
|--------|--------|
| `router/hybrid_router.py` | ✅ Untouched |
| `router/ml_classifier_router.py` | ✅ Untouched (reused) |
| `router/decision_engine.py` | ✅ Untouched |
| `router/planner.py` | ✅ Untouched |
| `router/orchestrator.py` | ✅ Untouched |
| `experts/*` | ✅ Untouched |
| `benchmark/*` | ✅ Untouched |
| `explainability/*` | ✅ Untouched |
| `evaluation/*` | ✅ Untouched |
| `services/*` | ✅ Untouched (reused) |

---

## 8. Graceful Degradation

If the Conversation AI Layer raises any exception:
1. The exception is caught silently.
2. A warning is logged.
3. The query falls through to the existing Hybrid Router pipeline.
4. The user experience is unaffected.

This means the Conversation AI Layer **cannot break** the existing system.

---

## 9. Files Created / Modified

### New Files

| File | Purpose |
|------|---------|
| `conversation_ai/__init__.py` | Public API exports |
| `conversation_ai/detector.py` | 3-tier intent classifier (rule → ML → LLM) |
| `conversation_ai/responder.py` | Context-aware LLM responder with memory |
| `conversation_ai/layer.py` | Main entry point / orchestrator |
| `conversation_ai.md` | This architecture document |

### Modified Files

| File | Change |
|------|--------|
| `ui/app.py` | Added `"conversational"` to `EXPERT_META`; inserted Conversation AI Layer intercept in `_handle_query()` |
