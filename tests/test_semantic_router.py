"""
tests/test_semantic_router.py
-----------------------------
Phase 37 automated verification test suite for the Semantic Router integration.
Checks single-expert and multi-expert routing predictions.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from router.router import ExpertRouter, ExpertName

def run_tests():
    router = ExpertRouter()
    
    test_cases = [
        # Conversational fallback queries
        ("Hi", []),
        ("Hello", []),
        ("Thanks", []),
        
        # Single-expert technical queries
        ("Hey explain Binary Search", [ExpertName.CODING]),
        ("Hello solve integration", [ExpertName.MATH]),
        ("Hi compare Random Forest and XGBoost", [ExpertName.ML]),
        ("Hey explain Transformers", [ExpertName.DEEP_LEARNING]),
        ("Hi summarize Attention Is All You Need", [ExpertName.RESEARCH]),
        ("Hello design Instagram backend", [ExpertName.SYSTEM_DESIGN]),
        ("Hi explain Prompt Engineering", [ExpertName.GENAI]),
        ("Describe this uploaded image", [ExpertName.VISION]),
        ("Hey who won today's ODI match", [ExpertName.NEWS]),
        ("Today's stock market analysis", [ExpertName.NEWS]),
        
        # Multi-expert query (Deep Learning + Math)
        ("Hello explain Transformers with equations", [ExpertName.DEEP_LEARNING, ExpertName.MATH]),
        
        # Multi-expert query (Deep Learning + Math + Coding)
        ("Hey explain Transformers mathematically and implement it in PyTorch", [ExpertName.DEEP_LEARNING, ExpertName.MATH, ExpertName.CODING])
    ]
    
    print("--- RUNNING PHASE 37 SEMANTIC INTENT ROUTER TESTS ---")
    all_passed = True
    
    for query, expected_experts in test_cases:
        actual_experts = router.selected_experts(query)
        # Note: Conversational fallback returns [] or forwards to Conversation Layer which is handled at route() / app.py level.
        # But wait! If semantic router has no confidence, it falls back to Hybrid Router, which will predict coding/ml/etc.
        # Wait, for "Hi", the clean query is empty, so Semantic Router returns []. Since fallback triggers, hybrid router returns [ExpertName.CODING] (fallback).
        # Wait! If the user query is "Hi", let's check: does the Conversation Layer intercept it BEFORE it reaches the router?
        # Yes! In ui/app.py:
        # if not img_path:
        #     conv_result = _conv_layer.process(query, memory)
        #     if conv_result.is_conversational:
        #         # Answered conversationally - never hits ExpertRouter!
        # So "Hi", "Hello", "Thanks" are intercepted by Conversation Layer and never reach the router!
        # But what if we check the other technical cases?
        # Let's inspect that they contain the expected experts:
        if not expected_experts:
            # For purely conversational queries, verify they are correctly detected by Conversation Layer!
            from conversation_ai.layer import ConversationLayer
            from utils.memory import ConversationMemory
            conv_layer = ConversationLayer()
            mem = ConversationMemory()
            res = conv_layer.process(query, mem)
            passed = res.is_conversational
            print(f"Query: '{query}' -> Conversational? {res.is_conversational} (Intent: {res.intent}) -> {'PASS' if passed else 'FAIL'}")
            if not passed:
                all_passed = False
        else:
            # Check actual experts returned by the router
            passed = all(exp in actual_experts for exp in expected_experts)
            print(f"Query: '{query}'")
            print(f"  Expected: {[e.value for e in expected_experts]}")
            print(f"  Actual  : {[e.value for e in actual_experts]} -> {'PASS' if passed else 'FAIL'}\n")
            if not passed:
                all_passed = False
                
    if all_passed:
        print("ALL ROUTING TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("SOME ROUTING TESTS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
