"""
tests/test_query_rewriter.py
-----------------------------
Phase 39 automated verification test suite for the Intelligent Query Rewriter.
Verifies spelling corrections, abbreviation expansion, and intent preservation.
"""

import sys
import os

# Ensure workspace root is in python path when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from news.query_rewriter import NewsQueryRewriter

def test_query_rewriter():
    print("--- RUNNING PHASE 39 INTENT REWRITER TESTS ---")
    rewriter = NewsQueryRewriter()

    test_cases = [
        ("Hey who won today's ODI match?", "Who won today's ODI cricket match today?"),
        ("todayy they is match or not odi", "Are there any ODI cricket matches scheduled today?"),
        ("deputy cm of andhar pradesh know", "Current Deputy Chief Minister of Andhra Pradesh"),
        ("latest ai update", "Latest Artificial Intelligence news today"),
        ("hello stock market", "Today's stock market news"),
    ]

    for original, expected in test_cases:
        rewritten = rewriter.rewrite(original)
        print(f"Original : '{original}'")
        print(f"Expected : '{expected}'")
        print(f"Rewritten: '{rewritten}'\n")
        assert rewritten == expected

if __name__ == "__main__":
    test_query_rewriter()
