"""
tests/test_news.py
------------------
Automated verification test suite for the redesigned search layer and NewsExpert.
Tests abstract SearchService, Tavily/Serper providers, and direct answer display.
"""

import sys
import os

# Reconfigure stdout to support unicode/emojis on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure workspace root is in python path when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from experts.news import NewsExpert
from services.search_client import SearchService, TavilySearchProvider, SerperSearchProvider

def test_search_providers_and_news_expert():
    print("--- RUNNING REDESIGNED SEARCH LAYER & NEWS EXPERT TESTS ---")
    
    # 1. Test abstract SearchService initialization and fallback setup
    service = SearchService()
    assert len(service._providers) == 2, "SearchService should default to 2 providers."
    assert isinstance(service._providers[0], TavilySearchProvider), "Primary provider should be TavilySearchProvider."
    assert isinstance(service._providers[1], SerperSearchProvider), "Fallback provider should be SerperSearchProvider."
    print("  -> Abstract SearchService configuration: PASS")

    # 2. Test NewsExpert answer capability using the search layer
    expert = NewsExpert()
    
    # Verify properties
    assert expert.prompt_name == "news"
    print("  -> NewsExpert properties configuration: PASS")

    # Test answering query directly
    # Since Tavily key is configured in .env, this should succeed.
    try:
        response = expert.answer("what is the latest breaking news")
        print("\n" + "=" * 60)
        print("News Expert Live Search Result Preview:")
        print("-" * 60)
        print(response[:400] + "...")
        print("=" * 60 + "\n")
        
        assert response is not None, "Response was None"
        assert len(response.strip()) > 0, "Response was empty"
        print("  -> NewsExpert.answer execution: PASS")
    except Exception as exc:
        pytest.fail(f"NewsExpert.answer failed: {exc}")

if __name__ == "__main__":
    print("Running redesigned NewsExpert test directly...")
    try:
        test_search_providers_and_news_expert()
        print("NewsExpert test passed successfully.")
    except Exception as e:
        print(f"NewsExpert test failed: {e}")
        sys.exit(1)
