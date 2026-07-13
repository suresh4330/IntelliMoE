"""
tests/test_gemini.py
--------------------
Integration test to verify Gemini API connection and response generation.
Supports execution via pytest (pytest tests/test_gemini.py -s) or as a direct python script.
"""

import os
import sys

# Ensure workspace root is in python path when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import GEMINI_API_KEY, GEMINI_MODEL_ID
from services.gemini_client import generate_response


def test_gemini_connection():
    """
    Test reading the Gemini API key, sending 'Hello', and printing the response.
    """
    # 1. Read Gemini API key
    assert GEMINI_API_KEY is not None, "GEMINI_API_KEY must be loaded from settings."
    masked_key = f"{GEMINI_API_KEY[:6]}...{GEMINI_API_KEY[-4:]}" if len(GEMINI_API_KEY) > 10 else "Too Short"
    print(f"\n[Test] Read Gemini API Key: {masked_key}")

    try:
        # 2. Send 'Hello' to Gemini
        print("[Test] Sending 'Hello' to Gemini model...")
        response = generate_response(prompt="Hello", model=GEMINI_MODEL_ID)

        # 3. Print response
        print("\n" + "=" * 60)
        print("Gemini API Response:")
        print("-" * 60)
        print(response)
        print("=" * 60 + "\n")

        assert response is not None, "Response must not be None."
        assert len(response.strip()) > 0, "Response must not be empty."

    # 4. Handle exceptions
    except Exception as exc:
        print(f"[Test] Failed with exception: {exc}", file=sys.stderr)
        raise exc


if __name__ == "__main__":
    print("Running Gemini API test directly...")
    try:
        test_gemini_connection()
        print("Gemini API integration test passed.")
    except Exception as e:
        print(f"Gemini API integration test failed: {e}")
        sys.exit(1)
