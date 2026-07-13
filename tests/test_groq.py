"""
tests/test_groq.py
------------------
Integration test to verify Groq API connection and response generation.
Supports execution via pytest (pytest tests/test_groq.py -s) or as a direct python script.
"""

import os
import sys

# Ensure workspace root is in python path when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import GROQ_API_KEY
from services.groq_client import generate_response


def test_groq_connection():
    """
    Test reading the Groq API key, sending 'Hello', and printing the response.
    """
    # 1. Read Groq API key
    assert GROQ_API_KEY is not None, "GROQ_API_KEY must be loaded from settings."
    masked_key = f"{GROQ_API_KEY[:6]}...{GROQ_API_KEY[-4:]}" if len(GROQ_API_KEY) > 10 else "Too Short"
    print(f"\n[Test] Read Groq API Key: {masked_key}")

    try:
        # 2. Send 'Hello' to the model
        print("[Test] Sending 'Hello' to Groq model...")
        response = generate_response(prompt="Hello", model="llama-3.1-8b-instant")

        # 3. Print response
        print("\n" + "=" * 60)
        print("Groq API Response:")
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
    print("Running Groq API test directly...")
    try:
        test_groq_connection()
        print("Groq API integration test passed.")
    except Exception as e:
        print(f"Groq API integration test failed: {e}")
        sys.exit(1)
