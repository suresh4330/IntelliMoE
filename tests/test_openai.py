import sys
from pathlib import Path
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from services.openai_client import generate_response

def test_openai_response():
    try:
        response = generate_response(prompt="Hello", model="gpt-4o-mini")
        assert len(response) > 0
    except Exception as e:
        if "insufficient_quota" in str(e) or "429" in str(e) or "invalid_api_key" in str(e):
            pytest.skip(f"OpenAI API quota/auth limitation: {e}")
        else:
            pytest.fail(f"OpenAI API call failed: {e}")
