"""
tests/test_coding_expert.py
---------------------------
Unit tests for the CodingExpert module.

Run with:
    pytest tests/test_coding_expert.py -v -s

Flags:
    -v   verbose output (test names + pass/fail)
    -s   disable stdout capture so print() output is visible in the terminal
"""

import pytest

from experts.coding import CodingExpert


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def expert() -> CodingExpert:
    """
    Create a single CodingExpert instance shared across all tests in this
    module. ``scope="module"`` ensures the model is loaded only once,
    avoiding repeated expensive I/O during the test session.
    """
    return CodingExpert()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCodingExpert:
    """Test suite for CodingExpert."""

    # ------------------------------------------------------------------
    # Core functionality
    # ------------------------------------------------------------------

    def test_answer_returns_string(self, expert: CodingExpert) -> None:
        """answer() must return a non-empty string."""
        result = expert.answer("What is Python?")
        assert isinstance(result, str), "Expected answer to be a string."
        assert len(result) > 0, "Expected a non-empty answer."

    def test_what_is_python(self, expert: CodingExpert) -> None:
        """
        Integration-style test: ask 'What is Python?' and print the response.

        The response is validated to:
          - Be a non-empty string.
          - Contain at least one of the known keywords associated with Python.
        """
        question = "What is Python?"
        answer = expert.answer(question)

        # Print the response so it is visible with pytest -s.
        print("\n" + "=" * 60)
        print(f"Question : {question}")
        print("-" * 60)
        print(f"Answer   :\n{answer}")
        print("=" * 60)

        # Basic sanity checks on the response.
        assert isinstance(answer, str), "Answer must be a string."
        assert len(answer) > 10, "Answer is suspiciously short."

        # The response should mention at least one Python-related keyword.
        keywords = ["python", "programming", "language", "code", "scripting"]
        answer_lower = answer.lower()
        assert any(kw in answer_lower for kw in keywords), (
            f"Expected answer to mention Python-related terms. Got:\n{answer}"
        )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_empty_question_raises_value_error(self, expert: CodingExpert) -> None:
        """answer() must raise ValueError for blank input."""
        with pytest.raises(ValueError, match="must not be empty"):
            expert.answer("")

    def test_whitespace_only_raises_value_error(self, expert: CodingExpert) -> None:
        """answer() must raise ValueError for whitespace-only input."""
        with pytest.raises(ValueError, match="must not be empty"):
            expert.answer("   ")

    # ------------------------------------------------------------------
    # Output quality
    # ------------------------------------------------------------------

    def test_answer_does_not_contain_prompt_template(self, expert: CodingExpert) -> None:
        """
        The returned answer must not leak ChatML template tokens.
        Confirms that only newly generated tokens are decoded.
        """
        answer = expert.answer("What is Python?")
        assert "<|system|>" not in answer, "Response leaked system token."
        assert "<|user|>" not in answer, "Response leaked user token."
        assert "<|assistant|>" not in answer, "Response leaked assistant token."

    def test_answer_is_stripped(self, expert: CodingExpert) -> None:
        """answer() must return a string with no leading/trailing whitespace."""
        answer = expert.answer("What is Python?")
        assert answer == answer.strip(), "Answer has unexpected leading/trailing whitespace."
