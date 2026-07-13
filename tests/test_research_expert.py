"""
tests/test_research_expert.py
-----------------------------
Unit and integration tests for the RAG-enabled ResearchExpert.
"""

import pytest
from pathlib import Path
from experts.research import ResearchExpert
from utils.vector_store import ResearchVectorStore

@pytest.fixture(scope="module")
def research_expert() -> ResearchExpert:
    """Create a single ResearchExpert instance shared across tests."""
    return ResearchExpert(top_k=2)

class TestResearchExpert:
    """Test suite for ResearchExpert RAG functionality."""

    def test_vector_store_initialization(self):
        """Verify the vector store can load and has seeded documents."""
        vs = ResearchVectorStore(auto_seed=True)
        # Ensure we have indexed chunks
        count = vs.document_count
        assert count > 0, "Vector store should have chunks after auto-seeding."
        
        # Verify specific papers are indexed
        sources = vs.paper_sources
        assert "attention_is_all_you_need.txt" in sources
        assert "retrieval_augmented_generation.txt" in sources

    def test_retrieval_context_injection(self, research_expert: ResearchExpert):
        """Verify the retrieved context is properly generated and formatted."""
        question = "What is the formula for scaled dot-product attention?"
        context = research_expert._retrieve(question)
        
        # Context must contain headers and content from the paper
        assert "=== Retrieved Research Context ===" in context
        assert "attention_is_all_you_need.txt" in context
        assert "Attention(Q, K, V)" in context or "softmax" in context

    def test_answer_generation(self, research_expert: ResearchExpert):
        """Verify the expert can answer questions using the RAG workflow."""
        question = "Explain scaled dot-product attention according to Attention Is All You Need."
        answer = research_expert.answer(question)
        
        print("\n" + "=" * 60)
        print(f"Question : {question}")
        print("-" * 60)
        print(f"Answer   :\n{answer}")
        print("=" * 60)

        assert isinstance(answer, str)
        assert len(answer) > 10
        # The answer should mention key terms from the retrieved paper
        assert "attention" in answer.lower() or "softmax" in answer.lower()
