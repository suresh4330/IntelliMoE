"""
experts/research.py
--------------------
ResearchExpert — RAG-powered academic research domain expert.

Upgrade over the base expert:
  - Uses ChromaDB (via ResearchVectorStore) to retrieve relevant research
    paper chunks based on the user's question.
  - Retrieved context is injected into the ChatML user turn BEFORE the
    question, following the standard RAG prompt pattern.
  - Falls back gracefully to prompt-only answering when no relevant
    documents are found (e.g. empty DB or low-similarity results).
  - All ChromaDB and embedding I/O is fully contained in this class;
    the rest of the system is unaffected.

RAG prompt structure:
  <|system|>
  {research system prompt from prompts/research.txt}</s>
  [history turns if memory provided]
  <|user|>
  === Retrieved Research Context ===
  [Source: paper_name.txt | Relevance: 0.82]
  {chunk_1_text}

  [Source: paper_name.txt | Relevance: 0.76]
  {chunk_2_text}
  ===================================
  Using the above research context, answer the following question:
  {question}</s>
  <|assistant|>

SOLID:
  - SRP: ResearchExpert is the only class responsible for research RAG logic.
  - OCP: Changing the retrieval strategy only requires replacing _retrieve().
  - LSP: Fully substitutable for BaseExpert via the answer(question) interface.
  - DIP: Depends on ResearchVectorStore abstraction, not ChromaDB directly.
"""

import logging
from typing import Optional, TYPE_CHECKING

from config.settings import EXPERT_CONFIGS, GEMINI_MODEL_ID, GenerationConfig
from experts.base import BaseExpert
from services.gemini_client import generate_response
from utils.vector_store import ResearchVectorStore

if TYPE_CHECKING:
    from utils.memory import ConversationMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retrieval configuration
# ---------------------------------------------------------------------------

# Number of chunks to retrieve per query.
TOP_K_CHUNKS: int = 3

# Minimum cosine distance threshold (0 = identical, 2 = opposite).
# Chunks with distance > MAX_DISTANCE are considered irrelevant and excluded.
MAX_DISTANCE: float = 1.2

# Character limit for each retrieved chunk shown in the prompt.
# Keeps the context block from blowing up the context window.
MAX_CHUNK_CHARS_IN_PROMPT: int = 500


class ResearchExpert(BaseExpert):
    """
    RAG-powered expert for academic research, paper analysis, and literature review.

    On every call to answer():
      1. ResearchVectorStore retrieves the top-K most similar paper chunks.
      2. Relevant chunks (below distance threshold) are formatted as context.
      3. Context + question are passed to Gemini API via augmented prompt.
      4. If no relevant chunks are found, Gemini answers from its own knowledge.

    Parameters
    ----------
    top_k : int
        Number of paper chunks to retrieve per query.
    max_distance : float
        Maximum acceptable cosine distance for a chunk to be included.
    """

    def __init__(
        self,
        top_k: int = TOP_K_CHUNKS,
        max_distance: float = MAX_DISTANCE,
    ) -> None:
        super().__init__()
        self._top_k        = top_k
        self._max_distance = max_distance
        self._vector_store = ResearchVectorStore(top_k=top_k, auto_seed=True)
        self.last_retrieved_chunks: list[dict] = []

    # ------------------------------------------------------------------
    # BaseExpert abstract interface
    # ------------------------------------------------------------------

    @property
    def prompt_name(self) -> str:
        return "research"

    @property
    def generation_config(self) -> GenerationConfig:
        return EXPERT_CONFIGS["research"]

    # ------------------------------------------------------------------
    # Override public answer: execute RAG using Gemini API
    # ------------------------------------------------------------------

    def answer(self, question: str, memory: "Optional[ConversationMemory]" = None) -> str:
        """
        Generate a research-expert answer using the Gemini API.

        This overrides the parent class method to bypass local model loading and
        use Gemini client for inference while preserving the same interface.
        """
        question = self._validate_question(question)

        # Lazy-load system prompt using inherited PromptManager
        if self._system_prompt is None:
            self._system_prompt = self._prompt_manager.get_prompt(self.prompt_name)

        # Retrieve context from ChromaDB via our ResearchVectorStore
        context_block = self._retrieve(question)

        # Build prompt incorporating history and context
        if memory and not memory.is_empty:
            history_text = ""
            for turn in memory.get_turns():
                history_text += f"User: {turn.question}\nAssistant: {turn.answer}\n\n"
            
            if context_block:
                user_content = (
                    f"Conversation history:\n{history_text}"
                    f"{context_block}\n"
                    f"INSTRUCTION: Answer the question below strictly using only the retrieved research context. "
                    f"Be factual and cite your sources. If the answer is not supported by the context, state: "
                    f"'I am sorry, but the provided context does not contain the answer.'\n"
                    f"Question: {question}"
                )
                logger.info(
                    "ResearchExpert: augmented prompt with %d retrieved chunk(s) and conversation history.",
                    context_block.count("[Source:"),
                )
            else:
                user_content = f"Conversation history:\n{history_text}User: {question}"
                logger.info("ResearchExpert: no relevant chunks found — using conversation history.")
        else:
            if context_block:
                user_content = (
                    f"{context_block}\n"
                    f"INSTRUCTION: Answer the question below strictly using only the retrieved research context. "
                    f"Be factual and cite your sources. If the answer is not supported by the context, state: "
                    f"'I am sorry, but the provided context does not contain the answer.'\n"
                    f"Question: {question}"
                )
                logger.info(
                    "ResearchExpert: augmented prompt with %d retrieved chunk(s).",
                    context_block.count("[Source:"),
                )
            else:
                user_content = question
                logger.info("ResearchExpert: no relevant chunks found — answering from model knowledge.")

        cfg = self.generation_config
        logger.info("ResearchExpert: sending request to Gemini API...")

        try:
            # Generate the response using Gemini client
            response = generate_response(
                prompt=user_content,
                system_prompt=self._system_prompt,
                model=GEMINI_MODEL_ID,
                temperature=cfg.temperature,
            )

            # Update token metrics for telemetry/cost tracking
            system_len = len(self._system_prompt) if self._system_prompt else 0
            self.last_prompt_tokens = (len(user_content) + system_len) // 4
            self.last_tokens_generated = len(response) // 4

            logger.info("ResearchExpert: successfully generated response from Gemini API.")
            return response

        except Exception as exc:
            logger.exception("ResearchExpert failed to generate answer using Gemini API.")
            raise RuntimeError(
                f"ResearchExpert failed to generate an answer. "
                f"Cause: {type(exc).__name__}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # RAG-specific public methods
    # ------------------------------------------------------------------

    def add_paper(self, text: str, source: str) -> int:
        """
        Index a new research paper into the vector store at runtime.

        Parameters
        ----------
        text : str
            Full paper text to chunk and embed.
        source : str
            Unique identifier for this paper (e.g. filename or DOI).

        Returns
        -------
        int
            Number of new chunks added to ChromaDB.
        """
        added = self._vector_store.add_paper(text, source)
        logger.info("ResearchExpert: added %d chunks for '%s'.", added, source)
        return added

    def add_pdf(self, file_bytes: bytes, filename: str) -> int:
        """
        Index a new PDF research paper into the vector store.

        Parameters
        ----------
        file_bytes : bytes
            Binary bytes of the PDF.
        filename : str
            Unique filename.

        Returns
        -------
        int
            Number of chunks added.
        """
        added = self._vector_store.add_pdf(file_bytes, filename)
        logger.info("ResearchExpert: added %d chunks from PDF '%s'.", added, filename)
        return added

    @property
    def indexed_papers(self) -> list[str]:
        """List of paper source names currently in the vector store."""
        return self._vector_store.paper_sources

    @property
    def chunk_count(self) -> int:
        """Total number of chunks in the vector store."""
        return self._vector_store.document_count

    # ------------------------------------------------------------------
    # Internal: retrieval + formatting
    # ------------------------------------------------------------------

    def _retrieve(self, question: str) -> str:
        """
        Query ChromaDB and format the retrieved chunks as a context block.

        Chunks exceeding ``MAX_DISTANCE`` are filtered out as irrelevant.
        Returns an empty string if no relevant chunks exist.
        """
        self.last_retrieved_chunks = []
        try:
            chunks = self._vector_store.query(question, top_k=self._top_k)
        except Exception as exc:
            logger.error("ResearchExpert: retrieval failed: %s", exc)
            return ""

        # Filter chunks below the relevance threshold.
        relevant = [c for c in chunks if c["distance"] <= self._max_distance]

        if not relevant:
            return ""

        # Record last retrieved chunks for ui visualization
        self.last_retrieved_chunks = relevant

        lines = ["=== Retrieved Research Context ==="]
        for chunk in relevant:
            # Cosine similarity from distance: sim = 1 - (distance / 2)
            relevance = 1.0 - (chunk["distance"] / 2.0)
            text      = chunk["text"][:MAX_CHUNK_CHARS_IN_PROMPT].strip()
            if len(chunk["text"]) > MAX_CHUNK_CHARS_IN_PROMPT:
                text += " [...]"
            lines.append(
                f"\n[Source: {chunk['source']} | Relevance: {relevance:.2f}]\n{text}"
            )
        lines.append("\n===================================\n")

        return "\n".join(lines)
