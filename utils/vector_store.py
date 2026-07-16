"""
utils/vector_store.py
---------------------
ChromaDB-backed vector store for IntelliMoE's Research Expert (RAG).

Pipeline:
  1. Research papers (.txt) are loaded from ``data/papers/``.
  2. Each paper is split into overlapping chunks (fixed-size sliding window).
  3. Chunks are embedded using ``sentence-transformers/all-MiniLM-L6-v2``
     (fast 22M-param model; 384-dim embeddings; MIT licence).
  4. Embeddings + metadata are persisted in ChromaDB at ``data/chroma_db/``.
  5. At query time, the top-K most similar chunks are retrieved and returned
     as plain strings ready to inject into the TinyLlama prompt.

Design decisions:
  - Singleton ChromaDB client (one connection per process).
  - Lazy seeding: papers are only indexed when the collection is empty,
    so the first query may be slower; subsequent startups are instant.
  - SentenceTransformer is loaded once and reused for all embed calls.
  - ChromaDB's built-in cosine similarity is used for retrieval.
  - Metadata stored per chunk: source filename, chunk index, char offsets.

SOLID:
  - SRP : ResearchVectorStore owns only ChromaDB I/O. Chunking logic is
          a separate private function (_chunk_text).
  - OCP : New embedding models can be swapped by changing EMBEDDING_MODEL.
  - DIP : ResearchExpert depends on ResearchVectorStore (abstraction), not
          directly on ChromaDB or sentence-transformers.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Lightweight, fast embedding model (22M params, 384-dim, MIT licence).
EMBEDDING_MODEL: str = os.getenv(
    "INTELLIMOE_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# Persistent ChromaDB storage directory (relative to project root).
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR: Path = _PROJECT_ROOT / "data" / "chroma_db"

# Research papers directory.
PAPERS_DIR: Path = _PROJECT_ROOT / "data" / "papers"

# ChromaDB collection name.
COLLECTION_NAME: str = "research_papers"

# Text chunking parameters.
CHUNK_SIZE: int    = 600   # characters per chunk
CHUNK_OVERLAP: int = 100   # overlapping characters between consecutive chunks

# Number of chunks to retrieve per query.
DEFAULT_TOP_K: int = 3


# ---------------------------------------------------------------------------
# Module-level singletons (lazy initialised)
# ---------------------------------------------------------------------------

_chroma_client   = None
_embed_model     = None


def _get_chroma_client():
    """Return the singleton ChromaDB persistent client."""
    global _chroma_client
    if _chroma_client is None:
        import chromadb  # noqa: PLC0415
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        logger.info("ChromaDB client initialised at: %s", CHROMA_PERSIST_DIR)
    return _chroma_client


def _get_embed_model():
    """Return the singleton SentenceTransformer embedding model."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        logger.info("Loading embedding model '%s' …", EMBEDDING_MODEL)
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded (dim=%d).", _embed_model.get_sentence_embedding_dimension())
    return _embed_model


# ---------------------------------------------------------------------------
# Text chunking helper
# ---------------------------------------------------------------------------

def _chunk_text(
    text: str,
    source: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split ``text`` into overlapping fixed-size chunks using LangChain's RecursiveCharacterTextSplitter.

    Parameters
    ----------
    text : str
        Full document text to chunk.
    source : str
        Filename or identifier to store as metadata.
    chunk_size : int
        Target character length per chunk.
    overlap : int
        Number of characters shared between consecutive chunks.

    Returns
    -------
    list[dict]
        Each dict has keys: ``text``, ``source``, ``chunk_index``,
        ``start_char``, ``end_char``.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: PLC0415
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    
    docs = splitter.create_documents([text.strip()], metadatas=[{"source": source}])
    
    chunks = []
    for index, doc in enumerate(docs):
        chunks.append({
            "text":        doc.page_content,
            "source":      source,
            "chunk_index": index,
            "start_char":  0,
            "end_char":    len(doc.page_content),
        })
    return chunks


# ---------------------------------------------------------------------------
# ResearchVectorStore
# ---------------------------------------------------------------------------

class ResearchVectorStore:
    """
    ChromaDB-backed vector store for research papers.

    Responsibilities:
      - Seed the collection from ``data/papers/*.txt`` on first run.
      - Embed and persist new documents via ``add_paper()``.
      - Retrieve the top-K most relevant chunks via ``query()``.

    Parameters
    ----------
    top_k : int
        Default number of chunks returned per query.
    auto_seed : bool
        If True (default), automatically index all papers in PAPERS_DIR
        when the collection is empty.

    Examples
    --------
    >>> store = ResearchVectorStore()
    >>> results = store.query("How does transformer attention work?")
    >>> for r in results:
    ...     print(r["source"], r["text"][:80])
    """

    def __init__(
        self,
        top_k: int   = DEFAULT_TOP_K,
        auto_seed: bool = True,
    ) -> None:
        self._top_k     = top_k
        self._auto_seed = auto_seed
        self._collection = None   # lazy init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Retrieve the most relevant text chunks for a research question.

        Parameters
        ----------
        question : str
            The user's research question to match against stored chunks.
        top_k : int | None
            Number of chunks to retrieve. Defaults to ``self._top_k``.

        Returns
        -------
        list[dict]
            Each dict contains: ``text``, ``source``, ``chunk_index``,
            ``distance`` (lower = more similar).
            Returns an empty list if the collection is empty.
        """
        self._ensure_ready()
        k = top_k or self._top_k

        count = self._collection.count()
        if count == 0:
            logger.warning("ChromaDB collection is empty — no context retrieved.")
            return []

        # Embed the query.
        embed_model = _get_embed_model()
        query_embedding = embed_model.encode([question], normalize_embeddings=True)

        # Query ChromaDB.
        results = self._collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text":        doc,
                "source":      meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "distance":    dist,
            })

        logger.info(
            "Retrieved %d chunk(s) for: '%.60s...' (top distance=%.4f)",
            len(chunks), question,
            chunks[0]["distance"] if chunks else float("nan"),
        )
        return chunks

    def add_paper(self, text: str, source: str) -> int:
        """
        Add a research paper to the vector store.

        The paper is chunked, embedded, and persisted. Chunks that already
        exist (same source + chunk_index) are skipped to avoid duplicates.

        Parameters
        ----------
        text : str
            Full paper text.
        source : str
            A unique identifier (e.g. filename) for this paper.

        Returns
        -------
        int
            Number of new chunks added.
        """
        self._ensure_ready()
        chunks = _chunk_text(text, source)
        if not chunks:
            logger.warning("No chunks produced for source '%s'.", source)
            return 0

        embed_model = _get_embed_model()
        texts      = [c["text"]  for c in chunks]
        embeddings = embed_model.encode(texts, normalize_embeddings=True).tolist()
        ids        = [f"{source}::chunk_{c['chunk_index']}" for c in chunks]
        metadatas  = [{
            "source":      c["source"],
            "chunk_index": c["chunk_index"],
            "start_char":  c["start_char"],
            "end_char":    c["end_char"],
        } for c in chunks]

        # ChromaDB will raise on duplicate IDs — filter them out first.
        existing_ids = set(self._collection.get(ids=ids)["ids"])
        new_indices  = [i for i, uid in enumerate(ids) if uid not in existing_ids]

        if not new_indices:
            logger.debug("All %d chunks for '%s' already exist — skipping.", len(chunks), source)
            return 0

        self._collection.add(
            ids        = [ids[i]        for i in new_indices],
            documents  = [texts[i]      for i in new_indices],
            embeddings = [embeddings[i] for i in new_indices],
            metadatas  = [metadatas[i]  for i in new_indices],
        )

        logger.info(
            "Added %d/%d new chunks for paper '%s' (total in DB: %d).",
            len(new_indices), len(chunks), source, self._collection.count(),
        )
        return len(new_indices)

    def add_pdf(self, file_bytes: bytes, filename: str) -> int:
        """
        Extract text content from a PDF file byte stream, chunk it,
        embed it, and save the embeddings inside ChromaDB.

        Parameters
        ----------
        file_bytes : bytes
            Raw binary bytes of the PDF file.
        filename : str
            Unique identifier or file name of the source PDF.

        Returns
        -------
        int
            Number of new document chunks added.
        """
        import io  # noqa: PLC0415
        from pypdf import PdfReader  # noqa: PLC0415

        self._ensure_ready()
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            text_parts = []
            for page in reader.pages:
                val = page.extract_text()
                if val:
                    text_parts.append(val)
            
            full_text = "\n".join(text_parts).strip()
            if not full_text:
                logger.warning("No readable text found inside PDF '%s'.", filename)
                return 0
                
            return self.add_paper(full_text, source=filename)
        except Exception as exc:
            logger.error("Failed to parse PDF '%s': %s", filename, exc)
            raise RuntimeError(f"Error parsing PDF '{filename}': {exc}") from exc

    @property
    def document_count(self) -> int:
        """Total number of chunks currently stored."""
        self._ensure_ready()
        return self._collection.count()

    @property
    def paper_sources(self) -> list[str]:
        """Unique source names (paper filenames) currently indexed."""
        self._ensure_ready()
        if self._collection.count() == 0:
            return []
        results = self._collection.get(include=["metadatas"])
        sources = {m.get("source", "") for m in results["metadatas"]}
        return sorted(sources)

    def clear(self) -> None:
        """Delete all documents from the collection (irreversible)."""
        client = _get_chroma_client()
        client.delete_collection(COLLECTION_NAME)
        self._collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ResearchVectorStore collection cleared.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_ready(self) -> None:
        """Initialise the ChromaDB collection and auto-seed if empty."""
        if self._collection is None:
            client = _get_chroma_client()
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB collection '%s' ready (%d chunks).",
                COLLECTION_NAME, self._collection.count(),
            )

        # Auto-seed from data/papers/ if the collection is empty.
        if self._auto_seed and self._collection.count() == 0:
            self._auto_seed = False
            try:
                self._seed_from_papers_dir()
            finally:
                self._auto_seed = True

    def _seed_from_papers_dir(self) -> None:
        """Index all .txt files in PAPERS_DIR into the collection."""
        if not PAPERS_DIR.is_dir():
            logger.warning(
                "Papers directory not found: %s — no papers seeded.", PAPERS_DIR
            )
            return

        txt_files = sorted(PAPERS_DIR.glob("*.txt"))
        if not txt_files:
            logger.warning("No .txt files found in %s.", PAPERS_DIR)
            return

        logger.info("Seeding %d paper(s) from %s …", len(txt_files), PAPERS_DIR)
        total = 0
        for path in txt_files:
            try:
                text   = path.read_text(encoding="utf-8")
                added  = self.add_paper(text, source=path.name)
                total += added
            except Exception as exc:
                logger.error("Failed to seed '%s': %s", path.name, exc)

        logger.info("Seeding complete — %d chunks indexed.", total)
