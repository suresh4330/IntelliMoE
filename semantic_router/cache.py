"""
semantic_router/cache.py
------------------------
Cache manager for storing embedded expert profiles and query representations
to avoid redundant SentenceTransformer inference calls.
"""

from typing import Dict, List

class EmbeddingCache:
    """In-memory cache for embeddings."""
    
    def __init__(self) -> None:
        # Maps string identifier to its vector list representation
        self._cache: Dict[str, List[float]] = {}

    def get(self, key: str) -> List[float]:
        """Retrieve embedding from cache or return None."""
        return self._cache.get(key)

    def set(self, key: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        self._cache[key] = embedding

    def clear(self) -> None:
        """Clear cache contents."""
        self._cache.clear()
