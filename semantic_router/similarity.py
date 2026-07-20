"""
semantic_router/similarity.py
-----------------------------
Helper function for calculating cosine similarity between two numpy vectors.
"""

import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors.

    Parameters
    ----------
    a : np.ndarray
        First vector.
    b : np.ndarray
        Second vector.

    Returns
    -------
    float
        Cosine similarity score in range [-1.0, 1.0].
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return float(dot_product / (norm_a * norm_b))
