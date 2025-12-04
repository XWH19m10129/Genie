"""
Text similarity detection module using embeddings and cosine similarity.

This module provides functions for computing text embeddings and measuring
semantic similarity between text pairs. It supports OpenAI embeddings
(text-embedding-3-large) with automatic fallback to local sentence-transformers
model (all-MiniLM-L6-v2).

Example:
    >>> from genie.similarity import is_semantic_similar
    >>> result = is_semantic_similar("Hello world", "Hi there")
    >>> print(result)
    {'score': 0.85, 'similar': True, 'method': 'openai'}
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_FILE = Path("embeddings_cache.json")
_cache: Dict[str, List[float]] = {}
_cache_loaded = False

# Model configuration
OPENAI_MODEL = "text-embedding-3-large"
LOCAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Global state for local model
_local_model = None
_local_model_available: Optional[bool] = None


def _load_cache() -> None:
    """Load embeddings cache from disk."""
    global _cache, _cache_loaded
    if _cache_loaded:
        return
    
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _cache = json.load(f)
            logger.debug("Loaded %d cached embeddings", len(_cache))
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load cache file: %s", e)
            _cache = {}
    
    _cache_loaded = True


def _save_cache() -> None:
    """Save embeddings cache to disk."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_cache, f)
        logger.debug("Saved %d embeddings to cache", len(_cache))
    except IOError as e:
        logger.warning("Failed to save cache file: %s", e)


def clear_cache() -> None:
    """
    Clear the embeddings cache from memory and disk.
    
    This function removes all cached embeddings and deletes the cache file
    if it exists. Useful for freeing memory or forcing fresh embeddings.
    """
    global _cache, _cache_loaded
    _cache = {}
    _cache_loaded = True  # Prevent reloading old cache
    
    if CACHE_FILE.exists():
        try:
            CACHE_FILE.unlink()
            logger.info("Cache file deleted")
        except IOError as e:
            logger.warning("Failed to delete cache file: %s", e)


def _get_cache_key(text: str, method: str) -> str:
    """Generate a cache key for a text and method combination."""
    import hashlib
    text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    return f"{method}:{text_hash}"


def _openai_available() -> bool:
    """Check if OpenAI API is available."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


def _get_openai_embedding(text: str) -> Optional[List[float]]:
    """
    Get embedding using OpenAI API.
    
    Args:
        text: Input text to embed.
        
    Returns:
        List of floats representing the embedding, or None if failed.
    """
    try:
        import openai
        
        client = openai.OpenAI()
        response = client.embeddings.create(
            model=OPENAI_MODEL,
            input=text
        )
        return response.data[0].embedding
    except ImportError:
        logger.debug("OpenAI package not installed")
        return None
    except Exception as e:
        logger.warning("OpenAI API request failed: %s", e)
        return None


def _init_local_model() -> bool:
    """
    Initialize the local sentence-transformers model.
    
    Returns:
        True if model was initialized successfully, False otherwise.
    """
    global _local_model, _local_model_available
    
    if _local_model_available is not None:
        return _local_model_available
    
    try:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(LOCAL_MODEL_NAME)
        _local_model_available = True
        logger.info("Local model '%s' loaded successfully", LOCAL_MODEL_NAME)
        return True
    except ImportError:
        logger.warning("sentence-transformers package not installed")
        _local_model_available = False
        return False
    except Exception as e:
        logger.warning("Failed to load local model: %s", e)
        _local_model_available = False
        return False


def _get_local_embedding(text: str) -> Optional[List[float]]:
    """
    Get embedding using local sentence-transformers model.
    
    Args:
        text: Input text to embed.
        
    Returns:
        List of floats representing the embedding, or None if failed.
    """
    if not _init_local_model():
        return None
    
    try:
        embedding = _local_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        logger.warning("Local embedding generation failed: %s", e)
        return None


def get_embedding(text: str, use_cache: bool = True) -> Tuple[Optional[List[float]], str]:
    """
    Get text embedding using available method.
    
    This function attempts to get an embedding using OpenAI's API first.
    If OpenAI is unavailable (missing API key or package), it falls back
    to a local sentence-transformers model.
    
    Args:
        text: The input text to generate an embedding for.
        use_cache: Whether to use cached embeddings if available. Default True.
        
    Returns:
        A tuple of (embedding, method) where:
        - embedding: List of floats representing the text embedding, or None if failed.
        - method: String indicating which method was used ('openai', 'local', or 'failed').
        
    Example:
        >>> embedding, method = get_embedding("Hello world")
        >>> print(f"Used {method}, got {len(embedding)} dimensions")
        Used openai, got 3072 dimensions
    """
    _load_cache()
    
    # Try OpenAI first
    if _openai_available():
        cache_key = _get_cache_key(text, "openai")
        if use_cache and cache_key in _cache:
            logger.debug("Using cached OpenAI embedding")
            return _cache[cache_key], "openai"
        
        embedding = _get_openai_embedding(text)
        if embedding is not None:
            if use_cache:
                _cache[cache_key] = embedding
                _save_cache()
            return embedding, "openai"
    
    # Fall back to local model
    cache_key = _get_cache_key(text, "local")
    if use_cache and cache_key in _cache:
        logger.debug("Using cached local embedding")
        return _cache[cache_key], "local"
    
    embedding = _get_local_embedding(text)
    if embedding is not None:
        if use_cache:
            _cache[cache_key] = embedding
            _save_cache()
        return embedding, "local"
    
    logger.error("No embedding method available. Install openai or sentence-transformers.")
    return None, "failed"


def cosine_similarity(vec1: Union[List[float], np.ndarray], 
                      vec2: Union[List[float], np.ndarray]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Uses numpy for stable numerical computation. Handles edge cases
    like zero vectors gracefully.
    
    Args:
        vec1: First vector as a list or numpy array.
        vec2: Second vector as a list or numpy array.
        
    Returns:
        Cosine similarity score between -1 and 1.
        Returns 0.0 if either vector has zero magnitude.
        
    Example:
        >>> vec1 = [1.0, 0.0, 0.0]
        >>> vec2 = [0.0, 1.0, 0.0]
        >>> cosine_similarity(vec1, vec2)
        0.0
        >>> cosine_similarity(vec1, vec1)
        1.0
    """
    arr1 = np.asarray(vec1, dtype=np.float64)
    arr2 = np.asarray(vec2, dtype=np.float64)
    
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    
    # Handle zero vectors
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Compute cosine similarity
    dot_product = np.dot(arr1, arr2)
    similarity = dot_product / (norm1 * norm2)
    
    # Clamp to [-1, 1] to handle floating point errors
    return float(np.clip(similarity, -1.0, 1.0))


def is_semantic_similar(text1: str, text2: str, 
                        threshold: float = 0.8,
                        use_cache: bool = True) -> Dict[str, Union[float, bool, str]]:
    """
    Determine if two texts are semantically similar.
    
    This function generates embeddings for both texts and calculates
    their cosine similarity. It returns a detailed result including
    the similarity score, whether the texts are considered similar
    based on the threshold, and which embedding method was used.
    
    Args:
        text1: First text to compare.
        text2: Second text to compare.
        threshold: Similarity threshold (0.0 to 1.0). Texts with similarity
                   >= threshold are considered similar. Default is 0.8.
        use_cache: Whether to use cached embeddings. Default True.
        
    Returns:
        Dictionary containing:
        - score (float): Cosine similarity score between the texts.
        - similar (bool): Whether the texts are considered similar.
        - method (str): The embedding method used ('openai', 'local', or 'failed').
        
    Raises:
        No exceptions are raised. If embedding fails, returns score=0.0,
        similar=False, and method='failed'.
        
    Example:
        >>> result = is_semantic_similar("The cat sat on the mat", 
        ...                               "A cat is sitting on a mat")
        >>> print(result)
        {'score': 0.92, 'similar': True, 'method': 'openai'}
    """
    # Get embeddings for both texts
    embedding1, method1 = get_embedding(text1, use_cache=use_cache)
    embedding2, method2 = get_embedding(text2, use_cache=use_cache)
    
    # Determine the method used (use the first one, they should match)
    method = method1 if method1 != "failed" else method2
    
    # If either embedding failed, return failure result
    if embedding1 is None or embedding2 is None:
        return {
            "score": 0.0,
            "similar": False,
            "method": "failed"
        }
    
    # Calculate similarity
    score = cosine_similarity(embedding1, embedding2)
    
    return {
        "score": score,
        "similar": score >= threshold,
        "method": method
    }


if __name__ == "__main__":
    # Simple demo
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) >= 3:
        text1 = sys.argv[1]
        text2 = sys.argv[2]
    else:
        text1 = "The quick brown fox jumps over the lazy dog."
        text2 = "A fast brown fox leaps over a sleepy dog."
    
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print()
    
    result = is_semantic_similar(text1, text2)
    print(f"Similarity Score: {result['score']:.4f}")
    print(f"Similar: {result['similar']}")
    print(f"Method: {result['method']}")
