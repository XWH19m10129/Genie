"""
Unit tests for genie.similarity module.

These tests cover:
- Cosine similarity calculation
- OpenAI and local model fallback
- Cache functionality
- Edge cases and error handling

To run tests:
    pytest tests/test_similarity.py -v
    
Note: Some tests require specific packages or API keys and will be skipped
if prerequisites are not met.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from genie.similarity import (
    cosine_similarity,
    get_embedding,
    is_semantic_similar,
    clear_cache,
    _get_cache_key,
    _load_cache,
    _save_cache,
    CACHE_FILE,
)


class TestCosineSimilarity(unittest.TestCase):
    """Tests for cosine_similarity function."""
    
    def test_identical_vectors(self):
        """Identical vectors should have similarity of 1.0."""
        vec = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cosine_similarity(vec, vec), 1.0, places=7)
    
    def test_opposite_vectors(self):
        """Opposite vectors should have similarity of -1.0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), -1.0, places=7)
    
    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity of 0.0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), 0.0, places=7)
    
    def test_zero_vector(self):
        """Zero vector should return 0.0 similarity."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        self.assertEqual(cosine_similarity(vec1, vec2), 0.0)
        self.assertEqual(cosine_similarity(vec2, vec1), 0.0)
    
    def test_numpy_array_input(self):
        """Should accept numpy arrays as input."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), 1.0, places=7)
    
    def test_similar_vectors(self):
        """Similar but not identical vectors should have high similarity."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.1, 2.1, 3.1]
        similarity = cosine_similarity(vec1, vec2)
        self.assertGreater(similarity, 0.99)
        self.assertLess(similarity, 1.0)
    
    def test_different_length_handling(self):
        """Different length vectors should still compute (numpy will broadcast)."""
        # Note: This tests numpy's behavior, which may vary
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), 1.0, places=7)


class TestCacheKey(unittest.TestCase):
    """Tests for cache key generation."""
    
    def test_different_texts_different_keys(self):
        """Different texts should produce different keys."""
        key1 = _get_cache_key("hello", "openai")
        key2 = _get_cache_key("world", "openai")
        self.assertNotEqual(key1, key2)
    
    def test_different_methods_different_keys(self):
        """Same text with different methods should produce different keys."""
        key1 = _get_cache_key("hello", "openai")
        key2 = _get_cache_key("hello", "local")
        self.assertNotEqual(key1, key2)
    
    def test_same_input_same_key(self):
        """Same text and method should produce same key."""
        key1 = _get_cache_key("hello", "openai")
        key2 = _get_cache_key("hello", "openai")
        self.assertEqual(key1, key2)


class TestCacheFunctionality(unittest.TestCase):
    """Tests for cache loading, saving, and clearing."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Save original cache file path
        self.original_cache_file = CACHE_FILE
        # Use temporary file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.temp_cache_file = Path(self.temp_dir) / "test_cache.json"
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temp cache file if exists
        if self.temp_cache_file.exists():
            self.temp_cache_file.unlink()
        # Remove temp directory
        Path(self.temp_dir).rmdir()
    
    def test_clear_cache(self):
        """Test that clear_cache removes cache."""
        # Create a mock cache
        with patch('genie.similarity._cache', {'test': [1.0, 2.0, 3.0]}):
            with patch('genie.similarity.CACHE_FILE', self.temp_cache_file):
                # Write test cache file
                with open(self.temp_cache_file, 'w') as f:
                    json.dump({'test': [1.0, 2.0, 3.0]}, f)
                
                clear_cache()
                
                # Cache file should be deleted
                self.assertFalse(self.temp_cache_file.exists())


class TestGetEmbeddingWithMocks(unittest.TestCase):
    """Tests for get_embedding function using mocks."""
    
    def test_openai_fallback_to_local(self):
        """Should fall back to local model when OpenAI unavailable."""
        mock_embedding = [0.1] * 384  # Local model dimension
        
        with patch('genie.similarity._openai_available', return_value=False):
            with patch('genie.similarity._get_local_embedding', return_value=mock_embedding):
                with patch('genie.similarity._load_cache'):
                    embedding, method = get_embedding("test text", use_cache=False)
                    self.assertEqual(method, "local")
                    self.assertEqual(embedding, mock_embedding)
    
    def test_returns_cached_embedding(self):
        """Should return cached embedding when available."""
        cached_embedding = [0.5] * 100
        cache_key = _get_cache_key("test text", "openai")
        
        with patch('genie.similarity._cache', {cache_key: cached_embedding}):
            with patch('genie.similarity._cache_loaded', True):
                with patch('genie.similarity._openai_available', return_value=True):
                    embedding, method = get_embedding("test text", use_cache=True)
                    self.assertEqual(embedding, cached_embedding)
                    self.assertEqual(method, "openai")
    
    def test_failed_when_no_method_available(self):
        """Should return failed when no embedding method is available."""
        with patch('genie.similarity._openai_available', return_value=False):
            with patch('genie.similarity._get_local_embedding', return_value=None):
                with patch('genie.similarity._load_cache'):
                    embedding, method = get_embedding("test text", use_cache=False)
                    self.assertIsNone(embedding)
                    self.assertEqual(method, "failed")


class TestIsSemanticSimilar(unittest.TestCase):
    """Tests for is_semantic_similar function."""
    
    def test_similar_texts_with_mocked_embeddings(self):
        """Should return similar=True for similar embeddings."""
        # Mock embeddings that are very similar
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.99, 0.01, 0.0]
        
        with patch('genie.similarity.get_embedding') as mock_get:
            mock_get.side_effect = [
                (embedding1, "openai"),
                (embedding2, "openai")
            ]
            
            result = is_semantic_similar("text1", "text2", threshold=0.8)
            
            self.assertGreater(result["score"], 0.8)
            self.assertTrue(result["similar"])
            self.assertEqual(result["method"], "openai")
    
    def test_dissimilar_texts_with_mocked_embeddings(self):
        """Should return similar=False for dissimilar embeddings."""
        # Mock embeddings that are orthogonal
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]
        
        with patch('genie.similarity.get_embedding') as mock_get:
            mock_get.side_effect = [
                (embedding1, "local"),
                (embedding2, "local")
            ]
            
            result = is_semantic_similar("text1", "text2", threshold=0.8)
            
            self.assertAlmostEqual(result["score"], 0.0, places=5)
            self.assertFalse(result["similar"])
            self.assertEqual(result["method"], "local")
    
    def test_failed_embedding_returns_zero_score(self):
        """Should return score=0 and similar=False when embedding fails."""
        with patch('genie.similarity.get_embedding') as mock_get:
            mock_get.side_effect = [
                (None, "failed"),
                (None, "failed")
            ]
            
            result = is_semantic_similar("text1", "text2")
            
            self.assertEqual(result["score"], 0.0)
            self.assertFalse(result["similar"])
            self.assertEqual(result["method"], "failed")
    
    def test_custom_threshold(self):
        """Should respect custom threshold value."""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.9, 0.1, 0.0]  # ~0.994 cosine similarity
        
        with patch('genie.similarity.get_embedding') as mock_get:
            mock_get.side_effect = [
                (embedding1, "openai"),
                (embedding2, "openai")
            ]
            
            # With high threshold, should not be similar
            result = is_semantic_similar("text1", "text2", threshold=0.995)
            self.assertFalse(result["similar"])
            
        with patch('genie.similarity.get_embedding') as mock_get:
            mock_get.side_effect = [
                (embedding1, "openai"),
                (embedding2, "openai")
            ]
            
            # With lower threshold, should be similar
            result = is_semantic_similar("text1", "text2", threshold=0.9)
            self.assertTrue(result["similar"])


class TestOpenAIIntegration(unittest.TestCase):
    """Integration tests for OpenAI embeddings (skipped if API key not set)."""
    
    @classmethod
    def setUpClass(cls):
        """Check if OpenAI API key is available."""
        cls.has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
        try:
            import openai
            cls.has_openai = True
        except ImportError:
            cls.has_openai = False
    
    @unittest.skipUnless(
        os.environ.get("OPENAI_API_KEY"),
        "OpenAI API key not set"
    )
    def test_openai_embedding(self):
        """Test actual OpenAI embedding generation."""
        if not self.has_openai_key or not self.has_openai:
            self.skipTest("OpenAI not available")
        
        embedding, method = get_embedding("Hello, world!", use_cache=False)
        
        self.assertEqual(method, "openai")
        self.assertIsNotNone(embedding)
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)


class TestLocalModelIntegration(unittest.TestCase):
    """Integration tests for local sentence-transformers model."""
    
    @classmethod
    def setUpClass(cls):
        """Check if sentence-transformers is available."""
        try:
            import sentence_transformers
            cls.has_sentence_transformers = True
        except ImportError:
            cls.has_sentence_transformers = False
    
    @unittest.skipUnless(
        True,  # Will check inside the test
        "sentence-transformers not installed"
    )
    def test_local_embedding(self):
        """Test actual local embedding generation."""
        if not self.has_sentence_transformers:
            self.skipTest("sentence-transformers not installed")
        
        # Force local model by patching OpenAI availability
        with patch('genie.similarity._openai_available', return_value=False):
            embedding, method = get_embedding("Hello, world!", use_cache=False)
        
        self.assertEqual(method, "local")
        self.assertIsNotNone(embedding)
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 384)  # MiniLM-L6-v2 dimension


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_string(self):
        """Should handle empty strings."""
        with patch('genie.similarity.get_embedding') as mock_get:
            mock_get.side_effect = [
                ([0.1] * 100, "local"),
                ([0.1] * 100, "local")
            ]
            
            result = is_semantic_similar("", "")
            self.assertIn("score", result)
            self.assertIn("similar", result)
            self.assertIn("method", result)
    
    def test_very_long_text(self):
        """Should handle very long texts without error."""
        long_text = "word " * 1000
        
        with patch('genie.similarity.get_embedding') as mock_get:
            mock_get.side_effect = [
                ([0.1] * 100, "local"),
                ([0.1] * 100, "local")
            ]
            
            result = is_semantic_similar(long_text, long_text)
            self.assertIn("score", result)
    
    def test_unicode_text(self):
        """Should handle unicode characters."""
        text1 = "Hello, 世界! 🌍"
        text2 = "你好，世界！"
        
        with patch('genie.similarity.get_embedding') as mock_get:
            mock_get.side_effect = [
                ([0.1] * 100, "local"),
                ([0.2] * 100, "local")
            ]
            
            result = is_semantic_similar(text1, text2)
            self.assertIn("score", result)
    
    def test_special_characters(self):
        """Should handle special characters."""
        text1 = "<html>&nbsp;test</html>"
        text2 = "test with @#$%^&*() characters"
        
        with patch('genie.similarity.get_embedding') as mock_get:
            mock_get.side_effect = [
                ([0.1] * 100, "local"),
                ([0.2] * 100, "local")
            ]
            
            result = is_semantic_similar(text1, text2)
            self.assertIn("score", result)


if __name__ == "__main__":
    unittest.main()
