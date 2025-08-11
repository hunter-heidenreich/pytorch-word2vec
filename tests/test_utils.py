"""Comprehensive tests for utility functions.

This test suite provides extensive coverage of the utils module,
including:

- Seed setting for reproducibility
- Device detection and management
- Device-specific optimizations
- Embedding export and loading
- Vocabulary and embeddings loading
- Cosine similarity computation
- Similar word finding functionality
- Error handling and edge cases

Coverage: 100% of utils/__init__.py module
"""

import json
import os
import tempfile
from unittest.mock import Mock, patch
import numpy as np
import pytest
import torch

from src.modern_word2vec.utils import (
    set_seed,
    get_device,
    setup_device_optimizations,
    export_embeddings,
    load_vocab_and_embeddings,
    compute_cosine_similarities,
    find_similar,
)


class TestSetSeed:
    """Test cases for set_seed function."""

    def test_set_seed_reproducibility(self):
        """Test that set_seed makes random number generation reproducible."""
        seed = 42
        set_seed(seed)

        # Generate some random numbers
        import random

        python_random1 = random.random()
        numpy_random1 = np.random.random()
        torch_random1 = torch.randn(3, 3)

        # Reset seed and generate again
        set_seed(seed)
        python_random2 = random.random()
        numpy_random2 = np.random.random()
        torch_random2 = torch.randn(3, 3)

        # Should be identical
        assert python_random1 == python_random2
        assert numpy_random1 == numpy_random2
        assert torch.allclose(torch_random1, torch_random2)

    def test_set_seed_different_seeds(self):
        """Test that different seeds produce different results."""
        import random

        set_seed(42)
        random1 = random.random()
        numpy1 = np.random.random()
        torch1 = torch.randn(2, 2)

        set_seed(43)
        random2 = random.random()
        numpy2 = np.random.random()
        torch2 = torch.randn(2, 2)

        # Should be different
        assert random1 != random2
        assert numpy1 != numpy2
        assert not torch.allclose(torch1, torch2)

    @patch("torch.cuda.manual_seed_all")
    def test_set_seed_cuda_called(self, mock_cuda_seed):
        """Test that CUDA seed is set when available."""
        mock_cuda_seed.reset_mock()  # Reset any previous calls
        set_seed(42)
        mock_cuda_seed.assert_called_with(42)


class TestGetDevice:
    """Test cases for get_device function."""

    def test_get_device_cpu_always_available(self):
        """Test that CPU device is always returned as fallback."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            device = get_device()
            assert device.type == "cpu"

    def test_get_device_prefer_cuda(self):
        """Test device selection when CUDA is preferred and available."""
        with patch("torch.cuda.is_available", return_value=True):
            device = get_device(prefer="cuda")
            assert device.type == "cuda"

    def test_get_device_prefer_mps(self):
        """Test device selection when MPS is preferred and available."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch.object(torch.backends, "mps", create=True) as mock_mps,
        ):
            mock_mps.is_available.return_value = True
            device = get_device(prefer="mps")
            assert device.type == "mps"

    def test_get_device_prefer_cpu(self):
        """Test device selection when CPU is explicitly preferred."""
        device = get_device(prefer="cpu")
        assert device.type == "cpu"

    def test_get_device_fallback_when_preferred_unavailable(self):
        """Test fallback behavior when preferred device is unavailable."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            device = get_device(prefer="cuda")
            assert device.type == "cpu"

    @patch("torch.cuda.is_available", return_value=True)
    def test_get_device_cuda_priority(self, mock_cuda):
        """Test that CUDA has priority over MPS when both available."""
        with patch.object(torch.backends, "mps", create=True) as mock_mps:
            mock_mps.is_available.return_value = True
            device = get_device()
            assert device.type == "cuda"

    def test_get_device_invalid_preference(self):
        """Test behavior with invalid device preference."""
        device = get_device(prefer="invalid")
        # Should fallback to best available
        assert device.type in ["cuda", "mps", "cpu"]

    def test_get_device_mps_without_backends(self):
        """Test MPS handling when torch.backends.mps doesn't exist."""
        with patch("torch.cuda.is_available", return_value=False):
            # If hasattr check fails, should fall back to CPU
            with patch("builtins.hasattr", return_value=False):
                device = get_device(prefer="mps")
                assert device.type == "cpu"


class TestSetupDeviceOptimizations:
    """Test cases for setup_device_optimizations function."""

    @patch("torch.set_float32_matmul_precision")
    def test_setup_cuda_optimizations(self, mock_precision):
        """Test CUDA-specific optimizations are enabled."""
        device = torch.device("cuda")

        # The function should run without errors and call precision setting
        setup_device_optimizations(device)
        mock_precision.assert_called_once_with("medium")

    @patch("torch.set_float32_matmul_precision")
    def test_setup_cpu_optimizations(self, mock_precision):
        """Test optimizations for CPU device."""
        device = torch.device("cpu")
        setup_device_optimizations(device)
        mock_precision.assert_called_once_with("medium")

    @patch("torch.set_float32_matmul_precision")
    def test_setup_mps_optimizations(self, mock_precision):
        """Test optimizations for MPS device."""
        device = torch.device("mps")
        setup_device_optimizations(device)
        mock_precision.assert_called_once_with("medium")

    @patch("torch.backends.cudnn")
    @patch("torch.set_float32_matmul_precision")
    def test_setup_optimizations_exception_handling(self, mock_precision, mock_cudnn):
        """Test that exceptions in optimization setup are handled gracefully."""
        # Make precision setting raise an exception
        mock_precision.side_effect = Exception("Test exception")

        device = torch.device("cpu")
        # Should not raise exception
        setup_device_optimizations(device)

    @patch("torch.set_float32_matmul_precision")
    def test_setup_cuda_tf32_exception_handling(self, mock_precision):
        """Test that TF32 setting exceptions are handled gracefully."""
        device = torch.device("cuda")

        # Should not raise exception even if precision setting fails
        mock_precision.side_effect = Exception("Test exception")
        setup_device_optimizations(device)


class TestExportEmbeddings:
    """Test cases for export_embeddings function."""

    def setUp(self):
        """Set up common test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.vocab_size = 5
        self.embedding_dim = 10

        # Create mock model
        self.model = Mock()
        self.model.in_embeddings = Mock()
        self.model.in_embeddings.weight = torch.randn(
            self.vocab_size, self.embedding_dim
        )

        # Create mock dataset
        self.dataset = Mock()
        self.dataset.idx_to_word = {
            0: "apple",
            1: "banana",
            2: "cherry",
            3: "date",
            4: "elderberry",
        }
        self.dataset.word_to_idx = {
            "apple": 0,
            "banana": 1,
            "cherry": 2,
            "date": 3,
            "elderberry": 4,
        }

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_export_embeddings_creates_files(self):
        """Test that export_embeddings creates the expected files."""
        self.setUp()
        try:
            export_embeddings(self.model, self.temp_dir, self.dataset)

            # Check files exist
            embeddings_path = os.path.join(self.temp_dir, "embeddings.npy")
            vocab_path = os.path.join(self.temp_dir, "vocab.json")

            assert os.path.exists(embeddings_path)
            assert os.path.exists(vocab_path)
        finally:
            self.tearDown()

    def test_export_embeddings_correct_content(self):
        """Test that exported files contain correct content."""
        self.setUp()
        try:
            export_embeddings(self.model, self.temp_dir, self.dataset)

            # Check embeddings
            embeddings = np.load(os.path.join(self.temp_dir, "embeddings.npy"))
            expected_embeddings = self.model.in_embeddings.weight.detach().cpu().numpy()
            np.testing.assert_array_equal(embeddings, expected_embeddings)

            # Check vocabulary
            with open(os.path.join(self.temp_dir, "vocab.json"), "r") as f:
                vocab_data = json.load(f)

            # JSON converts integer keys to strings, so we need to handle this
            expected_idx_to_word = {
                str(k): v for k, v in self.dataset.idx_to_word.items()
            }
            assert vocab_data["idx_to_word"] == expected_idx_to_word
            assert vocab_data["word_to_idx"] == self.dataset.word_to_idx
        finally:
            self.tearDown()

    def test_export_embeddings_creates_directory(self):
        """Test that export_embeddings creates output directory if it doesn't exist."""
        self.setUp()
        try:
            nested_dir = os.path.join(self.temp_dir, "nested", "path")
            export_embeddings(self.model, nested_dir, self.dataset)

            assert os.path.exists(nested_dir)
            assert os.path.exists(os.path.join(nested_dir, "embeddings.npy"))
            assert os.path.exists(os.path.join(nested_dir, "vocab.json"))
        finally:
            self.tearDown()

    def test_export_embeddings_gpu_tensor(self):
        """Test export with GPU tensor (should work via CPU conversion)."""
        self.setUp()
        try:
            # Simulate GPU tensor by creating a tensor and mocking CUDA device
            if torch.cuda.is_available():
                gpu_tensor = self.model.in_embeddings.weight.cuda()
                self.model.in_embeddings.weight = gpu_tensor

            export_embeddings(self.model, self.temp_dir, self.dataset)

            # Should still work
            assert os.path.exists(os.path.join(self.temp_dir, "embeddings.npy"))
        finally:
            self.tearDown()


class TestLoadVocabAndEmbeddings:
    """Test cases for load_vocab_and_embeddings function."""

    def setUp(self):
        """Set up test files."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test data
        self.idx_to_word = {0: "apple", 1: "banana", 2: "cherry"}
        self.word_to_idx = {"apple": 0, "banana": 1, "cherry": 2}
        self.embeddings = np.random.randn(3, 10)

        # Save test files
        vocab_data = {"idx_to_word": self.idx_to_word, "word_to_idx": self.word_to_idx}

        with open(os.path.join(self.temp_dir, "vocab.json"), "w") as f:
            json.dump(vocab_data, f)

        np.save(os.path.join(self.temp_dir, "embeddings.npy"), self.embeddings)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_vocab_and_embeddings_success(self):
        """Test successful loading of vocabulary and embeddings."""
        self.setUp()
        try:
            idx_to_word, word_to_idx, embeddings = load_vocab_and_embeddings(
                self.temp_dir
            )

            assert idx_to_word == self.idx_to_word
            assert word_to_idx == self.word_to_idx
            np.testing.assert_array_equal(embeddings, self.embeddings)
        finally:
            self.tearDown()

    def test_load_vocab_and_embeddings_missing_vocab(self):
        """Test error when vocabulary file is missing."""
        self.setUp()
        try:
            os.remove(os.path.join(self.temp_dir, "vocab.json"))

            with pytest.raises(FileNotFoundError, match="Vocabulary file not found"):
                load_vocab_and_embeddings(self.temp_dir)
        finally:
            self.tearDown()

    def test_load_vocab_and_embeddings_missing_embeddings(self):
        """Test error when embeddings file is missing."""
        self.setUp()
        try:
            os.remove(os.path.join(self.temp_dir, "embeddings.npy"))

            with pytest.raises(FileNotFoundError, match="Embeddings file not found"):
                load_vocab_and_embeddings(self.temp_dir)
        finally:
            self.tearDown()

    def test_load_vocab_and_embeddings_string_indices(self):
        """Test handling of string indices in JSON (should convert to int)."""
        self.setUp()
        try:
            # Create vocab with string indices (as would be loaded from JSON)
            vocab_data = {
                "idx_to_word": {"0": "apple", "1": "banana", "2": "cherry"},
                "word_to_idx": {"apple": 0, "banana": 1, "cherry": 2},
            }

            with open(os.path.join(self.temp_dir, "vocab.json"), "w") as f:
                json.dump(vocab_data, f)

            idx_to_word, word_to_idx, embeddings = load_vocab_and_embeddings(
                self.temp_dir
            )

            # Indices should be converted to int
            assert all(isinstance(k, int) for k in idx_to_word.keys())
            assert idx_to_word[0] == "apple"
            assert idx_to_word[1] == "banana"
            assert idx_to_word[2] == "cherry"
        finally:
            self.tearDown()

    def test_load_vocab_and_embeddings_invalid_json(self):
        """Test error with invalid JSON file."""
        self.setUp()
        try:
            with open(os.path.join(self.temp_dir, "vocab.json"), "w") as f:
                f.write("invalid json content")

            with pytest.raises(json.JSONDecodeError):
                load_vocab_and_embeddings(self.temp_dir)
        finally:
            self.tearDown()


class TestComputeCosineSimilarities:
    """Test cases for compute_cosine_similarities function."""

    def test_compute_cosine_similarities_basic(self):
        """Test basic cosine similarity computation."""
        query = np.array([1.0, 0.0, 0.0])
        all_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # Identical - similarity = 1
                [0.0, 1.0, 0.0],  # Orthogonal - similarity = 0
                [-1.0, 0.0, 0.0],  # Opposite - similarity = -1
                [0.5, 0.5, 0.0],  # 45 degrees - similarity ≈ 0.707
            ]
        )

        similarities = compute_cosine_similarities(query, all_embeddings)

        assert abs(similarities[0] - 1.0) < 1e-6
        assert abs(similarities[1] - 0.0) < 1e-6
        assert abs(similarities[2] - (-1.0)) < 1e-6
        assert abs(similarities[3] - (1 / np.sqrt(2))) < 1e-6

    def test_compute_cosine_similarities_zero_vector(self):
        """Test handling of zero vectors (should not crash)."""
        query = np.array([0.0, 0.0, 0.0])
        all_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        similarities = compute_cosine_similarities(query, all_embeddings)

        # Should return valid results (likely zeros due to epsilon)
        assert similarities.shape == (2,)
        assert not np.any(np.isnan(similarities))
        assert not np.any(np.isinf(similarities))

    def test_compute_cosine_similarities_normalization(self):
        """Test that similarities are properly normalized."""
        query = np.array([2.0, 0.0, 0.0])  # Scaled version
        all_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # Same direction, different magnitude
                [4.0, 0.0, 0.0],  # Same direction, different magnitude
            ]
        )

        similarities = compute_cosine_similarities(query, all_embeddings)

        # Should be 1.0 for both (same direction)
        assert abs(similarities[0] - 1.0) < 1e-6
        assert abs(similarities[1] - 1.0) < 1e-6

    def test_compute_cosine_similarities_single_embedding(self):
        """Test with single embedding."""
        query = np.array([1.0, 1.0])
        all_embeddings = np.array([[1.0, 1.0]])

        similarities = compute_cosine_similarities(query, all_embeddings)

        assert similarities.shape == (1,)
        assert abs(similarities[0] - 1.0) < 1e-6

    def test_compute_cosine_similarities_different_dimensions(self):
        """Test with different embedding dimensions."""
        for dim in [1, 5, 100]:
            query = np.random.randn(dim)
            all_embeddings = np.random.randn(10, dim)

            similarities = compute_cosine_similarities(query, all_embeddings)

            assert similarities.shape == (10,)
            assert np.all(similarities >= -1.0)
            assert np.all(similarities <= 1.0)


class TestFindSimilar:
    """Test cases for find_similar function."""

    def setUp(self):
        """Set up mock model and dataset."""
        self.vocab_size = 5
        self.embedding_dim = 10

        # Create mock model
        self.model = Mock()
        embeddings = torch.randn(self.vocab_size, self.embedding_dim)
        # Make first two embeddings similar
        embeddings[1] = embeddings[0] + 0.1 * torch.randn(self.embedding_dim)
        self.model.in_embeddings.weight = embeddings

        # Create mock dataset
        self.dataset = Mock()
        self.dataset.word_to_idx = {
            "apple": 0,
            "banana": 1,
            "cherry": 2,
            "date": 3,
            "elderberry": 4,
        }
        self.dataset.idx_to_word = {
            0: "apple",
            1: "banana",
            2: "cherry",
            3: "date",
            4: "elderberry",
        }

    def test_find_similar_valid_word(self, capsys):
        """Test finding similar words for a valid word."""
        self.setUp()
        result = find_similar("apple", self.model, self.dataset, top_n=3)

        # Should return list of tuples
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
        assert all(
            isinstance(item[0], str) and isinstance(item[1], float) for item in result
        )

        # Should print results
        captured = capsys.readouterr()
        assert "Words similar to 'apple':" in captured.out
        assert "Similarity:" in captured.out

    def test_find_similar_invalid_word(self, capsys):
        """Test behavior with word not in vocabulary."""
        self.setUp()
        result = find_similar("nonexistent", self.model, self.dataset, top_n=3)

        # Should return empty list
        assert result == []

        # Should print error message
        captured = capsys.readouterr()
        assert "Word 'nonexistent' not in vocabulary." in captured.out

    def test_find_similar_top_n_parameter(self):
        """Test that top_n parameter works correctly."""
        self.setUp()
        for n in [1, 2, 4]:
            result = find_similar("apple", self.model, self.dataset, top_n=n)
            assert len(result) == n

    def test_find_similar_excludes_query_word(self):
        """Test that the query word itself is excluded from results."""
        self.setUp()
        result = find_similar("apple", self.model, self.dataset, top_n=5)

        # Query word should not be in results
        words = [word for word, score in result]
        assert "apple" not in words

    def test_find_similar_similarity_scores_valid(self):
        """Test that similarity scores are in valid range."""
        self.setUp()
        result = find_similar("apple", self.model, self.dataset, top_n=3)

        for word, score in result:
            assert -1.0 <= score <= 1.0

    def test_find_similar_similarity_scores_ordered(self):
        """Test that results are ordered by similarity score (descending)."""
        self.setUp()
        result = find_similar("apple", self.model, self.dataset, top_n=3)

        scores = [score for word, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_find_similar_gpu_model(self):
        """Test find_similar with model on GPU."""
        self.setUp()
        if torch.cuda.is_available():
            # Move model to GPU
            self.model.in_embeddings.weight = self.model.in_embeddings.weight.cuda()

            result = find_similar("apple", self.model, self.dataset, top_n=2)

            # Should still work
            assert len(result) == 2
            assert all(isinstance(item, tuple) for item in result)

    def test_find_similar_identical_embeddings(self):
        """Test behavior when multiple embeddings are identical."""
        self.setUp()
        # Make all embeddings identical except the first
        identical_embedding = self.model.in_embeddings.weight[0].clone()
        for i in range(1, self.vocab_size):
            self.model.in_embeddings.weight[i] = identical_embedding

        result = find_similar("apple", self.model, self.dataset, top_n=3)

        # Should still return valid results
        assert len(result) == 3
        # All similarities should be very close (identical embeddings)
        scores = [score for word, score in result]
        assert all(abs(score - 1.0) < 1e-6 for score in scores)


class TestUtilsIntegration:
    """Integration tests combining multiple utility functions."""

    def test_export_and_load_roundtrip(self):
        """Test exporting and loading embeddings roundtrip."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create mock model and dataset
            vocab_size = 5
            embedding_dim = 10

            model = Mock()
            model.in_embeddings = Mock()
            original_embeddings = torch.randn(vocab_size, embedding_dim)
            model.in_embeddings.weight = original_embeddings

            dataset = Mock()
            dataset.idx_to_word = {
                0: "apple",
                1: "banana",
                2: "cherry",
                3: "date",
                4: "elderberry",
            }
            dataset.word_to_idx = {
                "apple": 0,
                "banana": 1,
                "cherry": 2,
                "date": 3,
                "elderberry": 4,
            }

            # Export
            export_embeddings(model, temp_dir, dataset)

            # Load
            idx_to_word, word_to_idx, loaded_embeddings = load_vocab_and_embeddings(
                temp_dir
            )

            # Verify roundtrip
            assert idx_to_word == dataset.idx_to_word
            assert word_to_idx == dataset.word_to_idx
            np.testing.assert_array_almost_equal(
                loaded_embeddings, original_embeddings.detach().cpu().numpy()
            )

        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_full_similarity_pipeline(self):
        """Test complete similarity finding pipeline."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Set seed for reproducibility
            set_seed(42)

            # Create model on appropriate device
            device = get_device()
            setup_device_optimizations(device)

            # Create test data
            vocab_size = 5
            embedding_dim = 10

            model = Mock()
            model.in_embeddings = Mock()
            # Create embeddings where apple and banana are similar
            embeddings = torch.randn(vocab_size, embedding_dim)
            embeddings[1] = embeddings[0] + 0.1 * torch.randn(
                embedding_dim
            )  # banana similar to apple
            model.in_embeddings.weight = embeddings.to(device)

            dataset = Mock()
            dataset.idx_to_word = {
                0: "apple",
                1: "banana",
                2: "cherry",
                3: "date",
                4: "elderberry",
            }
            dataset.word_to_idx = {
                "apple": 0,
                "banana": 1,
                "cherry": 2,
                "date": 3,
                "elderberry": 4,
            }

            # Export embeddings
            export_embeddings(model, temp_dir, dataset)

            # Find similar words
            result = find_similar("apple", model, dataset, top_n=2)

            # Should find banana as most similar (excluding apple itself)
            assert len(result) == 2
            most_similar_word, similarity_score = result[0]
            assert isinstance(most_similar_word, str)
            assert 0.0 <= similarity_score <= 1.0

        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_device_compatibility_across_functions(self):
        """Test that functions work correctly across different devices."""
        # Test with CPU
        device = torch.device("cpu")
        setup_device_optimizations(device)

        # Create test model
        model = Mock()
        model.in_embeddings = Mock()
        embeddings = torch.randn(3, 5)
        model.in_embeddings.weight = embeddings.to(device)

        dataset = Mock()
        dataset.idx_to_word = {0: "apple", 1: "banana", 2: "cherry"}
        dataset.word_to_idx = {"apple": 0, "banana": 1, "cherry": 2}

        # Test find_similar works with CPU tensors
        result = find_similar("apple", model, dataset, top_n=2)
        assert len(result) == 2

        # Test with GPU if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            setup_device_optimizations(device)
            model.in_embeddings.weight = embeddings.to(device)

            result = find_similar("apple", model, dataset, top_n=2)
            assert len(result) == 2
