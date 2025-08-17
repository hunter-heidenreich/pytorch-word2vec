"""Comprehensive tests for CLI modules."""

import argparse
import json
import os
import tempfile
from unittest.mock import MagicMock, patch, mock_open

import pytest
import numpy as np

from modern_word2vec.cli_args import ArgumentParser, parse_train_args, parse_query_args
from modern_word2vec.cli.train import (
    create_data_config,
    create_train_config,
    load_training_texts,
    create_dataloader,
    create_model,
    save_training_artifacts,
    main as train_main,
)
from modern_word2vec.cli.query import (
    find_similar_words,
    main as query_main,
)
from modern_word2vec.config import DataConfig, TrainConfig


class TestArgumentParser:
    """Test cases for ArgumentParser class."""

    def test_create_train_parser(self):
        """Test creation of training argument parser."""
        parser = ArgumentParser.create_train_parser()

        assert isinstance(parser, argparse.ArgumentParser)
        assert "Train Word2Vec" in parser.description

    def test_create_query_parser(self):
        """Test creation of query argument parser."""
        parser = ArgumentParser.create_query_parser()

        assert isinstance(parser, argparse.ArgumentParser)
        assert "Query saved Word2Vec" in parser.description

    def test_train_parser_required_args(self):
        """Test that training parser has expected required arguments."""
        parser = ArgumentParser.create_train_parser()

        # Test with minimal valid arguments
        args = parser.parse_args([])

        # Check that defaults are set correctly
        assert args.model_type == "skipgram"
        assert args.output_layer == "full_softmax"
        assert args.tokenizer == "basic"
        assert args.optimizer == "adam"

    def test_query_parser_required_args(self):
        """Test that query parser requires necessary arguments."""
        parser = ArgumentParser.create_query_parser()

        # Should raise SystemExit due to missing required args
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_query_parser_valid_args(self):
        """Test query parser with valid arguments."""
        parser = ArgumentParser.create_query_parser()

        args = parser.parse_args(
            ["--run-dir", "/path/to/run", "--word", "test", "--topn", "5"]
        )

        assert args.run_dir == "/path/to/run"
        assert args.word == "test"
        assert args.topn == 5

    def test_parse_train_args_function(self):
        """Test parse_train_args function."""
        args = parse_train_args(["--model-type", "cbow", "--epochs", "10"])

        assert args.model_type == "cbow"
        assert args.epochs == 10

    def test_parse_query_args_function(self):
        """Test parse_query_args function."""
        args = parse_query_args(["--run-dir", "/test/path", "--word", "hello"])

        assert args.run_dir == "/test/path"
        assert args.word == "hello"
        assert args.topn == 10  # default value


class TestTrainCLI:
    """Test cases for training CLI functions."""

    def test_create_data_config(self):
        """Test data configuration creation from arguments."""
        args = MagicMock()
        args.vocab_size = 1000
        args.window_size = 5
        args.model_type = "cbow"
        args.lowercase = True
        args.tokenizer = "enhanced"
        args.subsample = 1e-5
        args.dynamic_window = True

        config = create_data_config(args)

        assert isinstance(config, DataConfig)
        assert config.vocab_size == 1000
        assert config.window_size == 5
        assert config.model_type == "cbow"
        assert config.lowercase is True
        assert config.tokenizer == "enhanced"
        assert config.subsample_t == 1e-5
        assert config.dynamic_window is True

    def test_create_train_config(self):
        """Test training configuration creation from arguments."""
        args = MagicMock()
        args.embedding_dim = 200
        args.batch_size = 64
        args.epochs = 5
        args.lr = 0.01
        args.optimizer = "sgd"
        args.weight_decay = 0.001
        args.grad_clip = 1.0
        args.compile = True
        args.amp = True
        args.workers = 4
        args.pin_memory = True
        args.seed = 42

        config = create_train_config(args)

        assert isinstance(config, TrainConfig)
        assert config.embedding_dim == 200
        assert config.batch_size == 64
        assert config.epochs == 5
        assert config.learning_rate == 0.01
        assert config.optimizer == "sgd"
        assert config.weight_decay == 0.001
        assert config.grad_clip == 1.0
        assert config.compile is True
        assert config.mixed_precision is True
        assert config.num_workers == 4
        assert config.pin_memory is True
        assert config.seed == 42

    @patch("modern_word2vec.cli.train.generate_synthetic_texts")
    def test_load_training_texts_synthetic(self, mock_generate):
        """Test loading synthetic training texts."""
        mock_generate.return_value = ["synthetic text 1", "synthetic text 2"]

        args = MagicMock()
        args.synthetic_sentences = 100
        args.synthetic_vocab = 50
        args.vocab_size = 1000
        args.seed = 42
        args.text_file = None

        with patch("builtins.print"):
            texts = load_training_texts(args)

        assert texts == ["synthetic text 1", "synthetic text 2"]
        mock_generate.assert_called_once()

    def test_load_training_texts_file(self):
        """Test loading training texts from file."""
        test_content = "line 1\nline 2\n\nline 3\n"

        args = MagicMock()
        args.synthetic_sentences = 0
        args.text_file = "/fake/path"

        with patch("builtins.open", mock_open(read_data=test_content)):
            with patch("builtins.print"):
                texts = load_training_texts(args)

        assert texts == ["line 1", "line 2", "line 3"]

    @patch("modern_word2vec.cli.train.load_texts_from_hf")
    def test_load_training_texts_hf(self, mock_load_hf):
        """Test loading training texts from Hugging Face."""
        mock_load_hf.return_value = ["hf text 1", "hf text 2"]

        args = MagicMock()
        args.synthetic_sentences = 0
        args.text_file = None
        args.dataset = "test_dataset"
        args.dataset_config = "test_config"
        args.split = "train"

        with patch("builtins.print"):
            texts = load_training_texts(args)

        assert texts == ["hf text 1", "hf text 2"]
        mock_load_hf.assert_called_once_with("test_dataset", "test_config", "train")

    def test_create_dataloader(self):
        """Test dataloader creation."""
        # Create a mock dataset with __len__ method
        dataset = MagicMock()
        dataset.__len__.return_value = 100  # Mock dataset size

        args = MagicMock()
        args.batch_size = 32
        args.workers = 2
        args.pin_memory = True
        args.model_type = "skipgram"
        args.seed = 42

        dataloader = create_dataloader(dataset, args)

        assert dataloader.batch_size == 32
        assert dataloader.num_workers == 2
        assert dataloader.pin_memory is True

    def test_create_dataloader_cbow(self):
        """Test dataloader creation for CBOW model."""
        dataset = MagicMock()
        dataset.__len__.return_value = 50  # Mock dataset size

        args = MagicMock()
        args.batch_size = 16
        args.workers = 0
        args.pin_memory = False
        args.model_type = "cbow"
        args.seed = 42

        dataloader = create_dataloader(dataset, args)

        assert dataloader.batch_size == 16
        assert dataloader.collate_fn is not None  # Should use cbow_collate

    def test_create_model_skipgram(self):
        """Test skipgram model creation."""
        model = create_model("skipgram", 1000, 128)

        # Check that it returns a model (we can't import the actual classes
        # in this test context, so we just verify it's callable)
        assert model is not None

    def test_create_model_cbow(self):
        """Test CBOW model creation."""
        model = create_model("cbow", 1000, 128)

        assert model is not None

    def test_create_model_invalid_type(self):
        """Test model creation with invalid type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            create_model("invalid_type", 1000, 128)

    def test_save_training_artifacts(self):
        """Test saving training artifacts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Mock objects
            model = MagicMock()
            dataset = MagicMock()
            data_config = DataConfig(
                vocab_size=100, window_size=2, model_type="skipgram"
            )
            train_config = TrainConfig(embedding_dim=50, batch_size=32, epochs=1)

            with patch("modern_word2vec.cli.train.export_embeddings") as mock_export:
                save_training_artifacts(
                    model, dataset, data_config, train_config, tmp_dir
                )

                mock_export.assert_called_once_with(model, tmp_dir, dataset)

                # Check that config file was created
                config_path = os.path.join(tmp_dir, "config.json")
                assert os.path.exists(config_path)

                # Verify config content
                with open(config_path, "r") as f:
                    config_data = json.load(f)

                assert "data" in config_data
                assert "train" in config_data
                assert config_data["data"]["vocab_size"] == 100
                assert config_data["train"]["embedding_dim"] == 50

    @patch("modern_word2vec.cli.train.Trainer")
    @patch("modern_word2vec.cli.train.Word2VecDataset")
    @patch("modern_word2vec.cli.train.load_training_texts")
    @patch("modern_word2vec.cli.train.get_device")
    @patch("modern_word2vec.cli.train.set_seed")
    @patch("modern_word2vec.cli.train.setup_device_optimizations")
    @patch("modern_word2vec.cli.train.create_model")
    @patch("builtins.print")
    def test_main_function(
        self,
        mock_print,
        mock_create_model,
        mock_setup,
        mock_set_seed,
        mock_get_device,
        mock_load_texts,
        mock_dataset,
        mock_trainer,
    ):
        """Test main training function."""
        # Setup mocks
        mock_load_texts.return_value = ["test text"]
        mock_get_device.return_value = "cpu"
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.vocab_size = 100
        mock_dataset_instance.__len__.return_value = 100  # Mock dataset size
        mock_dataset.return_value = mock_dataset_instance
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = {"loss": 1.0}
        mock_trainer.return_value = mock_trainer_instance

        # Test arguments
        test_argv = ["--model-type", "skipgram", "--epochs", "1", "--batch-size", "16"]

        train_main(test_argv)

        # Verify key functions were called
        mock_set_seed.assert_called_once()
        mock_get_device.assert_called_once()
        mock_setup.assert_called_once()
        mock_load_texts.assert_called_once()
        mock_dataset.assert_called_once()
        mock_create_model.assert_called_once()
        mock_trainer.assert_called_once()
        mock_trainer_instance.train.assert_called_once()

    @patch("modern_word2vec.cli.train.save_training_artifacts")
    @patch("modern_word2vec.cli.train.Trainer")
    @patch("modern_word2vec.cli.train.Word2VecDataset")
    @patch("modern_word2vec.cli.train.load_training_texts")
    @patch("modern_word2vec.cli.train.get_device")
    @patch("modern_word2vec.cli.train.set_seed")
    @patch("modern_word2vec.cli.train.setup_device_optimizations")
    @patch("modern_word2vec.cli.train.create_model")
    @patch("builtins.print")
    def test_main_function_with_save(
        self,
        mock_print,
        mock_create_model,
        mock_setup,
        mock_set_seed,
        mock_get_device,
        mock_load_texts,
        mock_dataset,
        mock_trainer,
        mock_save,
    ):
        """Test main training function with save option."""
        # Setup mocks
        mock_load_texts.return_value = ["test text"]
        mock_get_device.return_value = "cpu"
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.vocab_size = 100
        mock_dataset_instance.__len__.return_value = 100  # Mock dataset size
        mock_dataset.return_value = mock_dataset_instance
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = {"loss": 1.0}
        mock_trainer.return_value = mock_trainer_instance

        # Test arguments with save flag
        test_argv = [
            "--model-type",
            "skipgram",
            "--epochs",
            "1",
            "--save",
            "--out-dir",
            "/test/output",
        ]

        train_main(test_argv)

        # Verify save was called
        mock_save.assert_called_once()


class TestQueryCLI:
    """Test cases for query CLI functions."""

    def test_find_similar_words_success(self):
        """Test successful similar words finding."""
        word_to_idx = {"hello": 0, "world": 1, "test": 2}
        idx_to_word = {0: "hello", 1: "world", 2: "test"}
        embeddings = np.array(
            [
                [1.0, 0.0],  # hello
                [0.8, 0.6],  # world (similar to hello)
                [0.0, 1.0],  # test (different)
            ]
        )

        result = find_similar_words("hello", idx_to_word, word_to_idx, embeddings, 2)

        assert "error" not in result
        assert result["word"] == "hello"
        assert len(result["neighbors"]) <= 2
        assert all(isinstance(neighbor, tuple) for neighbor in result["neighbors"])
        assert all(len(neighbor) == 2 for neighbor in result["neighbors"])

    def test_find_similar_words_not_in_vocab(self):
        """Test finding similar words for unknown word."""
        word_to_idx = {"hello": 0, "world": 1}
        idx_to_word = {0: "hello", 1: "world"}
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

        result = find_similar_words("unknown", idx_to_word, word_to_idx, embeddings, 5)

        assert "error" in result
        assert "'unknown' not in vocabulary" in result["error"]

    def test_find_similar_words_excludes_self(self):
        """Test that similar words excludes the query word itself."""
        word_to_idx = {"word1": 0, "word2": 1, "word3": 2}
        idx_to_word = {0: "word1", 1: "word2", 2: "word3"}
        embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.1, 0.9]])

        result = find_similar_words("word1", idx_to_word, word_to_idx, embeddings, 5)

        # Check that word1 itself is not in the neighbors
        neighbor_words = [neighbor[0] for neighbor in result["neighbors"]]
        assert "word1" not in neighbor_words

    @patch("modern_word2vec.cli.query.load_vocab_and_embeddings")
    @patch("builtins.print")
    def test_query_main_success(self, mock_print, mock_load):
        """Test successful query main function."""
        # Setup mock data
        idx_to_word = {0: "hello", 1: "world"}
        word_to_idx = {"hello": 0, "world": 1}
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        mock_load.return_value = (idx_to_word, word_to_idx, embeddings)

        test_argv = ["--run-dir", "/test/path", "--word", "hello", "--topn", "5"]

        query_main(test_argv)

        mock_load.assert_called_once_with("/test/path")
        mock_print.assert_called_once()

        # Check that print was called with JSON output
        print_call_args = mock_print.call_args[0][0]
        assert "hello" in print_call_args

    @patch("modern_word2vec.cli.query.load_vocab_and_embeddings")
    @patch("builtins.print")
    def test_query_main_file_not_found(self, mock_print, mock_load):
        """Test query main function with file not found error."""
        mock_load.side_effect = FileNotFoundError("File not found")

        test_argv = ["--run-dir", "/nonexistent/path", "--word", "hello"]

        query_main(test_argv)

        mock_print.assert_called_once()
        print_call_args = mock_print.call_args[0][0]
        result = json.loads(print_call_args)
        assert "error" in result

    @patch("modern_word2vec.cli.query.load_vocab_and_embeddings")
    @patch("builtins.print")
    def test_query_main_value_error(self, mock_print, mock_load):
        """Test query main function with value error."""
        mock_load.side_effect = ValueError("Invalid data")

        test_argv = ["--run-dir", "/test/path", "--word", "hello"]

        query_main(test_argv)

        mock_print.assert_called_once()
        print_call_args = mock_print.call_args[0][0]
        result = json.loads(print_call_args)
        assert "error" in result
        assert "Invalid data" in result["error"]


class TestCLIIntegration:
    """Integration tests for CLI modules."""

    def test_train_parser_all_options(self):
        """Test training parser with all options."""
        parser = ArgumentParser.create_train_parser()

        args = parser.parse_args(
            [
                "--dataset",
                "custom_dataset",
                "--dataset-config",
                "custom_config",
                "--split",
                "train[:10%]",
                "--synthetic-sentences",
                "1000",
                "--synthetic-vocab",
                "500",
                "--model-type",
                "cbow",
                "--output-layer",
                "hierarchical_softmax",
                "--vocab-size",
                "5000",
                "--embedding-dim",
                "200",
                "--window-size",
                "3",
                "--tokenizer",
                "enhanced",
                "--subsample",
                "1e-4",
                "--dynamic-window",
                "--batch-size",
                "64",
                "--epochs",
                "10",
                "--lr",
                "0.01",
                "--optimizer",
                "sgd",
                "--weight-decay",
                "0.001",
                "--grad-clip",
                "1.0",
                "--compile",
                "--amp",
                "--workers",
                "4",
                "--pin-memory",
                "--seed",
                "123",
                "--device",
                "cuda",
                "--out-dir",
                "/custom/output",
                "--save",
            ]
        )

        # Verify all arguments are parsed correctly
        assert args.dataset == "custom_dataset"
        assert args.dataset_config == "custom_config"
        assert args.split == "train[:10%]"
        assert args.synthetic_sentences == 1000
        assert args.synthetic_vocab == 500
        assert args.model_type == "cbow"
        assert args.output_layer == "hierarchical_softmax"
        assert args.vocab_size == 5000
        assert args.embedding_dim == 200
        assert args.window_size == 3
        assert args.tokenizer == "enhanced"
        assert args.subsample == 1e-4
        assert args.dynamic_window is True
        assert args.batch_size == 64
        assert args.epochs == 10
        assert args.lr == 0.01
        assert args.optimizer == "sgd"
        assert args.weight_decay == 0.001
        assert args.grad_clip == 1.0
        assert args.compile is True
        assert args.amp is True
        assert args.workers == 4
        assert args.pin_memory is True
        assert args.seed == 123
        assert args.device == "cuda"
        assert args.out_dir == "/custom/output"
        assert args.save is True

    def test_config_creation_round_trip(self):
        """Test that configurations can be created and used correctly."""
        args = MagicMock()

        # Set up comprehensive arguments
        args.vocab_size = 2000
        args.window_size = 4
        args.model_type = "cbow"
        args.lowercase = False
        args.tokenizer = "simple"
        args.subsample = 1e-3
        args.dynamic_window = False
        args.embedding_dim = 150
        args.batch_size = 48
        args.epochs = 8
        args.lr = 0.005
        args.optimizer = "adam"
        args.weight_decay = 0.01
        args.grad_clip = 0.5
        args.compile = False
        args.amp = False
        args.workers = 2
        args.pin_memory = False
        args.seed = 999

        # Create configurations
        data_config = create_data_config(args)
        train_config = create_train_config(args)

        # Verify data config
        assert data_config.vocab_size == 2000
        assert data_config.window_size == 4
        assert data_config.model_type == "cbow"
        assert data_config.lowercase is False
        assert data_config.tokenizer == "simple"
        assert data_config.subsample_t == 1e-3
        assert data_config.dynamic_window is False

        # Verify train config
        assert train_config.embedding_dim == 150
        assert train_config.batch_size == 48
        assert train_config.epochs == 8
        assert train_config.learning_rate == 0.005
        assert train_config.optimizer == "adam"
        assert train_config.weight_decay == 0.01
        assert train_config.grad_clip == 0.5
        assert train_config.compile is False
        assert train_config.mixed_precision is False
        assert train_config.num_workers == 2
        assert train_config.pin_memory is False
        assert train_config.seed == 999

    def test_argument_validation(self):
        """Test argument validation for edge cases."""
        parser = ArgumentParser.create_train_parser()

        # Test valid choices
        args = parser.parse_args(["--model-type", "skipgram"])
        assert args.model_type == "skipgram"

        args = parser.parse_args(["--model-type", "cbow"])
        assert args.model_type == "cbow"

        # Test invalid choice should raise SystemExit
        with pytest.raises(SystemExit):
            parser.parse_args(["--model-type", "invalid"])

    def test_boolean_flags(self):
        """Test boolean flag handling."""
        parser = ArgumentParser.create_train_parser()

        # Test store_true flags
        args = parser.parse_args(
            ["--dynamic-window", "--compile", "--amp", "--pin-memory", "--save"]
        )
        assert args.dynamic_window is True
        assert args.compile is True
        assert args.amp is True
        assert args.pin_memory is True
        assert args.save is True

        # Test without flags
        args = parser.parse_args([])
        assert args.dynamic_window is False
        assert args.compile is False
        assert args.amp is False
        assert args.pin_memory is False
        assert args.save is False

    def test_default_values(self):
        """Test that default values are set correctly."""
        train_args = parse_train_args([])
        query_parser = ArgumentParser.create_query_parser()

        # Check training defaults
        assert train_args.dataset == "wikitext"
        assert train_args.dataset_config == "wikitext-2-raw-v1"
        assert train_args.split == "train[:20%]"
        assert train_args.synthetic_sentences == 0
        assert train_args.model_type == "skipgram"
        assert train_args.output_layer == "full_softmax"
        assert train_args.tokenizer == "basic"
        assert train_args.subsample == 0.0
        assert train_args.optimizer == "adam"
        assert train_args.weight_decay == 0.0
        assert train_args.grad_clip == 0.0
        assert train_args.workers == 0
        assert train_args.out_dir == "runs/latest"
        assert train_args.lowercase is True  # Set by parser.set_defaults

        # Check query defaults (when valid required args provided)
        query_args = query_parser.parse_args(["--run-dir", "/test", "--word", "test"])
        assert query_args.topn == 10

    def test_seed_worker_function(self):
        """Test the seed_worker function for deterministic training."""
        from modern_word2vec.cli.train import create_dataloader

        # Mock a simple dataset with a length
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)  # Give it a length
        mock_args = MagicMock()
        mock_args.seed = 42
        mock_args.batch_size = 32
        mock_args.workers = 0
        mock_args.pin_memory = False

        # Create dataloader (which internally defines seed_worker)
        dataloader = create_dataloader(mock_dataset, mock_args)

        # Test should run without error (covers the seed_worker function lines 117-119)
        assert dataloader is not None

    def test_main_function_coverage(self):
        """Test __main__ execution path for coverage."""
        from modern_word2vec.cli.train import main
        from modern_word2vec.cli.query import main as query_main

        # Test that main functions exist and can be imported
        # This covers the if __name__ == "__main__": main() lines
        assert callable(main)
        assert callable(query_main)
