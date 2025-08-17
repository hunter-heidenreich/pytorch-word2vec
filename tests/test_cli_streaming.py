"""Tests for streaming CLI functionality."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch, mock_open

import pytest

from modern_word2vec.cli.train_streaming import (
    create_standard_dataloader,
    create_streaming_components,
    main as streaming_main,
)


class TestStreamingCLI:
    """Test cases for streaming CLI functionality."""

    @patch("modern_word2vec.cli.train_streaming.Word2VecDataset")
    @patch("modern_word2vec.cli.train_streaming.get_device")
    def test_create_standard_dataloader(self, mock_get_device, mock_dataset):
        """Test standard dataloader creation."""
        # Setup mocks
        mock_get_device.return_value = MagicMock(type="cpu")
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.vocab_size = 1000
        mock_dataset_instance.__len__.return_value = 100
        mock_dataset.return_value = mock_dataset_instance

        # Create mock args
        args = MagicMock()
        args.vocab_size = 1000
        args.window_size = 3
        args.model_type = "skipgram"
        args.lower = True
        args.tokenizer = "basic"
        args.subsample = 0.0
        args.dynamic_window = False
        args.batch_size = 32
        args.pin_memory = False
        args.workers = 0
        args.seed = 42
        args.device = "cpu"

        raw_texts = ["test text 1", "test text 2"]

        dataloader, vocab_size = create_standard_dataloader(args, raw_texts)

        assert vocab_size == 1000
        assert dataloader.batch_size == 32
        mock_dataset.assert_called_once()

    @patch("modern_word2vec.cli.train_streaming.Word2VecDataset")
    @patch("modern_word2vec.cli.train_streaming.get_device")
    def test_create_standard_dataloader_cbow(self, mock_get_device, mock_dataset):
        """Test standard dataloader creation for CBOW model."""
        mock_get_device.return_value = MagicMock(type="cpu")
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.vocab_size = 500
        mock_dataset_instance.__len__.return_value = 50
        mock_dataset.return_value = mock_dataset_instance

        args = MagicMock()
        args.vocab_size = 500
        args.window_size = 2
        args.model_type = "cbow"
        args.lower = False
        args.tokenizer = "enhanced"
        args.subsample = 1e-5
        args.dynamic_window = True
        args.batch_size = 16
        args.pin_memory = True
        args.workers = 2
        args.seed = 123
        args.device = "cuda"

        raw_texts = ["text sample"]

        dataloader, vocab_size = create_standard_dataloader(args, raw_texts)

        assert vocab_size == 500
        assert dataloader.batch_size == 16
        # Should use cbow_collate for CBOW model
        assert dataloader.collate_fn is not None

    @patch("modern_word2vec.data.streaming.StreamingWord2VecDataset")
    @patch("modern_word2vec.data.streaming.load_vocabulary")
    def test_create_streaming_components_with_vocab_file(
        self, mock_load_vocab, mock_streaming_dataset
    ):
        """Test streaming components creation with pre-built vocabulary."""
        # Setup mocks
        mock_vocab = {"word1": 0, "word2": 1}
        mock_load_vocab.return_value = mock_vocab
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.vocab_size = 2
        mock_streaming_dataset.return_value = mock_dataset_instance

        args = MagicMock()
        args.vocab_size = 1000
        args.window_size = 3
        args.model_type = "skipgram"
        args.lower = True
        args.tokenizer = "basic"
        args.subsample = 0.0
        args.dynamic_window = False
        args.batch_size = 32
        args.vocab_file = "/path/to/vocab.json"
        args.buffer_size = 5000
        args.vocab_sample_size = 100000
        args.shuffle_buffer_size = 500
        args.seed = 42

        text_source = "test_dataset"

        dataloader, streaming_dataset = create_streaming_components(args, text_source)

        mock_load_vocab.assert_called_once_with("/path/to/vocab.json")
        mock_streaming_dataset.assert_called_once()
        assert streaming_dataset == mock_dataset_instance

    @patch("modern_word2vec.data.streaming.StreamingWord2VecDataset")
    @patch("modern_word2vec.data.streaming.build_vocabulary_from_stream")
    def test_create_streaming_components_build_vocab_string(
        self, mock_build_vocab, mock_streaming_dataset
    ):
        """Test streaming components with vocabulary building from string source."""
        mock_vocab = {"word1": 0, "word2": 1, "word3": 2}
        mock_build_vocab.return_value = mock_vocab
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.vocab_size = 3
        mock_streaming_dataset.return_value = mock_dataset_instance

        args = MagicMock()
        args.vocab_size = 1000
        args.window_size = 2
        args.model_type = "cbow"
        args.lower = False
        args.tokenizer = "simple"
        args.subsample = 1e-4
        args.dynamic_window = True
        args.batch_size = 64
        args.vocab_file = None
        args.buffer_size = 8000
        args.vocab_sample_size = 500000
        args.shuffle_buffer_size = 1000
        args.seed = 99

        text_source = "/path/to/text/file.txt"  # String source

        with patch("builtins.print"):
            dataloader, streaming_dataset = create_streaming_components(
                args, text_source
            )

        mock_build_vocab.assert_called_once()
        mock_streaming_dataset.assert_called_once()

    @patch("modern_word2vec.data.streaming.StreamingWord2VecDataset")
    @patch("modern_word2vec.data.streaming.build_vocabulary_from_stream")
    def test_create_streaming_components_build_vocab_list(
        self, mock_build_vocab, mock_streaming_dataset
    ):
        """Test streaming components with vocabulary building from list source."""
        mock_vocab = {"hello": 0, "world": 1}
        mock_build_vocab.return_value = mock_vocab
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.vocab_size = 2
        mock_streaming_dataset.return_value = mock_dataset_instance

        args = MagicMock()
        args.vocab_size = 500
        args.window_size = 5
        args.model_type = "skipgram"
        args.lower = True
        args.tokenizer = "enhanced"
        args.subsample = 0.0
        args.dynamic_window = False
        args.batch_size = 128
        args.vocab_file = None
        args.buffer_size = 10000
        args.vocab_sample_size = 2000000
        args.shuffle_buffer_size = 2000
        args.seed = 777

        text_source = ["hello world", "test sentence"]  # List source

        with patch("builtins.print"):
            dataloader, streaming_dataset = create_streaming_components(
                args, text_source
            )

        mock_build_vocab.assert_called_once()
        mock_streaming_dataset.assert_called_once()

    @patch("modern_word2vec.cli.train_streaming.Trainer")
    @patch("modern_word2vec.cli.train_streaming.CBOWModel")
    @patch("modern_word2vec.cli.train_streaming.create_standard_dataloader")
    @patch("modern_word2vec.cli.train_streaming.generate_synthetic_texts")
    @patch("modern_word2vec.cli.train_streaming.set_seed")
    @patch("modern_word2vec.cli.train_streaming.get_device")
    @patch("builtins.print")
    def test_main_synthetic_standard_mode(
        self,
        mock_print,
        mock_get_device,
        mock_set_seed,
        mock_generate,
        mock_create_dataloader,
        mock_model,
        mock_trainer,
    ):
        """Test main function with synthetic data in standard mode."""
        # Setup mocks
        mock_get_device.return_value = MagicMock(type="cpu")
        mock_generate.return_value = ["synthetic text 1", "synthetic text 2"]
        mock_dataloader = MagicMock()
        mock_dataloader.dataset = MagicMock()
        mock_create_dataloader.return_value = (mock_dataloader, 100)
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = {"loss": 0.5}
        mock_trainer.return_value = mock_trainer_instance

        test_argv = [
            "--synthetic-sentences",
            "100",
            "--synthetic-vocab",
            "50",
            "--model-type",
            "cbow",
            "--epochs",
            "1",
            "--batch-size",
            "32",
        ]

        streaming_main(test_argv)

        mock_set_seed.assert_called_once()
        mock_get_device.assert_called_once()
        mock_generate.assert_called_once()
        mock_create_dataloader.assert_called_once()
        mock_model.assert_called_once_with(100, 100)  # vocab_size, embedding_dim
        mock_trainer.assert_called_once()
        mock_trainer_instance.train.assert_called_once()

    @patch("modern_word2vec.cli.train_streaming.Trainer")
    @patch("modern_word2vec.cli.train_streaming.SkipGramModel")
    @patch("modern_word2vec.cli.train_streaming.create_standard_dataloader")
    @patch("modern_word2vec.cli.train_streaming.set_seed")
    @patch("modern_word2vec.cli.train_streaming.get_device")
    @patch("builtins.open", mock_open(read_data="line1\nline2\nline3\n"))
    @patch("builtins.print")
    def test_main_file_standard_mode(
        self,
        mock_print,
        mock_get_device,
        mock_set_seed,
        mock_create_dataloader,
        mock_model,
        mock_trainer,
    ):
        """Test main function with file input in standard mode."""
        mock_get_device.return_value = MagicMock(type="cpu")
        mock_dataloader = MagicMock()
        mock_dataloader.dataset = MagicMock()
        mock_create_dataloader.return_value = (mock_dataloader, 200)
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = {"loss": 0.3}
        mock_trainer.return_value = mock_trainer_instance

        test_argv = [
            "--text-file",
            "/path/to/test.txt",
            "--model-type",
            "skipgram",
            "--epochs",
            "2",
        ]

        streaming_main(test_argv)

        mock_create_dataloader.assert_called_once()
        mock_model.assert_called_once_with(200, 100)  # vocab_size, embedding_dim

    @patch("modern_word2vec.cli.train_streaming.Trainer")
    @patch("modern_word2vec.cli.train_streaming.SkipGramModel")
    @patch("modern_word2vec.cli.train_streaming.create_streaming_components")
    @patch("modern_word2vec.cli.train_streaming.load_texts_from_hf")
    @patch("modern_word2vec.cli.train_streaming.set_seed")
    @patch("modern_word2vec.cli.train_streaming.get_device")
    @patch("builtins.print")
    def test_main_hf_standard_mode(
        self,
        mock_print,
        mock_get_device,
        mock_set_seed,
        mock_load_hf,
        mock_create_streaming,
        mock_model,
        mock_trainer,
    ):
        """Test main function with HF dataset in standard mode."""
        mock_get_device.return_value = MagicMock(type="cpu")
        mock_load_hf.return_value = ["hf text 1", "hf text 2"]
        mock_dataloader = MagicMock()
        mock_streaming_dataset = MagicMock()
        mock_streaming_dataset.vocab_size = 300
        mock_create_streaming.return_value = (mock_dataloader, mock_streaming_dataset)
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = {"loss": 0.4}
        mock_trainer.return_value = mock_trainer_instance

        test_argv = [
            "--dataset",
            "test_dataset",
            "--dataset-config",
            "test_config",
            "--split",
            "train[:5%]",
        ]

        streaming_main(test_argv)

        mock_load_hf.assert_called_once_with(
            "test_dataset", "test_config", "train[:5%]"
        )

    @patch("modern_word2vec.cli.train_streaming.Trainer")
    @patch("modern_word2vec.cli.train_streaming.CBOWModel")
    @patch("modern_word2vec.cli.train_streaming.create_streaming_components")
    @patch("modern_word2vec.cli.train_streaming.load_streaming_dataset")
    @patch("modern_word2vec.cli.train_streaming.set_seed")
    @patch("modern_word2vec.cli.train_streaming.get_device")
    @patch("builtins.print")
    def test_main_streaming_mode(
        self,
        mock_print,
        mock_get_device,
        mock_set_seed,
        mock_load_streaming,
        mock_create_streaming,
        mock_model,
        mock_trainer,
    ):
        """Test main function in streaming mode."""
        mock_get_device.return_value = MagicMock(type="cuda")
        mock_stream_source = MagicMock()
        mock_load_streaming.return_value = mock_stream_source
        mock_dataloader = MagicMock()
        mock_streaming_dataset = MagicMock()
        mock_streaming_dataset.vocab_size = 500
        mock_create_streaming.return_value = (mock_dataloader, mock_streaming_dataset)
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = {"loss": 0.2}
        mock_trainer.return_value = mock_trainer_instance

        test_argv = [
            "--streaming",
            "--dataset",
            "large_dataset",
            "--model-type",
            "cbow",
            "--batch-size",
            "64",
            "--buffer-size",
            "20000",
        ]

        streaming_main(test_argv)

        mock_load_streaming.assert_called_once()
        mock_create_streaming.assert_called_once()
        mock_model.assert_called_once_with(500, 100)

    @patch("modern_word2vec.cli.train_streaming.export_embeddings")
    @patch("modern_word2vec.cli.train_streaming.Trainer")
    @patch("modern_word2vec.cli.train_streaming.SkipGramModel")
    @patch("modern_word2vec.cli.train_streaming.create_standard_dataloader")
    @patch("modern_word2vec.cli.train_streaming.generate_synthetic_texts")
    @patch("modern_word2vec.cli.train_streaming.set_seed")
    @patch("modern_word2vec.cli.train_streaming.get_device")
    @patch("builtins.print")
    def test_main_with_save_standard_mode(
        self,
        mock_print,
        mock_get_device,
        mock_set_seed,
        mock_generate,
        mock_create_dataloader,
        mock_model,
        mock_trainer,
        mock_export,
    ):
        """Test main function with save option in standard mode."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_get_device.return_value = MagicMock(type="cpu")
            mock_generate.return_value = ["text1", "text2"]
            mock_dataloader = MagicMock()
            mock_dataset = MagicMock()
            mock_dataloader.dataset = mock_dataset

            # Mock the cfg attribute to be a proper dataclass-like object
            from modern_word2vec import DataConfig

            mock_dataset.cfg = DataConfig()

            mock_create_dataloader.return_value = (mock_dataloader, 150)
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance
            mock_trainer_instance = MagicMock()
            mock_trainer_instance.train.return_value = {"loss": 0.1}
            mock_trainer.return_value = mock_trainer_instance

            test_argv = ["--synthetic-sentences", "50", "--save", "--out-dir", tmp_dir]

            streaming_main(test_argv)

            mock_export.assert_called_once_with(
                mock_model_instance, tmp_dir, mock_dataset
            )

            # Check that config was saved
            config_path = os.path.join(tmp_dir, "config.json")
            assert os.path.exists(config_path)

            with open(config_path, "r") as f:
                config = json.load(f)
            assert "train" in config
            assert "data" in config

    @patch("modern_word2vec.cli.train_streaming.export_embeddings")
    @patch("modern_word2vec.cli.train_streaming.Trainer")
    @patch(
        "modern_word2vec.cli.train_streaming.SkipGramModel"
    )  # Changed from CBOWModel
    @patch("modern_word2vec.cli.train_streaming.create_streaming_components")
    @patch("modern_word2vec.cli.train_streaming.generate_synthetic_texts")
    @patch("modern_word2vec.cli.train_streaming.set_seed")
    @patch("modern_word2vec.cli.train_streaming.get_device")
    @patch("builtins.print")
    def test_main_with_save_streaming_mode(
        self,
        mock_print,
        mock_get_device,
        mock_set_seed,
        mock_generate,
        mock_create_streaming,
        mock_model,
        mock_trainer,
        mock_export,
    ):
        """Test main function with save option in streaming mode."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_get_device.return_value = MagicMock(type="cpu")
            mock_generate.return_value = ["stream text"]
            mock_dataloader = MagicMock()
            mock_streaming_dataset = MagicMock()
            mock_streaming_dataset.vocab_size = 75
            mock_create_streaming.return_value = (
                mock_dataloader,
                mock_streaming_dataset,
            )
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance
            mock_trainer_instance = MagicMock()
            mock_trainer_instance.train.return_value = {"loss": 0.15}
            mock_trainer.return_value = mock_trainer_instance

            test_argv = [
                "--streaming",
                "--synthetic-sentences",
                "25",
                "--save",
                "--out-dir",
                tmp_dir,
                "--buffer-size",
                "5000",
            ]

            streaming_main(test_argv)

            mock_export.assert_called_once_with(
                mock_model_instance, tmp_dir, mock_streaming_dataset
            )

            # Check config was saved with streaming info
            config_path = os.path.join(tmp_dir, "config.json")
            assert os.path.exists(config_path)

            with open(config_path, "r") as f:
                config = json.load(f)
            assert config["data"]["streaming"] is True
            assert config["data"]["buffer_size"] == 5000

    def test_argument_parser_comprehensive(self):
        """Test that the argument parser handles all options correctly."""
        # This tests the parser creation indirectly by testing main with various arguments
        test_cases = [
            # Basic arguments
            ["--model-type", "skipgram"],
            ["--model-type", "cbow"],
            # Data arguments
            [
                "--dataset",
                "custom",
                "--dataset-config",
                "config",
                "--split",
                "train[:1%]",
            ],
            ["--text-file", "/path/to/file.txt"],
            ["--synthetic-sentences", "100", "--synthetic-vocab", "50"],
            # Streaming arguments
            ["--streaming", "--vocab-file", "/vocab.json", "--buffer-size", "5000"],
            # Model configuration
            ["--vocab-size", "5000", "--embedding-dim", "200", "--window-size", "3"],
            [
                "--lower",
                "--tokenizer",
                "enhanced",
                "--subsample",
                "1e-5",
                "--dynamic-window",
            ],
            ["--no-lower"],
            # Training configuration
            ["--batch-size", "128", "--epochs", "5", "--lr", "0.01"],
            ["--optimizer", "sgd", "--weight-decay", "0.001", "--grad-clip", "1.0"],
            ["--compile", "--amp", "--workers", "4", "--pin-memory"],
            ["--seed", "123", "--device", "cuda"],
            # Output options
            ["--out-dir", "/custom/path", "--save"],
        ]

        for test_args in test_cases:
            # We don't actually run main, just test that the parser can parse these arguments
            from modern_word2vec.cli.train_streaming import main

            # Create a minimal test that the arguments parse without error
            # by mocking the execution part
            with patch("modern_word2vec.cli.train_streaming.set_seed"):
                with patch(
                    "modern_word2vec.cli.train_streaming.get_device"
                ) as mock_device:
                    with patch(
                        "modern_word2vec.cli.train_streaming.generate_synthetic_texts"
                    ) as mock_generate:
                        with patch(
                            "modern_word2vec.cli.train_streaming.create_standard_dataloader"
                        ) as mock_dataloader:
                            with patch(
                                "modern_word2vec.cli.train_streaming.Trainer"
                            ) as mock_trainer:
                                with patch("builtins.print"):
                                    mock_device.return_value = MagicMock(type="cpu")
                                    mock_generate.return_value = ["test"]
                                    mock_dataloader.return_value = (MagicMock(), 100)
                                    mock_trainer_instance = MagicMock()
                                    mock_trainer_instance.train.return_value = {}
                                    mock_trainer.return_value = mock_trainer_instance

                                    # Add synthetic sentences to avoid other data loading
                                    full_args = test_args + [
                                        "--synthetic-sentences",
                                        "1",
                                    ]

                                    try:
                                        main(full_args)
                                        # If we get here, the arguments parsed successfully
                                        assert True
                                    except SystemExit as e:
                                        # Argument parsing error
                                        if e.code != 0:
                                            pytest.fail(
                                                f"Argument parsing failed for {test_args}"
                                            )
                                    except Exception:
                                        # Some other error after parsing - that's ok for this test
                                        pass


class TestStreamingCLIEdgeCases:
    """Test edge cases and error conditions."""

    def test_device_optimization_cuda(self):
        """Test CUDA device optimizations are applied."""
        with patch(
            "modern_word2vec.cli.train_streaming.torch.backends.cudnn"
        ) as mock_cudnn:
            with patch("modern_word2vec.cli.train_streaming.torch.backends.cuda"):
                with patch(
                    "modern_word2vec.cli.train_streaming.torch.set_float32_matmul_precision"
                ):
                    with patch(
                        "modern_word2vec.cli.train_streaming.get_device"
                    ) as mock_device:
                        with patch(
                            "modern_word2vec.cli.train_streaming.Trainer"
                        ) as mock_trainer:
                            with patch(
                                "modern_word2vec.cli.train_streaming.create_standard_dataloader"
                            ) as mock_dataloader:
                                with patch(
                                    "modern_word2vec.cli.train_streaming.generate_synthetic_texts"
                                ) as mock_generate:
                                    with patch(
                                        "modern_word2vec.cli.train_streaming.set_seed"
                                    ):
                                        with patch("builtins.print"):
                                            mock_device.return_value = MagicMock(
                                                type="cuda"
                                            )
                                            mock_generate.return_value = ["test"]
                                            mock_dataloader.return_value = (
                                                MagicMock(),
                                                100,
                                            )
                                            mock_trainer_instance = MagicMock()
                                            mock_trainer_instance.train.return_value = {}
                                            mock_trainer.return_value = (
                                                mock_trainer_instance
                                            )

                                            streaming_main(
                                                [
                                                    "--synthetic-sentences",
                                                    "1",
                                                    "--device",
                                                    "cuda",
                                                ]
                                            )

                                            # Verify CUDA optimizations were attempted
                                            assert mock_cudnn.benchmark is True

    def test_device_optimization_exceptions(self):
        """Test that device optimization exceptions are handled gracefully."""
        with patch(
            "modern_word2vec.cli.train_streaming.torch.backends.cuda"
        ) as mock_cuda:
            with patch(
                "modern_word2vec.cli.train_streaming.torch.set_float32_matmul_precision",
                side_effect=Exception("Precision error"),
            ):
                mock_cuda.matmul.allow_tf32 = MagicMock(
                    side_effect=Exception("CUDA error")
                )

                with patch(
                    "modern_word2vec.cli.train_streaming.get_device"
                ) as mock_device:
                    with patch(
                        "modern_word2vec.cli.train_streaming.Trainer"
                    ) as mock_trainer:
                        with patch(
                            "modern_word2vec.cli.train_streaming.create_standard_dataloader"
                        ) as mock_dataloader:
                            with patch(
                                "modern_word2vec.cli.train_streaming.generate_synthetic_texts"
                            ) as mock_generate:
                                with patch(
                                    "modern_word2vec.cli.train_streaming.set_seed"
                                ):
                                    with patch("builtins.print"):
                                        mock_device.return_value = MagicMock(
                                            type="cuda"
                                        )
                                        mock_generate.return_value = ["test"]
                                        mock_dataloader.return_value = (
                                            MagicMock(),
                                            100,
                                        )
                                        mock_trainer_instance = MagicMock()
                                        mock_trainer_instance.train.return_value = {}
                                        mock_trainer.return_value = (
                                            mock_trainer_instance
                                        )

                                        # Should not raise an exception despite optimization errors
                                        streaming_main(
                                            [
                                                "--synthetic-sentences",
                                                "1",
                                                "--device",
                                                "cuda",
                                            ]
                                        )

                                        # Verify exception handling worked
                                        assert (
                                            True
                                        )  # If we get here, exceptions were handled
