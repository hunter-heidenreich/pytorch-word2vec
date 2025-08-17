"""Enhanced training CLI with streaming support.

FIXED BUGS in streaming workflow:
1. Two-Pass Bug: Eliminated the temporary dataset creation that consumed the iterator
2. Fake Vocabulary Bug: Now saves the real vocabulary from the streaming dataset

Key improvements:
- Single StreamingWord2VecDataset instance is created and reused
- Real vocabulary is saved, not fake placeholder words
- Clear guidance for vocabulary building strategies
- Support for pre-built vocabulary files for large datasets
"""

import argparse
import json
import os
import random
from dataclasses import asdict
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from modern_word2vec import (
    CBOWModel,
    DataConfig,
    SkipGramModel,
    TrainConfig,
    Trainer,
    Word2VecDataset,
    export_embeddings,
    get_device,
    set_seed,
)
from modern_word2vec.data import (
    cbow_collate,
    generate_synthetic_texts,
    load_texts_from_hf,
)

from modern_word2vec.data.streaming import (
    load_streaming_dataset,
)


def create_standard_dataloader(args, raw_texts: List[str]) -> DataLoader:
    """Create a standard (in-memory) dataloader."""
    data_config = DataConfig(
        vocab_size=args.vocab_size,
        window_size=args.window_size,
        model_type=args.model_type,
        lowercase=args.lower,
        tokenizer=args.tokenizer,
        subsample_t=args.subsample,
        dynamic_window=args.dynamic_window,
    )

    dataset = Word2VecDataset(raw_texts, data_config, rng=random.Random(args.seed))

    collate_fn = cbow_collate if args.model_type == "cbow" else None
    pin_memory = args.pin_memory and get_device(args.device).type == "cuda"

    def seed_worker(worker_id: int):
        base_seed = (args.seed + worker_id) % (2**32 - 1)
        random.seed(base_seed)
        np.random.seed(base_seed % (2**32 - 1))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=bool(args.workers > 0),
        worker_init_fn=seed_worker if args.workers > 0 else None,
    )

    return dataloader, dataset.vocab_size


def create_streaming_components(args, text_source):
    """Create streaming dataset and dataloader.

    For true streaming datasets, this requires either:
    1. A pre-built vocabulary file (recommended for large datasets)
    2. A repeatable text source (files, lists, HF dataset names)

    Note: For very large datasets, consider building vocabulary offline first.
    """
    from modern_word2vec.data.streaming import (
        StreamingDataConfig,
        StreamingWord2VecDataset,
        build_vocabulary_from_stream,
        load_vocabulary,
    )

    streaming_config = StreamingDataConfig(
        vocab_size=args.vocab_size,
        window_size=args.window_size,
        model_type=args.model_type,
        lowercase=args.lower,
        tokenizer=args.tokenizer,
        subsample_t=args.subsample,
        dynamic_window=args.dynamic_window,
        buffer_size=getattr(args, "buffer_size", 10000),
        vocab_sample_size=getattr(args, "vocab_sample_size", 1000000),
        shuffle_buffer_size=getattr(args, "shuffle_buffer_size", 1000),
    )

    # Check if user provided a pre-built vocabulary
    vocab_file = getattr(args, "vocab_file", None)

    if vocab_file:
        print(f"Loading pre-built vocabulary from {vocab_file}")
        vocab = load_vocabulary(vocab_file)
        training_source = text_source
    else:
        # For repeatable sources, we can build vocab and then retrain
        # This works for: file paths, lists, HF dataset names
        if isinstance(text_source, str):
            # File path or HF dataset name - repeatable
            print("Building vocabulary from repeatable source...")
            vocab = build_vocabulary_from_stream(text_source, streaming_config)
            training_source = text_source  # Can reuse the same source
        elif isinstance(text_source, list):
            # List - repeatable
            print("Building vocabulary from list...")
            vocab = build_vocabulary_from_stream(text_source, streaming_config)
            training_source = text_source  # Can reuse the same list
        else:
            # True iterator - can't repeat
            print("""
WARNING: Building vocabulary from a true iterator. This will consume the iterator
for vocabulary building, and training will have no data left.

For large streaming datasets, consider:
1. Pre-building vocabulary offline with build_vocabulary_from_stream()
2. Using --vocab-file to load pre-built vocabulary
3. Using file paths or HF dataset names instead of iterators
""")
            vocab = build_vocabulary_from_stream(text_source, streaming_config)
            training_source = []  # Empty - iterator was consumed

    # Create the streaming dataset
    streaming_dataset = StreamingWord2VecDataset(
        training_source, vocab, streaming_config, rng=random.Random(args.seed)
    )

    # Create dataloader
    def collate_fn(batch):
        if not batch:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        if args.model_type == "cbow":
            inputs = []
            targets = []
            for src, tgt in batch:
                if isinstance(src, torch.Tensor) and src.numel() > 0:
                    inputs.append(src)
                    targets.append(tgt)

            if inputs:
                max_len = max(inp.numel() for inp in inputs)
                padded_inputs = []
                for inp in inputs:
                    if inp.numel() < max_len:
                        padding = torch.zeros(max_len - inp.numel(), dtype=inp.dtype)
                        padded_inputs.append(torch.cat([inp, padding]))
                    else:
                        padded_inputs.append(inp)
                return torch.stack(padded_inputs), torch.stack(targets)
            else:
                return torch.empty(0, dtype=torch.long), torch.empty(
                    0, dtype=torch.long
                )
        else:
            inputs, targets = zip(*batch)
            return torch.stack(inputs), torch.stack(targets)

    dataloader = torch.utils.data.DataLoader(
        streaming_dataset,
        batch_size=args.batch_size,
        num_workers=0,  # Streaming datasets require num_workers=0
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader, streaming_dataset


def main(argv: Optional[List[str]] = None) -> None:
    """Main training function with streaming support."""
    parser = argparse.ArgumentParser(
        description="Train Word2Vec (Skip-gram/CBOW) with optional streaming"
    )

    # Data args
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="HF dataset name (default: wikitext)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-2-raw-v1",
        help="HF dataset config",
    )
    parser.add_argument(
        "--split", type=str, default="train[:20%]", help="HF dataset split slice"
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="Optional path to a local text file to train on",
    )
    parser.add_argument(
        "--synthetic-sentences",
        type=int,
        default=0,
        help="Generate N synthetic sentences for benchmarking (skips HF load)",
    )
    parser.add_argument(
        "--synthetic-vocab",
        type=int,
        default=None,
        help="Synthetic vocab size (defaults to --vocab-size)",
    )

    # Streaming options
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming dataset for large corpora",
    )
    parser.add_argument(
        "--vocab-file",
        type=str,
        default=None,
        help="Path to pre-built vocabulary JSON file (recommended for large streaming datasets)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=10000,
        help="Token buffer size for streaming (default: 10000)",
    )
    parser.add_argument(
        "--vocab-sample-size",
        type=int,
        default=1000000,
        help="Tokens to sample for vocab building in streaming mode (default: 1M)",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=1000,
        help="Training pairs shuffle buffer size for streaming (default: 1000)",
    )

    # Model/Data config
    parser.add_argument(
        "--model-type", type=str, default="skipgram", choices=["skipgram", "cbow"]
    )
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--lower", action="store_true", help="Lowercase text")
    parser.add_argument("--no-lower", dest="lower", action="store_false")
    parser.set_defaults(lower=True)
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="basic",
        choices=["basic", "enhanced", "simple", "split"],
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.0,
        help="Enable word subsampling with t (e.g., 1e-5)",
    )
    parser.add_argument("--dynamic-window", action="store_true")

    # Train config
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd"]
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument(
        "--compile", action="store_true", help="Use torch.compile if available"
    )
    parser.add_argument(
        "--amp", action="store_true", help="Use mixed precision on CUDA"
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "mps", "cpu", None]
    )

    # Output / eval
    parser.add_argument("--out-dir", type=str, default="runs/latest")
    parser.add_argument(
        "--save", action="store_true", help="Save embeddings and vocab after training"
    )

    args = parser.parse_args(argv)

    # Seed and device
    set_seed(args.seed)
    device = get_device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

    # Prepare data source
    if args.synthetic_sentences and args.synthetic_sentences > 0:
        rng = random.Random(args.seed)
        syn_vocab = args.synthetic_vocab or args.vocab_size
        raw_texts = generate_synthetic_texts(args.synthetic_sentences, syn_vocab, rng)
        print(f"Generated {len(raw_texts)} synthetic sentences (vocab={syn_vocab}).")
        text_source = raw_texts
    elif args.text_file:
        if args.streaming:
            # For streaming, we can pass the file path directly
            text_source = args.text_file
        else:
            with open(args.text_file, "r") as f:
                text_source = [line.strip() for line in f if line.strip()]
    else:
        if args.streaming:
            print("Loading streaming dataset from Hugging Face...")
            text_source = load_streaming_dataset(
                args.dataset, args.dataset_config, args.split
            )
        else:
            print("Loading dataset from Hugging Face...")
            text_source = load_texts_from_hf(
                args.dataset, args.dataset_config, args.split
            )
            print(f"Loaded {len(text_source)} lines of text.")

    # Create dataloader
    if args.streaming:
        print("Creating streaming dataloader...")
        dataloader, streaming_dataset = create_streaming_components(args, text_source)
        vocab_size = streaming_dataset.vocab_size
        print(f"Streaming mode enabled with vocab size: {vocab_size}")
    else:
        print("Creating standard dataloader...")
        dataloader, vocab_size = create_standard_dataloader(args, text_source)
        streaming_dataset = None  # No streaming dataset in standard mode
        print(f"Standard mode with vocab size: {vocab_size}")

    # Model
    if args.model_type == "skipgram":
        model = SkipGramModel(vocab_size, args.embedding_dim)
    else:
        model = CBOWModel(vocab_size, args.embedding_dim)

    # Train config
    train_config = TrainConfig(
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        compile=args.compile,
        mixed_precision=args.amp,
        num_workers=args.workers,
        pin_memory=args.pin_memory and device.type == "cuda",
        seed=args.seed,
    )

    # Train
    trainer = Trainer(model, device, train_config)
    print("Starting training...")

    train_stats = trainer.train(dataloader)

    print(json.dumps(train_stats, indent=2))

    if args.save:
        # For streaming datasets, we need to create a minimal dataset object for export
        if args.streaming:
            # Use the REAL streaming dataset with the actual vocabulary
            export_embeddings(model, args.out_dir, streaming_dataset)
        else:
            # We already have the dataset from standard loading
            export_embeddings(model, args.out_dir, dataloader.dataset)

        # Save config
        config_data = {"train": asdict(train_config)}
        if args.streaming:
            config_data["data"] = {
                "vocab_size": vocab_size,
                "window_size": args.window_size,
                "model_type": args.model_type,
                "streaming": True,
                "buffer_size": args.buffer_size,
                "vocab_sample_size": args.vocab_sample_size,
            }
        else:
            # Get data config from the dataset
            config_data["data"] = asdict(dataloader.dataset.cfg)

        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump(config_data, f, indent=2)


if __name__ == "__main__":
    main()
