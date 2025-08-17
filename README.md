# modern-word2vec

Modern Word2Vec (Skip-gram and CBOW) implementation with PyTorch best practices and comprehensive CLI. Built with [uv](https://github.com/astral-sh/uv) for fast dependency management.

## Features

- **Skip-gram and CBOW models** with full softmax, **hierarchical softmax**, and **negative sampling** output layers
- **Modern PyTorch practices**: mixed precision, torch.compile, proper device handling  
- **Configurable preprocessing**: tokenization, vocabulary building, dynamic context windows
- **Advanced training**: subsampling, gradient clipping, streaming for large datasets
- **Comprehensive CLI**: easy training and querying commands with extensive examples
- **Modular architecture**: clean separation of concerns, fully typed codebase

Complete benchmarking suite for classical word2vec variants with efficient large-vocabulary training.

## Installation

### Using uv (Recommended)

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install modern-word2vec
uv pip install modern-word2vec
```

### Development Installation

```bash
git clone https://github.com/hunterheidenreich/modern-word2vec.git
cd modern-word2vec
uv venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

**Why uv?** 10-100x faster than pip, reliable cross-platform dependency resolution, modern Rust-based tooling.

*Alternative: `pip install modern-word2vec`*

This provides the `word2vec-train` and `word2vec-query` console commands.

## Quick Start

### Essential Examples

**Basic Skip-gram training:**
```bash
word2vec-train --dataset wikitext --dataset-config wikitext-2-raw-v1 --split "train[:20%]" \
    --model-type skipgram --vocab-size 20000 --embedding-dim 100 --save
```

**CBOW with hierarchical softmax:**
```bash
word2vec-train --model-type cbow --output-layer hierarchical_softmax \
    --vocab-size 50000 --embedding-dim 300 --epochs 5 --save
```

**Skip-gram with negative sampling:**
```bash
word2vec-train --model-type skipgram --output-layer negative_sampling \
    --num-negative 10 --vocab-size 50000 --embedding-dim 300 --save
```

**GPU-optimized training:**
```bash
word2vec-train --device cuda --amp --compile --batch-size 1024 \
    --workers 4 --pin-memory --save
```

**Query embeddings:**
```bash
word2vec-query --run-dir runs/latest --word king --topn 10
```

### More Examples

<details>
<summary>Click to expand additional training configurations</summary>

**Local text file:**
```bash
word2vec-train --text-file corpus.txt --model-type skipgram --save
```

**Large vocabulary with streaming:**
```bash
word2vec-train --streaming --vocab-file vocab.json --output-layer negative_sampling \
    --num-negative 15 --vocab-size 200000 --buffer-size 50000 --save
```

**Synthetic data for testing:**
```bash
word2vec-train --synthetic-sentences 10000 --synthetic-vocab 5000 --save
```

</details>

### Python API

```python
from modern_word2vec import (
    SkipGramModel, CBOWModel, DataConfig, Word2VecDataset, 
    TrainConfig, Trainer, get_device, set_seed
)

# Basic training setup
set_seed(42)
device = get_device()

data_config = DataConfig(vocab_size=10000, window_size=2, model_type="skipgram")
dataset = Word2VecDataset(["your training texts"], data_config)
model = SkipGramModel(dataset.vocab_size, embedding_dim=100)

train_config = TrainConfig(embedding_dim=100, batch_size=256, epochs=5)
trainer = Trainer(model, device, train_config)
stats = trainer.train(DataLoader(dataset, batch_size=256, shuffle=True))
```

See [examples/](examples/) for advanced configurations including streaming datasets and custom similarity analysis.

## Architecture

The project is organized into a clean, modular structure:

```
src/modern_word2vec/
├── __init__.py              # Main package exports
├── models/                  # Model definitions
│   └── __init__.py         # SkipGramModel, CBOWModel
├── data/                   # Data processing
│   └── __init__.py         # DataConfig, Word2VecDataset, data utilities
├── training/               # Training logic
│   └── __init__.py         # TrainConfig, Trainer
├── utils/                  # Utilities
│   └── __init__.py         # Device handling, embeddings export/import
└── cli/                    # Command-line interface
    ├── __init__.py
    ├── train.py            # Training CLI
    └── query.py            # Query CLI
```

## Performance & Configuration

### Key Optimizations

- **Device auto-selection**: CUDA > MPS > CPU, or specify `--device {cuda|mps|cpu}`
- **Mixed precision**: `--amp` for 1.5-2x CUDA speedup
- **Compilation**: `--compile` for PyTorch 2.0+ optimization  
- **Multi-processing**: `--workers N` for parallel data loading
- **Memory**: `--pin-memory` auto-enabled for CUDA

### Model Selection Guide

| Use Case | Model | Output Layer | Notes |
|----------|--------|-------------|--------|
| Small datasets | Skip-gram | Full softmax | Better for rare words |
| Medium datasets | CBOW | Negative sampling | Good balance of speed/quality |
| Large datasets | CBOW | Hierarchical softmax | Fastest training |
| Large vocab (>50k) | Either | Hierarchical softmax | O(log V) complexity |
| Quality focus | Skip-gram | Negative sampling | Best embeddings with k=5-20 |

### Dataset Size Guidelines

- **Small (<100MB)**: Standard mode, batch size 256-512
- **Medium (100MB-1GB)**: Enable `--amp --workers 4`  
- **Large (>1GB)**: Use `--streaming` with pre-built vocabulary

## Output Files

Training with `--save` creates:
- `embeddings.npy` — Input embeddings matrix (vocab_size × embedding_dim)
- `vocab.json` — Word mappings and metadata  
- `config.json` — Training configuration

### Working with Embeddings

```python
from modern_word2vec.utils import load_vocab_and_embeddings

# Load saved embeddings
idx_to_word, word_to_idx, embeddings = load_vocab_and_embeddings("runs/latest")
word_vector = embeddings[word_to_idx["king"]]
```

Export to other formats or integrate with gensim - see [examples/](examples/) for details.

## Troubleshooting

**Memory issues**: Use `--streaming --batch-size 128` or reduce `--vocab-size`  
**Slow training**: Enable `--amp --compile --workers 4` for GPU, use hierarchical softmax for large vocabs  
**Poor quality**: Increase `--epochs`, use `--subsample 1e-5`, try larger `--embedding-dim`

## Best Practices

1. **Start simple**: Use defaults first, then optimize
2. **Use uv**: Faster, more reliable dependency management  
3. **Monitor training**: Check JSON output for loss/throughput
4. **Save models**: Always use `--save` for important runs
5. **Scale gradually**: Increase vocab size and dimensions incrementally

## Notes

- **Hierarchical softmax**: O(log V) complexity, ideal for large vocabularies (>50k words)
- **Negative sampling**: O(k) complexity where k is number of negative samples, excellent quality with k=5-20
- **Reproducibility**: All training uses deterministic seeding for consistent results

## License

MIT License - see [LICENSE](LICENSE) for details.
