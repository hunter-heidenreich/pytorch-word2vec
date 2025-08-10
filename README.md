# modern-word2vec

Modern Word2Vec (Skip-gram and CBOW) implementation with PyTorch best practices and a comprehensive CLI for training and evaluation.

## Features

- **Skip-gram and CBOW models** with full softmax and **hierarchical softmax** output layers
- **Modern PyTorch practices**: mixed precision, torch.compile, proper device handling
- **Configurable preprocessing**: tokenization, casing, vocabulary size, dynamic context windows
- **Advanced training features**: subsampling of frequent words, gradient clipping, reproducible seeding
- **Modular architecture**: clean separation of models, data processing, training, and utilities
- **Comprehensive CLI**: easy-to-use commands for training and querying embeddings
- **Type hints and documentation**: fully typed codebase with comprehensive documentation
- **Testing and CI**: pytest-based testing suite with coverage reporting

This implementation provides a complete benchmarking suite for classical word2vec training variants with both full softmax (baseline) and hierarchical softmax for efficient large-vocabulary training.

## Installation

### Using pip

```bash
pip install modern-word2vec
```

### Development Installation

```bash
git clone https://github.com/hunterheidenreich/modern-word2vec.git
cd modern-word2vec
pip install -e ".[dev]"
```

This provides the `word2vec-train` and `word2vec-query` console commands.

## Quick Start

### CLI Usage Examples

#### Basic Training Examples

**1. Skip-gram on WikiText-2 (recommended starting point):**
```bash
word2vec-train --dataset wikitext --dataset-config wikitext-2-raw-v1 --split "train[:20%]" \
    --model-type skipgram --vocab-size 20000 --embedding-dim 100 --window-size 2 \
    --batch-size 256 --epochs 3 --lr 2e-3 --save --out-dir runs/latest
```

**2. CBOW with subsampling and hierarchical softmax:**
```bash
word2vec-train --model-type cbow --subsample 1e-5 --output-layer hierarchical_softmax \
    --vocab-size 30000 --embedding-dim 300 --epochs 5 --save
```

**3. Train on local text file (one sentence per line):**
```bash
word2vec-train --text-file path/to/corpus.txt --model-type skipgram \
    --vocab-size 50000 --embedding-dim 200 --save --out-dir my_embeddings
```

#### Advanced Training Configurations

**4. High-performance GPU training with optimizations:**
```bash
word2vec-train --dataset wikitext --dataset-config wikitext-103-raw-v1 \
    --model-type skipgram --vocab-size 100000 --embedding-dim 300 \
    --batch-size 1024 --epochs 10 --lr 1e-3 \
    --amp --compile --workers 4 --pin-memory \
    --device cuda --save --out-dir runs/gpu_optimized
```

**5. Large vocabulary with hierarchical softmax for efficiency:**
```bash
word2vec-train --dataset bookcorpus --model-type skipgram \
    --vocab-size 200000 --embedding-dim 300 --window-size 5 \
    --output-layer hierarchical_softmax --tokenizer enhanced --lower --subsample 1e-5 \
    --batch-size 512 --epochs 20 --lr 5e-4 --optimizer sgd \
    --grad-clip 1.0 --save
```

**6. Streaming mode for very large datasets:**
```bash
# First, optionally build vocabulary offline for better performance
# Note: Future versions will include 'word2vec-build-vocab' command for this
python -m modern_word2vec.data.streaming build_vocab \
    --dataset openwebtext --vocab-size 500000 --output vocab_500k.json

# Then train with streaming
word2vec-train --streaming --vocab-file vocab_500k.json \
    --dataset openwebtext --model-type skipgram \
    --embedding-dim 300 --batch-size 256 --epochs 1 \
    --buffer-size 50000 --shuffle-buffer-size 5000 --save
```

#### Synthetic Data for Testing and Benchmarking

**7. Quick synthetic test run:**
```bash
word2vec-train --synthetic-sentences 10000 --synthetic-vocab 5000 \
    --model-type skipgram --embedding-dim 100 --epochs 3 --save
```

**8. Large synthetic benchmark with hierarchical softmax:**
```bash
word2vec-train --synthetic-sentences 1000000 --synthetic-vocab 50000 \
    --model-type cbow --output-layer hierarchical_softmax --embedding-dim 300 --batch-size 1024 \
    --epochs 10 --amp --compile --save --out-dir benchmark_run
```

#### Querying and Evaluation

**9. Query similar words after training:**
```bash
word2vec-query --run-dir runs/latest --word king --topn 10
word2vec-query --run-dir runs/latest --word computer --topn 15
word2vec-query --run-dir runs/latest --word beautiful --topn 5
```

**10. Batch querying multiple words:**
```bash
# Query multiple words and save results
for word in king queen man woman; do
    word2vec-query --run-dir runs/latest --word $word --topn 10 > results_${word}.json
done
```

### Python API Examples

#### Basic Usage

```python
from modern_word2vec import (
    SkipGramModel, 
    CBOWModel, 
    DataConfig, 
    Word2VecDataset, 
    TrainConfig, 
    Trainer,
    get_device,
    set_seed
)

# Set up reproducible training
set_seed(42)
device = get_device()

# Configure data processing
data_config = DataConfig(
    vocab_size=10000,
    window_size=2,
    model_type="skipgram",
    lowercase=True,
    subsample_t=1e-5
)

# Create dataset
texts = ["your training texts here", "one sentence per item"]
dataset = Word2VecDataset(texts, data_config)

# Create model with hierarchical softmax for large vocabularies
model = SkipGramModel(dataset.vocab_size, embedding_dim=100, 
                     output_layer_type="hierarchical_softmax", dataset=dataset)

# Configure training
train_config = TrainConfig(
    embedding_dim=100,
    batch_size=256,
    epochs=5,
    learning_rate=0.001
)

# Train the model
trainer = Trainer(model, device, train_config)
dataloader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True)
stats = trainer.train(dataloader)

print(f"Training completed in {stats['time_sec']:.2f} seconds")
```

#### Advanced Configuration Examples

**Advanced data configuration:**
```python
from modern_word2vec import DataConfig, Tokenizer, VocabularyBuilder

# Advanced data configuration
data_config = DataConfig(
    vocab_size=50000,
    window_size=5,
    model_type="skipgram",  # dynamic_window primarily designed for skip-gram
    lowercase=True,
    tokenizer="enhanced",  # Uses contractions, URL/email normalization
    min_token_length=2,
    max_token_length=30,
    normalize_numbers=True,
    subsample_t=1e-5,
    dynamic_window=True  # Recommended for skip-gram; less conventional for CBOW
)

# Custom tokenization workflow
tokenizer = Tokenizer(data_config)
vocab_builder = VocabularyBuilder(data_config.vocab_size, data_config.subsample_t)

# Process texts
texts = load_your_corpus()
tokens = tokenizer.tokenize_corpus(texts)
word_to_idx, idx_to_word, filtered_tokens = vocab_builder.build_vocabulary(
    tokens, random.Random(42)
)
```

**High-performance training setup:**
```python
from modern_word2vec import TrainConfig, setup_device_optimizations

# Optimal configuration for GPU training
device = get_device("cuda")
setup_device_optimizations(device)

train_config = TrainConfig(
    embedding_dim=300,
    batch_size=1024,
    epochs=20,
    learning_rate=1e-3,
    optimizer="sgd",  # As per original Word2Vec paper
    weight_decay=1e-6,
    grad_clip=1.0,
    compile=True,  # PyTorch 2.0+ optimization
    mixed_precision=True,  # Faster training on modern GPUs
    num_workers=4,
    pin_memory=True,
    seed=42
)
```

**Streaming dataset for large corpora:**
```python
from modern_word2vec.data.streaming import (
    StreamingWord2VecDataset,
    StreamingDataConfig,
    build_vocabulary_from_stream,
    create_streaming_dataloader
)

# Build vocabulary offline (one-time process)
streaming_config = StreamingDataConfig(
    vocab_size=100000,
    window_size=5,
    model_type="skipgram",
    buffer_size=50000,
    vocab_sample_size=10000000  # Sample 10M tokens for vocab
)

vocab = build_vocabulary_from_stream(
    text_source="path/to/large_corpus.txt",
    config=streaming_config,
    output_path="vocab_100k.json"
)

# Create streaming dataloader
dataloader = create_streaming_dataloader(
    text_source="path/to/large_corpus.txt",
    vocab=vocab,
    config=streaming_config,
    batch_size=512,
    seed=42
)
```

**Custom similarity and evaluation:**
```python
from modern_word2vec.utils import (
    load_vocab_and_embeddings,
    compute_cosine_similarities,
    find_similar
)

# Load trained embeddings
idx_to_word, word_to_idx, embeddings = load_vocab_and_embeddings("runs/my_model")

# Find similar words
similar_words = find_similar("king", model, dataset, top_n=10)

# Custom similarity computation
word_idx = word_to_idx["queen"]
query_embedding = embeddings[word_idx]
similarities = compute_cosine_similarities(query_embedding, embeddings)

# Word analogy: king - man + woman ≈ queen
king_vec = embeddings[word_to_idx["king"]]
man_vec = embeddings[word_to_idx["man"]]
woman_vec = embeddings[word_to_idx["woman"]]
analogy_vec = king_vec - man_vec + woman_vec

# Find closest to analogy vector, excluding input words
analogy_similarities = compute_cosine_similarities(analogy_vec, embeddings)
exclude_indices = {word_to_idx["king"], word_to_idx["man"], word_to_idx["woman"]}

# Get top candidates excluding the input words
sorted_indices = np.argsort(analogy_similarities)[::-1]  # Descending order
for idx in sorted_indices:
    if idx not in exclude_indices:
        result_word = idx_to_word[idx]
        similarity = analogy_similarities[idx]
        print(f"king - man + woman ≈ {result_word} (similarity: {similarity:.3f})")
        break
```

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

## Performance and Devices

### Device Optimization Strategies

- **Device auto-selection**: CUDA > MPS > CPU, or override with `--device {cuda|mps|cpu}`
- **Mixed precision**: Enable with `--amp` for faster training on CUDA (typically 1.5-2x speedup)
- **Compilation**: Use `--compile` for extra speed on PyTorch 2.0+ (hardware dependent, up to 20% faster)
- **Multi-processing**: Use `--workers N` for parallel data loading (Linux typically benefits more than macOS)
- **Memory optimization**: `--pin-memory` automatically enabled for CUDA training

### Performance Guidelines

**Small datasets (< 100MB text):**
```bash
word2vec-train --text-file small_corpus.txt --batch-size 256 --workers 2
```

**Medium datasets (100MB - 1GB text):**
```bash
word2vec-train --dataset wikitext --dataset-config wikitext-103-raw-v1 \
    --batch-size 512 --workers 4 --amp --pin-memory
```

**Large datasets (> 1GB text):**
```bash
word2vec-train --streaming --vocab-file prebuilt_vocab.json \
    --batch-size 1024 --amp --compile --buffer-size 100000
```

**GPU optimization (CUDA):**
```bash
word2vec-train --device cuda --amp --compile --batch-size 2048 \
    --workers 4 --pin-memory --grad-clip 1.0
```

**Apple Silicon optimization (MPS):**
```bash
word2vec-train --device mps --batch-size 512 --compile \
    --workers 2  # Note: MPS doesn't support mixed precision
```

### Streaming vs Standard Mode

**Use Standard Mode When:**
- Dataset fits in memory (< 1GB text)
- You need maximum throughput
- You want to use multiple workers
- Development and experimentation

**Use Streaming Mode When:**
- Dataset is very large (> 1GB text)
- Memory is limited
- One-pass training is acceptable
- Working with datasets that don't fit in RAM

**Streaming Performance Notes:**
- Requires `--workers 0` (single-threaded data loading)
- Best with pre-built vocabulary for large datasets
- Use larger `--buffer-size` for better shuffling
- Consider `--vocab-sample-size` for vocabulary quality vs. speed tradeoff

### Model-Specific Recommendations

**Skip-gram vs CBOW:**
- **Skip-gram**: Better for infrequent words, works well with small datasets
- **CBOW**: Faster training, better for frequent words, good for larger datasets

**Output Layer Types:**
- **Full Softmax**: Standard approach, O(V) complexity per prediction
  - Best for small-to-medium vocabularies (< 50,000 words)
  - Exact probability computation for all words
  - Use: `--output-layer full_softmax` (default)

- **Hierarchical Softmax**: Efficient tree-based approach, O(log V) complexity
  - Ideal for large vocabularies (> 50,000 words)
  - Uses Huffman tree based on word frequencies
  - Faster training and inference for large vocabularies
  - Use: `--output-layer hierarchical_softmax`

**Dynamic Window:**
- Primarily designed for and recommended with **Skip-gram** models
- Randomly shortens context window to emphasize nearby words
- Less conventional with CBOW (which averages context vectors)
- Can be used with CBOW as a form of regularization, but not standard practice

**Subsampling:**
- Beneficial for both Skip-gram and CBOW models
- Reduces frequency of common words like "the", "a", "is"
- Recommended threshold: `1e-5` for most corpora

The training script outputs JSON statistics including loss and throughput metrics for performance monitoring.

## Output Files

When using `--save`, the following files are written to `--out-dir`:

- `embeddings.npy` — NumPy matrix of input embeddings (vocab_size × embedding_dim)
- `vocab.json` — Word-to-index and index-to-word mappings
- `config.json` — Complete data and training configurations used

### Working with Saved Embeddings

**Load embeddings in Python:**
```python
from modern_word2vec.utils import load_vocab_and_embeddings
import numpy as np

# Load all components
idx_to_word, word_to_idx, embeddings = load_vocab_and_embeddings("runs/latest")

# Use embeddings
word_vector = embeddings[word_to_idx["king"]]
print(f"Embedding shape: {embeddings.shape}")
print(f"Vocabulary size: {len(word_to_idx)}")
```

**Convert to other formats:**
```python
import json
import numpy as np

# Save as Word2Vec text format
def save_word2vec_format(embeddings, vocab, filename):
    with open(filename, 'w') as f:
        f.write(f"{len(vocab)} {embeddings.shape[1]}\n")
        for word, idx in vocab.items():
            vector_str = ' '.join(map(str, embeddings[idx]))
            f.write(f"{word} {vector_str}\n")

# Usage
idx_to_word, word_to_idx, embeddings = load_vocab_and_embeddings("runs/latest")
save_word2vec_format(embeddings, word_to_idx, "embeddings.txt")
```

**Integration with other tools:**
```python
# For use with gensim
from gensim.models import KeyedVectors

def create_gensim_model(embeddings_path):
    idx_to_word, word_to_idx, embeddings = load_vocab_and_embeddings(embeddings_path)
    
    # Create gensim KeyedVectors
    kv = KeyedVectors(embeddings.shape[1])
    words = [idx_to_word[i] for i in range(len(idx_to_word))]
    kv.add_vectors(words, embeddings)
    
    return kv

# Usage
kv = create_gensim_model("runs/latest")
similar_words = kv.most_similar("king", topn=10)
```

## Troubleshooting and Tips

### Common Issues and Solutions

**Memory Issues:**
```bash
# Reduce batch size and use streaming
word2vec-train --streaming --batch-size 128 --buffer-size 5000

# Use gradient accumulation for effective larger batches
word2vec-train --batch-size 64 --grad-clip 1.0  # Train with smaller batches
```

**Slow Training:**
```bash
# Enable all optimizations
word2vec-train --amp --compile --workers 4 --pin-memory --batch-size 1024

# Use SGD with linear LR decay (as per original paper)
word2vec-train --optimizer sgd --lr 2.5e-2  # Higher LR for SGD
```

**Vocabulary Issues:**
```bash
# Increase vocabulary size for large corpora
word2vec-train --vocab-size 200000

# Use enhanced tokenization for better preprocessing
word2vec-train --tokenizer enhanced --lower --subsample 1e-5
```

**Quality Issues:**
```bash
# Increase training time
word2vec-train --epochs 20 --window-size 5 --embedding-dim 300

# Use subsampling for frequent words
word2vec-train --subsample 1e-5 --dynamic-window
```

### Best Practices

1. **Start Small**: Begin with small datasets and default settings
2. **Monitor Progress**: Check the JSON output for loss and throughput
3. **Save Frequently**: Use `--save` to preserve your trained models
4. **Experiment**: Try different model types (skip-gram vs CBOW)
5. **Scale Gradually**: Increase vocab size and dimensions incrementally
6. **Use Streaming**: For datasets > 1GB, consider streaming mode
7. **Optimize Device**: Match settings to your hardware (GPU vs CPU)

## Notes

- **Full softmax**: Standard implementation that can be slow for very large vocabularies but provides exact probabilities
- **Hierarchical softmax**: Efficient O(log V) implementation using Huffman trees, ideal for large vocabularies (>50k words)
- **Future work**: Negative sampling will be added to complete the suite of classical word2vec training variants
- Tokenization is intentionally simple for reproducibility; consider more robust tokenizers for production use
- Future CLI improvements will include a dedicated `word2vec-build-vocab` command for better user experience with vocabulary building

## License

MIT License - see [LICENSE](LICENSE) for details.
