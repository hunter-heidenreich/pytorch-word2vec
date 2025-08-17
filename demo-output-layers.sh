#!/bin/bash

# Demonstration of all three Word2Vec output layer types
# This script shows the usage of full softmax, hierarchical softmax, and negative sampling

echo "=== Word2Vec Output Layer Comparison ==="
echo "Training small models with synthetic data to demonstrate all three approaches"
echo

# Common parameters
SENTENCES=100
VOCAB=50
EPOCHS=2
EMBEDDING_DIM=32
BATCH_SIZE=64

# 1. Full Softmax (baseline)
echo "1. Training with Full Softmax (baseline approach)..."
uv run word2vec-train \
    --synthetic-sentences $SENTENCES \
    --synthetic-vocab $VOCAB \
    --model-type skipgram \
    --output-layer full_softmax \
    --epochs $EPOCHS \
    --embedding-dim $EMBEDDING_DIM \
    --batch-size $BATCH_SIZE \
    --save \
    --out-dir runs/demo-full-softmax

echo

# 2. Hierarchical Softmax (efficient for large vocabularies)
echo "2. Training with Hierarchical Softmax (O(log V) complexity)..."
uv run word2vec-train \
    --synthetic-sentences $SENTENCES \
    --synthetic-vocab $VOCAB \
    --model-type skipgram \
    --output-layer hierarchical_softmax \
    --epochs $EPOCHS \
    --embedding-dim $EMBEDDING_DIM \
    --batch-size $BATCH_SIZE \
    --save \
    --out-dir runs/demo-hierarchical-softmax

echo

# 3. Negative Sampling (high quality, efficient)
echo "3. Training with Negative Sampling (O(k) complexity, high quality)..."
uv run word2vec-train \
    --synthetic-sentences $SENTENCES \
    --synthetic-vocab $VOCAB \
    --model-type skipgram \
    --output-layer negative_sampling \
    --num-negative 10 \
    --epochs $EPOCHS \
    --embedding-dim $EMBEDDING_DIM \
    --batch-size $BATCH_SIZE \
    --save \
    --out-dir runs/demo-negative-sampling

echo
echo "=== Training Complete ==="
echo "You can now query the models using:"
echo "  uv run word2vec-query --run-dir runs/demo-full-softmax --word <word> --topn 5"
echo "  uv run word2vec-query --run-dir runs/demo-hierarchical-softmax --word <word> --topn 5"
echo "  uv run word2vec-query --run-dir runs/demo-negative-sampling --word <word> --topn 5"
echo
echo "Output layer comparison:"
echo "  • Full Softmax:        O(V) complexity, exact probabilities"
echo "  • Hierarchical Softmax: O(log V) complexity, fast for large vocabularies"
echo "  • Negative Sampling:    O(k) complexity, excellent quality with k=5-20"
