# word2vec-train \
#     --synthetic-sentences 1000 \
#     --model-type skipgram \
#     --output-layer full_softmax \
#     --vocab-size 4096 \
#     --embedding-dim 16 \
#     --window-size 2 \
#     --batch-size 128 \
#     --epochs 3 \
#     --lr 2e-3 \
#     --optimizer adam \
#     --save \
#     --out-dir runs/skipgram-2-16D-4kV-full_softmax-synthetic \
#     --tensorboard \
#     --tensorboard-dir runs/skipgram-2-16D-4kV-full_softmax-synthetic/tensorboard \
#     --log-gradients \
#     --log-system-stats \
#     --log-interval 1

# word2vec-train \
#     --synthetic-sentences 1000 \
#     --model-type skipgram \
#     --output-layer negative_sampling \
#     --num-negative 5 \
#     --vocab-size 4096 \
#     --embedding-dim 16 \
#     --window-size 2 \
#     --batch-size 128 \
#     --epochs 3 \
#     --lr 2e-3 \
#     --optimizer adam \
#     --save \
#     --out-dir runs/skipgram-2-16D-4kV-negative_sampling-synthetic \
#     --tensorboard \
#     --tensorboard-dir runs/skipgram-2-16D-4kV-negative_sampling-synthetic/tensorboard \
#     --log-gradients \
#     --log-system-stats \
#     --log-interval 1
    
# word2vec-train \
#     --synthetic-sentences 1000 \
#     --model-type skipgram \
#     --output-layer hierarchical_softmax \
#     --vocab-size 4096 \
#     --embedding-dim 16 \
#     --window-size 2 \
#     --batch-size 128 \
#     --epochs 3 \
#     --lr 2e-3 \
#     --optimizer adam \
#     --save \
#     --out-dir runs/skipgram-2-16D-4kV-hierarchical_softmax-synthetic \
#     --tensorboard \
#     --tensorboard-dir runs/skipgram-2-16D-4kV-hierarchical_softmax-synthetic/tensorboard \
#     --log-gradients \
#     --log-system-stats \
#     --log-interval 1

# word2vec-train \
#     --synthetic-sentences 1000 \
#     --model-type skipgram \
#     --output-layer full_softmax \
#     --vocab-size 4096 \
#     --embedding-dim 16 \
#     --window-size 2 \
#     --dynamic-window \
#     --batch-size 128 \
#     --epochs 3 \
#     --lr 2e-3 \
#     --optimizer adam \
#     --save \
#     --out-dir runs/skipgram-dynamic-2-16D-4kV-full_softmax-synthetic \
#     --tensorboard \
#     --tensorboard-dir runs/skipgram-dynamic-2-16D-4kV-full_softmax-synthetic/tensorboard \
#     --log-gradients \
#     --log-system-stats \
#     --log-interval 1

# word2vec-train \
#     --synthetic-sentences 1000 \
#     --model-type skipgram \
#     --output-layer hierarchical_softmax \
#     --vocab-size 4096 \
#     --embedding-dim 16 \
#     --window-size 2 \
#     --dynamic-window \
#     --batch-size 128 \
#     --epochs 3 \
#     --lr 2e-3 \
#     --optimizer adam \
#     --save \
#     --out-dir runs/skipgram-dynamic-2-16D-4kV-hierarchical_softmax-synthetic \
#     --tensorboard \
#     --tensorboard-dir runs/skipgram-dynamic-2-16D-4kV-hierarchical_softmax-synthetic/tensorboard \
#     --log-gradients \
#     --log-system-stats \
#     --log-interval 1

word2vec-train \
    --synthetic-sentences 1000 \
    --model-type skipgram \
    --output-layer full_softmax \
    --vocab-size 4096 \
    --embedding-dim 16 \
    --window-size 2 \
    --subsample 1e-5 \
    --dynamic-window \
    --batch-size 128 \
    --epochs 3 \
    --lr 2e-3 \
    --optimizer adam \
    --save \
    --out-dir runs/skipgram-dynamic-2-16D-4kV-full_softmax-synthetic

# Skipgram + Negative Sampling (efficient, high quality)
# word2vec-train \
#     --synthetic-sentences 1000 \
#     --model-type skipgram \
#     --output-layer negative_sampling \
#     --num-negative 10 \
#     --vocab-size 4096 \
#     --embedding-dim 16 \
#     --window-size 2 \
#     --dynamic-window \
#     --batch-size 128 \
#     --epochs 3 \
#     --lr 2e-3 \
#     --optimizer adam \
#     --save \
#     --out-dir runs/skipgram-dynamic-2-16D-4kV-negative_sampling-synthetic-subsample_1e-5 \
    --tensorboard \
    --tensorboard-dir runs/skipgram-dynamic-2-16D-4kV-full_softmax-synthetic-subsample_1e-5/tensorboard \
    --log-gradients \
    --log-system-stats \
    --log-interval 1

# word2vec-train \
#     --synthetic-sentences 1000 \
#     --model-type skipgram \
#     --output-layer hierarchical_softmax \
#     --vocab-size 4096 \
#     --embedding-dim 16 \
#     --window-size 2 \
#     --subsample 1e-5 \
#     --dynamic-window \
#     --batch-size 128 \
#     --epochs 3 \
#     --lr 2e-3 \
#     --optimizer adam \
#     --save \
#     --out-dir runs/skipgram-dynamic-2-16D-4kV-hierarchical_softmax-synthetic-subsample_1e-5 \
#     --tensorboard \
#     --tensorboard-dir runs/skipgram-dynamic-2-16D-4kV-hierarchical_softmax-synthetic-subsample_1e-5/tensorboard \
#     --log-gradients \
#     --log-system-stats \
#     --log-interval 1