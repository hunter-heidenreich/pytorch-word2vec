NUM_SENTENCES=10000
VOCAB_SIZE=512
EMBEDDING_DIM=16
WINDOW_SIZE=4
BATCH_SIZE=256
NUM_EPOCHS=3
LEARNING_RATE=0.002

word2vec-train \
    --synthetic-sentences $NUM_SENTENCES \
    --vocab-size $VOCAB_SIZE \
    --embedding-dim $EMBEDDING_DIM \
    --window-size $WINDOW_SIZE \
    --batch-size $BATCH_SIZE \
    --epochs $NUM_EPOCHS \
    --lr $LEARNING_RATE \
    --save \
    --model-type cbow \
    --output-layer full_softmax \
    --out-dir "runs/cbow-${WINDOW_SIZE}-${EMBEDDING_DIM}D-${VOCAB_SIZE}V-full_softmax-synthetic" \
    --tensorboard \
    --tensorboard-dir "runs/cbow-${WINDOW_SIZE}-${EMBEDDING_DIM}D-${VOCAB_SIZE}V-full_softmax-synthetic/tensorboard" \
    --log-gradients \
    --log-system-stats \
    --log-interval 1
    
word2vec-train \
    --synthetic-sentences $NUM_SENTENCES \
    --vocab-size $VOCAB_SIZE \
    --embedding-dim $EMBEDDING_DIM \
    --window-size $WINDOW_SIZE \
    --batch-size $BATCH_SIZE \
    --epochs $NUM_EPOCHS \
    --lr $LEARNING_RATE \
    --save \
    --model-type cbow \
    --output-layer hierarchical_softmax \
    --out-dir "runs/cbow-${WINDOW_SIZE}-${EMBEDDING_DIM}D-${VOCAB_SIZE}V-hierarchical_softmax-synthetic" \
    --tensorboard \
    --tensorboard-dir "runs/cbow-${WINDOW_SIZE}-${EMBEDDING_DIM}D-${VOCAB_SIZE}V-hierarchical_softmax-synthetic/tensorboard" \
    --log-gradients \
    --log-system-stats \
    --log-interval 1

word2vec-train \
    --synthetic-sentences $NUM_SENTENCES \
    --vocab-size $VOCAB_SIZE \
    --embedding-dim $EMBEDDING_DIM \
    --window-size $WINDOW_SIZE \
    --batch-size $BATCH_SIZE \
    --epochs $NUM_EPOCHS \
    --lr $LEARNING_RATE \
    --save \
    --model-type cbow \
    --output-layer full_softmax \
    --dynamic-window \
    --out-dir "runs/cbow-dynamic-${WINDOW_SIZE}-${EMBEDDING_DIM}D-${VOCAB_SIZE}V-full_softmax-synthetic" \
    --tensorboard \
    --tensorboard-dir "runs/cbow-dynamic-${WINDOW_SIZE}-${EMBEDDING_DIM}D-${VOCAB_SIZE}V-full_softmax-synthetic/tensorboard" \
    --log-gradients \
    --log-system-stats \
    --log-interval 1

word2vec-train \
    --synthetic-sentences $NUM_SENTENCES \
    --vocab-size $VOCAB_SIZE \
    --embedding-dim $EMBEDDING_DIM \
    --window-size $WINDOW_SIZE \
    --batch-size $BATCH_SIZE \
    --epochs $NUM_EPOCHS \
    --lr $LEARNING_RATE \
    --save \
    --model-type cbow \
    --output-layer hierarchical_softmax \
    --dynamic-window \
    --out-dir runs/cbow-dynamic-${WINDOW_SIZE}-${EMBEDDING_DIM}D-${VOCAB_SIZE}V-hierarchical_softmax-synthetic \
    --tensorboard \
    --tensorboard-dir runs/cbow-dynamic-${WINDOW_SIZE}-${EMBEDDING_DIM}D-${VOCAB_SIZE}V-hierarchical_softmax-synthetic/tensorboard \
    --log-gradients \
    --log-system-stats \
    --log-interval 1

word2vec-train \
    --synthetic-sentences $NUM_SENTENCES \
    --vocab-size $VOCAB_SIZE \
    --embedding-dim $EMBEDDING_DIM \
    --window-size $WINDOW_SIZE \
    --batch-size $BATCH_SIZE \
    --epochs $NUM_EPOCHS \
    --lr $LEARNING_RATE \
    --save \
    --model-type cbow \
    --output-layer full_softmax \
    --subsample 1e-5 \
    --dynamic-window \
    --out-dir runs/cbow-dynamic-${WINDOW_SIZE}-${EMBEDDING_DIM}D-${VOCAB_SIZE}V-full_softmax-synthetic-subsample_1e-5 \
    --tensorboard \
    --tensorboard-dir runs/cbow-dynamic-${WINDOW_SIZE}-${EMBEDDING_DIM}D-${VOCAB_SIZE}V-full_softmax-synthetic-subsample_1e-5/tensorboard \
    --log-gradients \
    --log-system-stats \
    --log-interval 1

word2vec-train \
    --synthetic-sentences $NUM_SENTENCES \
    --vocab-size $VOCAB_SIZE \
    --embedding-dim $EMBEDDING_DIM \
    --window-size $WINDOW_SIZE \
    --batch-size $BATCH_SIZE \
    --epochs $NUM_EPOCHS \
    --lr $LEARNING_RATE \
    --save \
    --model-type cbow \
    --output-layer hierarchical_softmax \
    --subsample 1e-5 \
    --dynamic-window \
    --out-dir runs/cbow-dynamic-${WINDOW_SIZE}-${EMBEDDING_DIM}D-${VOCAB_SIZE}V-hierarchical_softmax-synthetic-subsample_1e-5 \
    --tensorboard \
    --tensorboard-dir runs/cbow-dynamic-${WINDOW_SIZE}-${EMBEDDING_DIM}D-${VOCAB_SIZE}V-hierarchical_softmax-synthetic-subsample_1e-5/tensorboard \
    --log-gradients \
    --log-system-stats \
    --log-interval 1