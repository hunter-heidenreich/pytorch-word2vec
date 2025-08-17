word2vec-train \
    --dataset wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --split "train" \
    --model-type skipgram \
    --output-layer full_softmax \
    --vocab-size 0 \
    --embedding-dim 64 \
    --window-size 5 \
    --batch-size 256 \
    --epochs 1 \
    --lr 2e-3 \
    --optimizer adam \
    --save \
    --out-dir runs/wikitext-2-skipgram-2-64D-V-full_softmax \
    --tensorboard \
    --tensorboard-dir runs/wikitext-2-skipgram-2-64D-V-full_softmax/tensorboard \
    --log-gradients \
    --log-system-stats \
    --log-interval 1

    