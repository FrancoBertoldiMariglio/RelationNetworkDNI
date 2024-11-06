#!/bin/bash

python main.py \
    --valid_dir data/train/valid \
    --invalid_dir data/train/invalid \
    --test_dir_valid data/test/valid \
    --test_dir_invalid data/test/invalid \
    --epochs 100 \
    --episodes_per_epoch 200 \
    --n_shot 5 \
    --n_query 15 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --num_workers 4 \
    --hidden_size 8 \
    --dropout 0.1 \
    --device cuda \
    --seed 42