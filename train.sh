#!/bin/bash

# Set CUDA environment variables
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.9"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Run training script with optimized parameters
python main.py \
    --valid_dir data/train/valid \
    --invalid_dir data/train/invalid \
    --test_dir_valid data/test/valid \
    --test_dir_invalid data/test/invalid \
    --epochs 100 \
    --episodes_per_epoch 100 \
    --n_shot 5 \
    --n_query 15 \
    --learning_rate 0.001 \
    --batch_size 64 \
    --num_workers 28 \
    --hidden_size 8 \
    --dropout 0.1 \
    --device cuda \
    --seed 42 \
    --mixed_precision fp16 \
    --prefetch_factor 2 \
    --optimize_memory \
    --save_interval 5 \
    2>&1 | tee "training_log_$(date +%Y%m%d_%H%M%S).log"