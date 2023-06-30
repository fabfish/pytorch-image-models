#!/bin/bash
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 python train.py --model deit_small_patch16_224 --batch-size 128 --epochs 16 --warmup-epochs 1 --data-dir /home/fish/Documents/imagenet --output ./output --log-wandb True --experiment timmtest --use-eva