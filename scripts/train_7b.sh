#!/bin/bash
# Train 7B model on 8 B200 GPUs
torchrun --nproc_per_node=8 \
    train.py \
    --config configs/xl_7b.yaml \
    --output-dir checkpoints/7b \
   
