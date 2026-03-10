#!/bin/bash
# Train 1T model on multi-node B200 cluster
# Adjust --nnodes and --node_rank per machine
torchrun \
    --nnodes=64 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --config configs/titan_1t.yaml \
    --output-dir checkpoints/1t \
   
