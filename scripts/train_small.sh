#!/bin/bash
# Train 125M model on single GPU
python train.py --config configs/small_125m.yaml --output-dir checkpoints/small --wandb
