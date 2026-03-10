#!/bin/bash
# ═══════════════════════════════════════════════════════════
# ONE COMMAND — Collect and tokenize ALL training data
# Run this on your data-prep pod (5090s) with network volume
# ═══════════════════════════════════════════════════════════
#
# Usage:
#   bash scripts/prepare_data.sh
#
# Output goes to /workspace/data/ (network volume)
# This takes a while — SlimPajama alone is 627B tokens
# Script is resumable: re-run if interrupted

set -e

pip install -q tiktoken datasets tqdm

echo "Starting full data preparation..."
echo "Output: /workspace/data/"
echo ""

python prepare_data.py --output /workspace/data

echo ""
echo "Done! Data is ready on network volume at /workspace/data/"
echo "  /workspace/data/pretrain/  — pre-training shards"
echo "  /workspace/data/sft/       — chat fine-tuning data"
echo ""
echo "Now launch training on B200s:"
echo "  bash scripts/train_1t_chatbot.sh"
