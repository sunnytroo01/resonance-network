#!/bin/bash
# ═══════════════════════════════════════════════════════════
# ONE COMMAND — Train 1T Resonance Network chatbot
# Pre-training → SFT, with very frequent checkpoints
#
# Run on B200 cluster with network volume at /workspace/
# Data must be prepared first: bash scripts/prepare_data.sh
# ═══════════════════════════════════════════════════════════
#
# Usage (on each node):
#   export MASTER_ADDR=<master-ip>
#   export MASTER_PORT=29500
#   export NNODES=64
#   export NODE_RANK=<this-node-rank>
#   bash scripts/train_1t_chatbot.sh
#
# Or with RunPod multi-node, just set NNODES and let
# RunPod handle MASTER_ADDR/PORT/NODE_RANK

set -e

NNODES=${NNODES:-64}
NPROC=${NPROC:-8}
DATA_DIR=${DATA_DIR:-/workspace/data}
CKPT_DIR=${CKPT_DIR:-/workspace/checkpoints}

pip install -q tiktoken datasets wandb pyyaml

echo "═══════════════════════════════════════════════════════"
echo "  RESONANCE NETWORK 1T — FULL TRAINING PIPELINE"
echo "═══════════════════════════════════════════════════════"
echo "  Nodes: $NNODES x $NPROC GPUs = $(($NNODES * $NPROC)) total"
echo "  Data:  $DATA_DIR"
echo "  Ckpts: $CKPT_DIR"
echo "═══════════════════════════════════════════════════════"

# ─── Phase 1: Pre-training ──────────────────────────────────
echo ""
echo "PHASE 1: PRE-TRAINING on SlimPajama + OpenWebText + Code + Wiki"
echo ""

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC \
    --node_rank=${NODE_RANK:-0} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR:-localhost}:${MASTER_PORT:-29500} \
    train.py \
    --config configs/titan_1t.yaml \
    --data-dir $DATA_DIR/pretrain \
    --output-dir $CKPT_DIR/pretrain \
    --stage pretrain \
    --save-every 250 \
    --keep-checkpoints 3 \
    --permanent-save-every 2000 \
    --wandb --wandb-project resonance-1t-pretrain

echo ""
echo "Pre-training complete!"
echo ""

# ─── Phase 2: SFT (Chat Fine-tuning) ────────────────────────
echo "PHASE 2: SFT — Chat fine-tuning"
echo ""

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC \
    --node_rank=${NODE_RANK:-0} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR:-localhost}:${MASTER_PORT:-29500} \
    train.py \
    --config configs/titan_1t_sft.yaml \
    --data-dir $DATA_DIR/sft \
    --output-dir $CKPT_DIR/sft \
    --stage sft \
    --resume $CKPT_DIR/pretrain/latest \
    --save-every 100 \
    --keep-checkpoints 3 \
    --permanent-save-every 1000 \
    --wandb --wandb-project resonance-1t-sft

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  TRAINING COMPLETE"
echo "  Chat model: $CKPT_DIR/sft/latest"
echo "  Run: python chat.py --checkpoint $CKPT_DIR/sft/latest"
echo "═══════════════════════════════════════════════════════"
