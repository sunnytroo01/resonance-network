#!/bin/bash
set -e
# ═══════════════════════════════════════════════════════════
#  TRAINING ONLY — run on your GPU pod (4x RTX 4090)
#  Assumes data is already on the network volume
# ═══════════════════════════════════════════════════════════
DATA_DIR=${DATA_DIR:-/workspace/data}
CKPT_DIR=${CKPT_DIR:-/workspace/checkpoints}

pip install -q -r requirements.txt

# Verify data exists
if [ ! -f "$DATA_DIR/pretrain/manifest.json" ]; then
    echo "ERROR: No pretrain data found at $DATA_DIR/pretrain/"
    echo "Run 'bash scripts/prepare_only.sh' on a cheap pod first."
    exit 1
fi
if [ ! -f "$DATA_DIR/sft/index.json" ]; then
    echo "No SFT data found — preparing it now (takes ~10 min)..."
    python prepare_data.py --output $DATA_DIR --phase sft
fi

echo "═══════════════════════════════════════════════════════"
echo "  RESONANCE NETWORK 1.3B — TRAINING (4x RTX 4090)"
echo "  Data: $DATA_DIR"
echo "  Checkpoints: $CKPT_DIR"
echo "═══════════════════════════════════════════════════════"

# Phase 1: Pretrain
echo "PHASE 1: PRE-TRAINING"
torchrun --nproc_per_node=4 \
    train.py \
    --config configs/fast_1b_4090.yaml \
    --data-dir $DATA_DIR/pretrain \
    --output-dir $CKPT_DIR/pretrain \
    --save-every 200 --keep-checkpoints 3 --permanent-save-every 1000 \
   

# Phase 2: SFT
echo "PHASE 2: SFT CHAT"
torchrun --nproc_per_node=4 \
    train.py \
    --config configs/fast_1b_4090_sft.yaml \
    --data-dir $DATA_DIR/sft \
    --output-dir $CKPT_DIR/sft \
    --stage sft \
    --resume $CKPT_DIR/pretrain/latest \
    --save-every 50 --keep-checkpoints 3 \
   

echo "═══════════════════════════════════════════════════════"
echo "  TRAINING COMPLETE!"
echo "  Test your chatbot:"
echo "    python chat.py --checkpoint $CKPT_DIR/sft/inference_model.pt"
echo "═══════════════════════════════════════════════════════"
