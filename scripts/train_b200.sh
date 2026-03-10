#!/bin/bash
set -e
# ═══════════════════════════════════════════════════════════
#  RESONANCE NETWORK 1.3B — 1x NVIDIA B200
#  Single GPU training (no torchrun needed)
#  Assumes data is on the network volume at /workspace/data
# ═══════════════════════════════════════════════════════════
DATA_DIR=${DATA_DIR:-/workspace/data}
CKPT_DIR=${CKPT_DIR:-/workspace/checkpoints}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/resonance-network || { echo "ERROR: repo not found. Run: git clone https://github.com/sunnytroo01/resonance-network.git /workspace/resonance-network"; exit 1; }

pip install -q -r requirements.txt

# Verify pretrain data exists
if [ ! -f "$DATA_DIR/pretrain/manifest.json" ]; then
    echo "ERROR: No pretrain data found at $DATA_DIR/pretrain/"
    echo "Run 'bash scripts/prepare_only.sh' on a cheap pod first."
    exit 1
fi

# Prepare SFT data if missing
if [ ! -f "$DATA_DIR/sft/index.json" ]; then
    echo "No SFT data found — preparing it now (takes ~10 min)..."
    python prepare_data.py --output $DATA_DIR --phase sft
fi

echo "═══════════════════════════════════════════════════════"
echo "  RESONANCE NETWORK 1.3B — 1x B200"
echo "  Data: $DATA_DIR"
echo "  Checkpoints: $CKPT_DIR"
echo "═══════════════════════════════════════════════════════"

# Phase 1: Pretrain (single GPU — no torchrun)
echo "PHASE 1: PRE-TRAINING"
python train.py \
    --config configs/fast_1b_b200.yaml \
    --data-dir $DATA_DIR/pretrain \
    --output-dir $CKPT_DIR/pretrain \
    --save-every 200 --keep-checkpoints 3 --permanent-save-every 2000

# Phase 2: SFT
echo "PHASE 2: SFT CHAT"
python train.py \
    --config configs/fast_1b_b200_sft.yaml \
    --data-dir $DATA_DIR/sft \
    --output-dir $CKPT_DIR/sft \
    --stage sft \
    --resume $CKPT_DIR/pretrain/latest \
    --save-every 100 --keep-checkpoints 3

echo "═══════════════════════════════════════════════════════"
echo "  TRAINING COMPLETE!"
echo "  Test your chatbot:"
echo "    python chat.py --checkpoint $CKPT_DIR/sft/inference_model.pt"
echo "═══════════════════════════════════════════════════════"
