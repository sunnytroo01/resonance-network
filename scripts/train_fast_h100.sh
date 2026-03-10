#!/bin/bash
# ═══════════════════════════════════════════════════════════
# SPEED RUN — 1.3B Resonance Network chatbot
# 4x H100 80GB + network volume — done in ~5 hours
# ═══════════════════════════════════════════════════════════
#
#   git clone https://github.com/sunnytroo01/resonance-network.git
#   cd resonance-network
#   bash scripts/train_fast_h100.sh

set -e

DATA_DIR=${DATA_DIR:-/workspace/data}
CKPT_DIR=${CKPT_DIR:-/workspace/checkpoints}

pip install -q -r requirements.txt

echo "═══════════════════════════════════════════════════════"
echo "  RESONANCE NETWORK 1.3B — SPEED RUN"
echo "  4x H100 80GB — ~4 hrs pretrain + ~1 hr SFT"
echo "═══════════════════════════════════════════════════════"
echo ""

# ─── Data prep (if not already on network volume) ───────────
if [ ! -f "$DATA_DIR/pretrain/manifest.json" ]; then
    echo "Preparing data..."
    python prepare_data.py --output $DATA_DIR
    echo ""
fi

if [ ! -f "$DATA_DIR/sft/index.json" ]; then
    echo "Preparing SFT data..."
    python prepare_data.py --output $DATA_DIR --phase sft
    echo ""
fi

echo "Data: $(cat $DATA_DIR/pretrain/manifest.json | python3 -c 'import sys,json; m=json.load(sys.stdin); print(f"{m[\"total_tokens\"]/1e9:.1f}B tokens, {m[\"num_shards\"]} shards")')"
echo ""

# ─── Phase 1: Pre-training (~4 hours) ───────────────────────
echo "PHASE 1: PRE-TRAINING (3,500 steps, ~4 hours)"
echo ""

torchrun --nproc_per_node=4 \
    train.py \
    --config configs/fast_1b_h100.yaml \
    --data-dir $DATA_DIR/pretrain \
    --output-dir $CKPT_DIR/pretrain \
    --save-every 200 \
    --keep-checkpoints 3 \
    --permanent-save-every 500 \
    --wandb --wandb-project resonance-1b-fast

echo ""
echo "Pre-training done!"
echo ""

# ─── Phase 2: SFT (~1 hour) ─────────────────────────────────
echo "PHASE 2: SFT CHAT (800 steps, ~1 hour)"
echo ""

torchrun --nproc_per_node=4 \
    train.py \
    --config configs/fast_1b_h100_sft.yaml \
    --data-dir $DATA_DIR/sft \
    --output-dir $CKPT_DIR/sft \
    --stage sft \
    --resume $CKPT_DIR/pretrain/latest \
    --save-every 50 \
    --keep-checkpoints 3 \
    --wandb --wandb-project resonance-1b-fast-sft

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  DONE in ~5 hours. Chatbot ready."
echo "  python chat.py --checkpoint $CKPT_DIR/sft/latest"
echo "═══════════════════════════════════════════════════════"
