#!/bin/bash
# ═══════════════════════════════════════════════════════════
# FULL PIPELINE — 1.3B Resonance Network chatbot
# 4x RTX 5090 + network volume, $72 budget (~20 hours)
# ═══════════════════════════════════════════════════════════
#
# Step 1: Create network volume (250GB)
# Step 2: Spin up 4x RTX 5090 pod with network volume
# Step 3: Run this script — it does everything
#
#   git clone https://github.com/sunnytroo01/resonance-network.git
#   cd resonance-network
#   bash scripts/train_4x5090.sh

set -e

DATA_DIR=${DATA_DIR:-/workspace/data}
CKPT_DIR=${CKPT_DIR:-/workspace/checkpoints}

pip install -q -r requirements.txt

echo "═══════════════════════════════════════════════════════"
echo "  RESONANCE NETWORK 1.3B — 4x RTX 5090 + NET VOLUME"
echo "  ~18 hrs pretrain + ~2 hrs SFT = chatbot"
echo "  Checkpoints safe on network volume"
echo "═══════════════════════════════════════════════════════"
echo ""

# ─── Data prep (if not already done) ────────────────────────
if [ ! -f "$DATA_DIR/pretrain/manifest.json" ]; then
    echo "No pre-training data found. Preparing data first..."
    echo "(This runs on GPU pod — not ideal but works)"
    echo ""
    python prepare_data.py --output $DATA_DIR
    echo ""
fi

if [ ! -f "$DATA_DIR/sft/index.json" ]; then
    echo "No SFT data found. Preparing SFT data..."
    echo ""
    python prepare_data.py --output $DATA_DIR --phase sft
    echo ""
fi

echo "Data: $(cat $DATA_DIR/pretrain/manifest.json | python3 -c 'import sys,json; m=json.load(sys.stdin); print(f"{m[\"total_tokens\"]/1e9:.1f}B pretrain tokens, {m[\"num_shards\"]} shards")')"
echo ""

# ─── Phase 1: Pre-training (~18 hours) ──────────────────────
echo "PHASE 1: PRE-TRAINING (12,000 steps, ~18 hours)"
echo "  Checkpoints every 200 steps on network volume"
echo ""

torchrun --nproc_per_node=4 \
    train.py \
    --config configs/budget_1b_4x5090.yaml \
    --data-dir $DATA_DIR/pretrain \
    --output-dir $CKPT_DIR/pretrain \
    --save-every 200 \
    --keep-checkpoints 3 \
    --permanent-save-every 1000 \
    --wandb --wandb-project resonance-1b-4x5090

echo ""
echo "Pre-training done!"
echo ""

# ─── Phase 2: SFT Chat Fine-tuning (~2 hours) ───────────────
echo "PHASE 2: SFT CHAT FINE-TUNING (1,500 steps, ~2 hours)"
echo ""

torchrun --nproc_per_node=4 \
    train.py \
    --config configs/budget_1b_4x5090_sft.yaml \
    --data-dir $DATA_DIR/sft \
    --output-dir $CKPT_DIR/sft \
    --stage sft \
    --resume $CKPT_DIR/pretrain/latest \
    --save-every 50 \
    --keep-checkpoints 3 \
    --wandb --wandb-project resonance-1b-4x5090-sft

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  DONE! Chatbot ready."
echo "  Model saved on network volume: $CKPT_DIR/sft/latest"
echo ""
echo "  Test it:"
echo "    python chat.py --checkpoint $CKPT_DIR/sft/inference_model.pt"
echo ""
echo "  Your model is SAFE on network volume."
echo "  You can stop this pod and start a cheaper one to test."
echo "═══════════════════════════════════════════════════════"
