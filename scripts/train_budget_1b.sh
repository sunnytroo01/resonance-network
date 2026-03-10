#!/bin/bash
# ═══════════════════════════════════════════════════════════
# FULL PIPELINE — 1.3B Resonance Network chatbot
# Optimized for: 4x B200, $72 RunPod budget (~3.6 hours)
# ═══════════════════════════════════════════════════════════
#
# BEFORE RUNNING: data must be on network volume
# On a separate pod (5090s):
#   git clone https://github.com/sunnytroo01/resonance-network.git
#   cd resonance-network
#   bash scripts/prepare_data.sh
#
# THEN on 4x B200 pod:
#   git clone https://github.com/sunnytroo01/resonance-network.git
#   cd resonance-network
#   bash scripts/train_budget_1b.sh

set -e

DATA_DIR=${DATA_DIR:-/workspace/data}
CKPT_DIR=${CKPT_DIR:-/workspace/checkpoints}

pip install -q -r requirements.txt

echo "═══════════════════════════════════════════════════════"
echo "  RESONANCE NETWORK 1.3B — $72 BUDGET RUN"
echo "  4x B200, ~3 hrs pretrain + ~30 min SFT"
echo "═══════════════════════════════════════════════════════"
echo ""

# Verify data exists
if [ ! -f "$DATA_DIR/pretrain/manifest.json" ]; then
    echo "ERROR: No pre-training data found at $DATA_DIR/pretrain/"
    echo "Run prepare_data.sh on your data-prep pod first."
    exit 1
fi

if [ ! -f "$DATA_DIR/sft/index.json" ]; then
    echo "ERROR: No SFT data found at $DATA_DIR/sft/"
    echo "Run prepare_data.sh on your data-prep pod first."
    exit 1
fi

echo "Data OK: $(cat $DATA_DIR/pretrain/manifest.json | python3 -c 'import sys,json; m=json.load(sys.stdin); print(f"{m[\"total_tokens\"]/1e9:.1f}B pretrain tokens, {m[\"num_shards\"]} shards")')"
echo ""

# ─── Phase 1: Pre-training (~3 hours) ───────────────────────
echo "PHASE 1: PRE-TRAINING (4,000 steps, ~3 hours)"
echo ""

torchrun --nproc_per_node=4 \
    train.py \
    --config configs/budget_1b.yaml \
    --data-dir $DATA_DIR/pretrain \
    --output-dir $CKPT_DIR/pretrain \
    --save-every 200 \
    --keep-checkpoints 3 \
    --permanent-save-every 1000 \
   

echo ""
echo "Pre-training done!"
echo ""

# ─── Phase 2: SFT Chat Fine-tuning (~30 min) ────────────────
echo "PHASE 2: SFT CHAT FINE-TUNING (500 steps, ~30 min)"
echo ""

torchrun --nproc_per_node=4 \
    train.py \
    --config configs/budget_1b_sft.yaml \
    --data-dir $DATA_DIR/sft \
    --output-dir $CKPT_DIR/sft \
    --stage sft \
    --resume $CKPT_DIR/pretrain/latest \
    --save-every 50 \
    --keep-checkpoints 3 \
   

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  DONE! Chatbot ready."
echo "  Model: $CKPT_DIR/sft/latest"
echo ""
echo "  Test it:"
echo "    python chat.py --checkpoint $CKPT_DIR/sft/inference_model.pt"
echo "═══════════════════════════════════════════════════════"
