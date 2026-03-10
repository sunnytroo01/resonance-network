#!/bin/bash
# ═══════════════════════════════════════════════════════════
# FULL PIPELINE — 1.3B Resonance Network chatbot
# Optimized for: 8x RTX 5090, $72 RunPod budget (~10 hours)
# ═══════════════════════════════════════════════════════════
#
# BEFORE RUNNING: data must be on network volume
# On a separate pod:
#   git clone https://github.com/sunnytroo01/resonance-network.git
#   cd resonance-network
#   bash scripts/prepare_data.sh
#
# THEN on 8x RTX 5090 pod:
#   git clone https://github.com/sunnytroo01/resonance-network.git
#   cd resonance-network
#   bash scripts/train_budget_5090.sh

set -e

DATA_DIR=${DATA_DIR:-/workspace/data}
CKPT_DIR=${CKPT_DIR:-/workspace/checkpoints}

pip install -q -r requirements.txt

echo "═══════════════════════════════════════════════════════"
echo "  RESONANCE NETWORK 1.3B — 8x RTX 5090"
echo "  ~9 hrs pretrain + ~1 hr SFT = chatbot"
echo "═══════════════════════════════════════════════════════"
echo ""

# Verify data exists
if [ ! -f "$DATA_DIR/pretrain/manifest.json" ]; then
    echo "ERROR: No pre-training data at $DATA_DIR/pretrain/"
    echo "Run prepare_data.sh on your data-prep pod first."
    exit 1
fi

if [ ! -f "$DATA_DIR/sft/index.json" ]; then
    echo "ERROR: No SFT data at $DATA_DIR/sft/"
    echo "Run prepare_data.sh on your data-prep pod first."
    exit 1
fi

echo "Data OK: $(cat $DATA_DIR/pretrain/manifest.json | python3 -c 'import sys,json; m=json.load(sys.stdin); print(f"{m[\"total_tokens\"]/1e9:.1f}B pretrain tokens, {m[\"num_shards\"]} shards")')"
echo ""

# ─── Phase 1: Pre-training (~9 hours) ───────────────────────
echo "PHASE 1: PRE-TRAINING (7,000 steps, ~9 hours)"
echo ""

torchrun --nproc_per_node=8 \
    train.py \
    --config configs/budget_1b_5090.yaml \
    --data-dir $DATA_DIR/pretrain \
    --output-dir $CKPT_DIR/pretrain \
    --save-every 200 \
    --keep-checkpoints 3 \
    --permanent-save-every 1000 \
   

echo ""
echo "Pre-training done!"
echo ""

# ─── Phase 2: SFT Chat Fine-tuning (~1 hour) ────────────────
echo "PHASE 2: SFT CHAT FINE-TUNING (800 steps, ~1 hour)"
echo ""

torchrun --nproc_per_node=8 \
    train.py \
    --config configs/budget_1b_5090_sft.yaml \
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
