#!/bin/bash
set -e
# ═══════════════════════════════════════════════════════════
#  RESONANCE NETWORK 1.3B — FULL PIPELINE
#  One command: data prep → pretrain → SFT
#
#  Everything saves to the network volume. If this script
#  crashes or gets stopped, just run it again — it skips
#  whatever is already done and picks up where it left off.
#
#  Usage (paste into any RunPod terminal):
#    bash /workspace/resonance-network/scripts/run.sh
# ═══════════════════════════════════════════════════════════
DATA_DIR=${DATA_DIR:-/workspace/data}
CKPT_DIR=${CKPT_DIR:-/workspace/checkpoints}
REPO_DIR=${REPO_DIR:-/workspace/resonance-network}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "═══════════════════════════════════════════════════════"
echo "  RESONANCE NETWORK — FULL PIPELINE"
echo "  Data:        $DATA_DIR"
echo "  Checkpoints: $CKPT_DIR"
echo "  Repo:        $REPO_DIR"
echo "═══════════════════════════════════════════════════════"

# ── Step 0: Get the code ──────────────────────────────────
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "Cloning repo..."
    git clone https://github.com/sunnytroo01/resonance-network.git "$REPO_DIR"
else
    echo "Pulling latest code..."
    cd "$REPO_DIR" && git pull
fi
cd "$REPO_DIR"
pip install -q -r requirements.txt

# ── Step 1: Prepare pretrain data ─────────────────────────
# Streams from HuggingFace, tokenizes into binary shards on the network volume.
# Fully resumable — if interrupted, re-running skips finished shards/sources.
if [ -f "$DATA_DIR/pretrain/manifest.json" ]; then
    echo ""
    echo "  [SKIP] Pretrain data already exists at $DATA_DIR/pretrain/"
    echo "         $(cat $DATA_DIR/pretrain/manifest.json)"
else
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STEP 1: PREPARING PRETRAIN DATA"
    echo "  This streams ~20B+ tokens from HuggingFace and"
    echo "  tokenizes them into binary shards."
    echo "  Takes a few hours on first run. Fully resumable."
    echo "═══════════════════════════════════════════════════════"
    python prepare_data.py --output "$DATA_DIR" --phase pretrain
fi

# ── Step 2: Prepare SFT data ─────────────────────────────
# Downloads chat/instruction datasets (OpenAssistant, Dolly, Alpaca, OpenOrca)
if [ -f "$DATA_DIR/sft/index.json" ]; then
    echo ""
    echo "  [SKIP] SFT data already exists at $DATA_DIR/sft/"
else
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STEP 2: PREPARING SFT DATA"
    echo "  Downloads chat datasets (~10 min)"
    echo "═══════════════════════════════════════════════════════"
    python prepare_data.py --output "$DATA_DIR" --phase sft
fi

# ── Step 3: Pretrain ──────────────────────────────────────
# Picks the right config based on available VRAM.
# Resumes from latest checkpoint if one exists.
PRETRAIN_DONE="$CKPT_DIR/pretrain/inference_model.pt"
if [ -f "$PRETRAIN_DONE" ]; then
    echo ""
    echo "  [SKIP] Pretraining already finished (inference_model.pt exists)"
else
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STEP 3: PRE-TRAINING (1.3B params)"
    echo "═══════════════════════════════════════════════════════"

    RESUME_FLAG=""
    if [ -f "$CKPT_DIR/pretrain/latest" ]; then
        echo "  Resuming from checkpoint: $(cat $CKPT_DIR/pretrain/latest)"
        RESUME_FLAG="--resume $CKPT_DIR/pretrain/latest"
    fi

    python train.py \
        --config configs/fast_1b_b200.yaml \
        --data-dir "$DATA_DIR/pretrain" \
        --output-dir "$CKPT_DIR/pretrain" \
        --save-every 200 --keep-checkpoints 3 --permanent-save-every 2000 \
        $RESUME_FLAG
fi

# ── Step 4: SFT ──────────────────────────────────────────
SFT_DONE="$CKPT_DIR/sft/inference_model.pt"
if [ -f "$SFT_DONE" ]; then
    echo ""
    echo "  [SKIP] SFT already finished (inference_model.pt exists)"
else
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STEP 4: SFT CHAT FINE-TUNING"
    echo "═══════════════════════════════════════════════════════"

    RESUME_FLAG="--resume $CKPT_DIR/pretrain/latest"
    if [ -f "$CKPT_DIR/sft/latest" ]; then
        echo "  Resuming SFT from checkpoint: $(cat $CKPT_DIR/sft/latest)"
        RESUME_FLAG="--resume $CKPT_DIR/sft/latest"
    fi

    python train.py \
        --config configs/fast_1b_b200_sft.yaml \
        --data-dir "$DATA_DIR/sft" \
        --output-dir "$CKPT_DIR/sft" \
        --stage sft \
        $RESUME_FLAG \
        --save-every 100 --keep-checkpoints 3
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ALL DONE!"
echo "  Test your chatbot:"
echo "    python chat.py --checkpoint $CKPT_DIR/sft/inference_model.pt"
echo "═══════════════════════════════════════════════════════"
