#!/bin/bash
set -e
# ═══════════════════════════════════════════════════════════
#  DATA PREPARATION ONLY — run on a cheap CPU/small GPU pod
#  with network volume mounted at /workspace
# ═══════════════════════════════════════════════════════════
DATA_DIR=${DATA_DIR:-/workspace/data}

pip install -q -r requirements.txt

echo "═══════════════════════════════════════════════════════"
echo "  RESONANCE NETWORK — DATA PREPARATION"
echo "  Run this on a CHEAP pod (CPU-only or 1x small GPU)"
echo "  Network volume: /workspace"
echo "═══════════════════════════════════════════════════════"

# Phase 1: Pretrain data (FineWeb-Edu, OpenWebText, Wikipedia)
if [ ! -f "$DATA_DIR/pretrain/manifest.json" ]; then
    echo "Preparing pretrain data..."
    python prepare_data.py --output $DATA_DIR
else
    echo "Pretrain data already exists, skipping."
fi

# Phase 2: SFT data (OpenAssistant, Dolly, Alpaca, OpenOrca)
if [ ! -f "$DATA_DIR/sft/index.json" ]; then
    echo "Preparing SFT data..."
    python prepare_data.py --output $DATA_DIR --phase sft
else
    echo "SFT data already exists, skipping."
fi

echo "═══════════════════════════════════════════════════════"
echo "  DATA PREP COMPLETE!"
echo "  Data saved to: $DATA_DIR"
echo ""
echo "  Next: stop this pod, spin up your GPU pod with the"
echo "  same network volume, and run:"
echo "    bash scripts/train_only.sh"
echo "═══════════════════════════════════════════════════════"
