#!/bin/bash
set -e
DATA_DIR=${DATA_DIR:-/workspace/data}
CKPT_DIR=${CKPT_DIR:-/workspace/checkpoints}

pip install -q -r requirements.txt

echo "═══════════════════════════════════════════════════════"
echo "  RESONANCE NETWORK 1.3B — 4x RTX 4090"
echo "═══════════════════════════════════════════════════════"

# Data prep if needed
if [ ! -f "$DATA_DIR/pretrain/manifest.json" ]; then
    echo "Preparing data first..."
    python prepare_data.py --output $DATA_DIR
fi
if [ ! -f "$DATA_DIR/sft/index.json" ]; then
    python prepare_data.py --output $DATA_DIR --phase sft
fi

# Phase 1: Pretrain
echo "PHASE 1: PRE-TRAINING"
torchrun --nproc_per_node=4 \
    train.py \
    --config configs/fast_1b_4090.yaml \
    --data-dir $DATA_DIR/pretrain \
    --output-dir $CKPT_DIR/pretrain \
    --save-every 200 --keep-checkpoints 3 --permanent-save-every 1000 \
    --wandb --wandb-project resonance-1b

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
    --wandb --wandb-project resonance-1b-sft

echo "DONE! python chat.py --checkpoint $CKPT_DIR/sft/inference_model.pt"
