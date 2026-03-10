# Resonance Network

A novel non-transformer architecture for large language models that replaces self-attention with **Kuramoto oscillatory phase coupling**, **Hopfield associative memory**, and **complex-valued neural computation**.

## Architecture

Resonance Network operates on complex-valued hidden states (amplitude + phase) and processes information through:

- **Kuramoto Phase Coupling** — Oscillatory neurons synchronize through learned coupling weights, performing content mixing via phase dynamics rather than dot-product attention. Achieves O(S·D) complexity instead of O(S²·D).
- **Hopfield Associative Memory** — Learned stored patterns + dynamic patterns from input, with sparsemax retrieval for sharp memory access.
- **Complex-valued Residual Stream** — Hidden states are complex tensors where magnitude carries content and phase carries positional/relational information.
- **SwiGLU Magnitude FFN** — Feed-forward network operates on magnitudes only, preserving phase information.
- **ComplexRMSNorm** — Normalizes magnitude while preserving phase.
- **Rotary Positional Embeddings in Complex Space** — Natural rotation via complex multiplication.

## Model Scales

| Config | Params | Layers | Dim | Heads | GPUs |
|--------|--------|--------|-----|-------|------|
| `small_125m` | 125M | 12 | 768 | 12 | 1 |
| `medium_350m` | 350M | 24 | 1024 | 16 | 1-8 |
| `large_1b` | 1.3B | 24 | 2048 | 16 | 8 |
| `xl_7b` | 7B | 32 | 4096 | 32 | 8 |
| `giant_70b` | 70B | 80 | 8192 | 64 | 64+ |
| `titan_1t` | 1T | 96 | 16384 | 128 | 512+ |

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Training

```bash
# Single GPU (125M)
python train.py --config configs/small_125m.yaml --output-dir checkpoints/small --wandb

# Multi-GPU with FSDP (7B)
torchrun --nproc_per_node=8 train.py --config configs/xl_7b.yaml --output-dir checkpoints/7b --wandb

# Multi-node (1T)
torchrun --nnodes=64 --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py --config configs/titan_1t.yaml --output-dir checkpoints/1t --wandb
```

### Generation

```bash
python chat.py --checkpoint checkpoints/checkpoint_100000.pt
```

## Key Differences from Transformers

| | Transformer | Resonance Network |
|---|---|---|
| **Core mechanism** | Scaled dot-product attention | Kuramoto phase coupling |
| **Hidden states** | Real-valued | Complex-valued (amplitude + phase) |
| **Memory** | None (in-context only) | Hopfield associative memory |
| **Positional encoding** | RoPE on real vectors | Complex rotation (natural in C) |
| **Sequence complexity** | O(S²·D) | O(S·D) |
| **Normalization** | RMSNorm | ComplexRMSNorm (preserves phase) |

## Project Structure

```
resonance-network/
├── resonance/
│   ├── model/
│   │   ├── complex_ops.py         # Complex-valued operations
│   │   ├── oscillatory_block.py   # Kuramoto coupling + SwiGLU FFN
│   │   ├── hopfield_memory.py     # Hopfield associative memory
│   │   └── resonance_network.py   # Full model (N-layer stack)
│   ├── data.py                    # Streaming data pipeline (tiktoken)
│   └── generate.py                # Text generation (top-k, top-p, etc.)
├── configs/                       # YAML configs for each scale
├── scripts/                       # Training launch scripts
├── train.py                       # Distributed training (FSDP)
└── chat.py                        # Interactive generation
```

## Requirements

- PyTorch >= 2.2.0
- NVIDIA GPUs with bfloat16 support (A100, H100, B200)
- For multi-node: NCCL backend

## License

All rights reserved.
