"""
Distributed training script for Resonance Network.

Supports:
- Single GPU
- Multi-GPU with PyTorch FSDP (Fully Sharded Data Parallel)
- Mixed precision (bfloat16)
- Gradient accumulation
- Streaming datasets
- Checkpoint saving/resuming
- Wandb logging

Usage:
    # Single GPU
    python train.py --config configs/small_125m.yaml

    # Multi-GPU (FSDP)
    torchrun --nproc_per_node=8 train.py --config configs/xl_7b.yaml

    # Multi-node
    torchrun --nnodes=4 --nproc_per_node=8 train.py --config configs/titan_1t.yaml
"""

import os
import sys
import time
import math
import json
import yaml
import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from resonance.model.resonance_network import ResonanceNetwork, ResonanceLayer
from resonance.data import create_dataloader


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def log(msg: str, rank: int = 0):
    """Only print on rank 0."""
    if rank == 0:
        print(msg, flush=True)


def save_checkpoint(model, optimizer, step, config, output_dir, rank):
    """Save model checkpoint (only on rank 0 for FSDP)."""
    if rank != 0:
        return

    path = Path(output_dir) / f"checkpoint_{step}.pt"
    state = {
        "step": step,
        "config": config,
    }

    # For FSDP, we need to gather the full state dict
    if isinstance(model, FSDP):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state["model_state_dict"] = model.state_dict()
            state["optimizer_state_dict"] = optimizer.state_dict()
    else:
        state["model_state_dict"] = model.state_dict()
        state["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(state, path)
    log(f"Checkpoint saved: {path}", rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="resonance-network")
    args = parser.parse_args()

    # Distributed setup
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}"
    is_distributed = world_size > 1

    # Load config
    config = load_config(args.config)
    mc = config["model"]
    tc = config["training"]
    dc = config["data"]

    # Output
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

    log(f"=" * 60, rank)
    log(f"RESONANCE NETWORK - DISTRIBUTED TRAINING", rank)
    log(f"=" * 60, rank)
    log(f"World size: {world_size}", rank)
    log(f"Model dim: {mc['dim']}, layers: {mc['n_layers']}, heads: {mc['num_heads']}", rank)
    log(f"Seq len: {tc['seq_len']}, batch: {tc['batch_size']} x {tc.get('gradient_accumulation', 1)} accum", rank)
    log(f"Effective batch: {tc['batch_size'] * tc.get('gradient_accumulation', 1) * world_size}", rank)
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name()}", rank)
        log(f"VRAM: {torch.cuda.get_device_properties(local_rank).total_mem / 1e9:.1f} GB", rank)
    log(f"=" * 60, rank)

    # Data
    log("Loading streaming dataset...", rank)
    dataset_map = {
        "slimpajama": "cerebras/SlimPajama-627B",
        "redpajama": "togethercomputer/RedPajama-Data-1T",
        "openwebtext": "Skylion007/openwebtext",
    }
    dataset_name = dataset_map.get(dc["dataset"], dc["dataset"])

    train_loader, vocab_size = create_dataloader(
        dataset_name=dataset_name,
        split="train",
        seq_len=tc["seq_len"],
        batch_size=tc["batch_size"],
        tokenizer_name=dc.get("tokenizer", "gpt2"),
        num_workers=dc.get("num_workers", 4),
    )
    log(f"Vocab size: {vocab_size}", rank)

    # Build model
    log("Building Resonance Network...", rank)
    model = ResonanceNetwork(
        vocab_size=vocab_size,
        dim=mc["dim"],
        n_layers=mc["n_layers"],
        max_seq_len=mc.get("max_seq_len", tc["seq_len"]),
        coupling_rank=mc.get("coupling_rank", mc["dim"] // 4),
        num_heads=mc["num_heads"],
        num_stored_patterns=mc.get("num_stored_patterns", 256),
        hopfield_steps=mc.get("hopfield_steps", 1),
        mag_expansion=mc.get("mag_expansion", 4),
        dropout=mc.get("dropout", 0.1),
        dt=mc.get("dt", 0.1),
        use_sparsemax=mc.get("use_sparsemax", True),
        stability_weight=mc.get("stability_weight", 0.01),
    ).to(device)

    n_params = model.get_num_params()
    log(f"Parameters: {n_params:,} (non-embedding)", rank)
    log(f"Parameters: {model.get_num_params(False):,} (total)", rank)

    # Wrap with FSDP for multi-GPU
    if is_distributed:
        log("Wrapping with FSDP...", rank)
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={ResonanceLayer},
        )

        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=local_rank,
            limit_all_gathers=True,
        )
        log("FSDP ready", rank)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc["learning_rate"],
        weight_decay=tc.get("weight_decay", 0.1),
        betas=(0.9, 0.95),
        fused=True,
    )

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        log(f"Resuming from {args.resume}...", rank)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]
        log(f"Resumed at step {start_step}", rank)

    # Wandb
    if args.wandb and rank == 0:
        import wandb
        wandb.init(project=args.wandb_project, config=config)

    # Training loop
    grad_accum = tc.get("gradient_accumulation", 1)
    max_steps = tc["max_steps"]
    log_interval = 10
    save_interval = 5000
    amp_dtype = torch.bfloat16

    log(f"\nStarting training for {max_steps} steps...", rank)

    model.train()
    step = start_step
    accum_loss = 0.0
    accum_count = 0
    data_iter = iter(train_loader)

    t0 = time.time()

    while step < max_steps:
        # Accumulate gradients
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(grad_accum):
            try:
                input_ids, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                input_ids, targets = next(data_iter)

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Forward
            ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
            with ctx:
                logits, loss, info = model(input_ids, targets)
                loss = loss / grad_accum

            # Backward
            loss.backward()
            accum_loss += loss.item() * grad_accum
            accum_count += 1

        # Update LR
        lr = get_lr(step, tc["warmup_steps"], max_steps, tc["learning_rate"], tc.get("min_lr", 1e-5))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Clip and step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("max_grad_norm", 1.0))
        optimizer.step()

        step += 1

        # Logging
        if step % log_interval == 0:
            avg_loss = accum_loss / accum_count
            ppl = math.exp(min(avg_loss, 20))
            dt = time.time() - t0
            tokens_per_sec = (tc["batch_size"] * tc["seq_len"] * log_interval * world_size) / dt

            log(
                f"step {step:>7d} | loss {avg_loss:.4f} | ppl {ppl:.1f} | "
                f"lr {lr:.2e} | grad {grad_norm:.2f} | "
                f"tok/s {tokens_per_sec:.0f} | "
                f"gpu_mem {torch.cuda.max_memory_allocated(device) / 1e9:.1f}GB",
                rank,
            )

            if args.wandb and rank == 0:
                import wandb
                wandb.log({
                    "loss": avg_loss, "perplexity": ppl, "lr": lr,
                    "grad_norm": grad_norm, "tokens_per_sec": tokens_per_sec,
                    "step": step,
                })

            accum_loss = 0.0
            accum_count = 0
            t0 = time.time()

        # Save checkpoint
        if step % save_interval == 0:
            if is_distributed:
                dist.barrier()
            save_checkpoint(model, optimizer, step, config, output_dir, rank)

    # Final save
    if is_distributed:
        dist.barrier()
    save_checkpoint(model, optimizer, step, config, output_dir, rank)

    log("\nTraining complete!", rank)
    cleanup_distributed()


if __name__ == "__main__":
    main()
