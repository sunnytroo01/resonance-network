"""
Distributed training script for Resonance Network.

Supports:
- Single GPU or Multi-GPU with FSDP
- Pre-training on binary shards (prepare_data.py output)
- SFT chat fine-tuning with loss masking
- Very frequent rolling checkpoints (cannot lose progress)
- Permanent checkpoints at intervals
- Streaming dataset fallback
- Mixed precision (bfloat16)
- Wandb logging

Usage:
    # Pre-train 1T on B200 cluster with frequent checkpoints
    torchrun --nnodes=64 --nproc_per_node=8 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train.py --config configs/titan_1t.yaml \
        --data-dir /workspace/data/pretrain \
        --output-dir /workspace/checkpoints/pretrain \
        --save-every 250 --keep-checkpoints 3 --wandb

    # SFT fine-tune for chat
    torchrun --nnodes=64 --nproc_per_node=8 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train.py --config configs/titan_1t_sft.yaml \
        --data-dir /workspace/data/sft --stage sft \
        --resume /workspace/checkpoints/pretrain/latest \
        --output-dir /workspace/checkpoints/sft \
        --save-every 100 --keep-checkpoints 3 --wandb
"""

import os
import time
import math
import json
import yaml
import shutil
import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from resonance.model.resonance_network import ResonanceNetwork, ResonanceLayer
from resonance.data import create_dataloader, create_pretrain_loader, create_sft_loader


# ── Distributed setup ──────────────────────────────────────

def setup_distributed():
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


# ── LR schedule ────────────────────────────────────────────

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ── Helpers ────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    # PyYAML sometimes parses scientific notation (e.g. 3e-4) as strings.
    # Normalize all string values that look like numbers to int/float.
    for section in config.values():
        if not isinstance(section, dict):
            continue
        for key, val in section.items():
            if isinstance(val, str):
                try:
                    section[key] = int(val)
                except ValueError:
                    try:
                        section[key] = float(val)
                    except ValueError:
                        pass
    return config


def log(msg, rank=0):
    if rank == 0:
        print(msg, flush=True)


# ── Checkpoint management ──────────────────────────────────

def save_checkpoint(model, optimizer, step, config, output_dir, rank, is_distributed):
    """Save checkpoint using distributed checkpoint for FSDP, regular for single GPU."""
    if rank != 0 and not is_distributed:
        return

    ckpt_dir = Path(output_dir) / f"step_{step}"

    if is_distributed:
        # Use torch.distributed.checkpoint for sharded saves (fast, scalable)
        import torch.distributed.checkpoint as dist_cp
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        dist_cp.save(state_dict, checkpoint_id=str(ckpt_dir))

        if rank == 0:
            with open(ckpt_dir / "metadata.json", "w") as f:
                json.dump({"step": step, "config": config}, f)
    else:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "step": step,
            "config": config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(state, ckpt_dir / "checkpoint.pt")

    # Update "latest" symlink/marker
    if rank == 0:
        latest_file = Path(output_dir) / "latest"
        latest_file.write_text(str(ckpt_dir.resolve()))
        log(f"  Checkpoint saved: {ckpt_dir}", rank)


def manage_rolling_checkpoints(output_dir, keep_n, rank):
    """Delete old rolling checkpoints, keeping the last N."""
    if rank != 0:
        return
    output_dir = Path(output_dir)
    # Find all step_* directories that are NOT in permanent/
    ckpt_dirs = sorted(output_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    while len(ckpt_dirs) > keep_n:
        oldest = ckpt_dirs.pop(0)
        shutil.rmtree(oldest, ignore_errors=True)


def save_permanent_checkpoint(model, optimizer, step, config, output_dir, rank, is_distributed):
    """Save a permanent checkpoint that never gets deleted."""
    perm_dir = Path(output_dir) / "permanent"
    if rank == 0:
        perm_dir.mkdir(parents=True, exist_ok=True)
    if is_distributed:
        dist.barrier()

    ckpt_dir = perm_dir / f"step_{step}"
    if is_distributed:
        import torch.distributed.checkpoint as dist_cp
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        dist_cp.save(state_dict, checkpoint_id=str(ckpt_dir))
        if rank == 0:
            with open(ckpt_dir / "metadata.json", "w") as f:
                json.dump({"step": step, "config": config}, f)
    else:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "step": step, "config": config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(state, ckpt_dir / "checkpoint.pt")

    if rank == 0:
        log(f"  Permanent checkpoint saved: {ckpt_dir}", rank)


def save_inference_checkpoint(model, config, output_dir, rank, is_distributed):
    """Save a consolidated single-file checkpoint that chat.py can load."""
    if is_distributed:
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model.state_dict()
    else:
        state_dict = model.state_dict()

    if rank == 0:
        ckpt_path = Path(output_dir) / "inference_model.pt"
        torch.save({
            "model_state_dict": state_dict,
            "config": config,
        }, ckpt_path)
        log(f"  Inference checkpoint saved: {ckpt_path}", rank)


def load_checkpoint(model, optimizer, resume_path, device, rank, is_distributed):
    """Load from a checkpoint directory."""
    resume_path = Path(resume_path)

    # If it's a "latest" file, read the actual path
    if resume_path.name == "latest" and resume_path.is_file():
        resume_path = Path(resume_path.read_text().strip())

    if is_distributed:
        import torch.distributed.checkpoint as dist_cp
        state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        dist_cp.load(state_dict, checkpoint_id=str(resume_path))
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        meta_file = resume_path / "metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            return meta.get("step", 0)
        return 0
    else:
        ckpt_file = resume_path / "checkpoint.pt"
        if ckpt_file.exists():
            ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        else:
            # Legacy single-file checkpoint
            ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt.get("step", 0)


# ── Main ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to pre-tokenized data (from prepare_data.py)")
    parser.add_argument("--stage", type=str, default="pretrain", choices=["pretrain", "sft"],
                        help="Training stage: pretrain or sft")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint dir to resume from (or path to 'latest' file)")
    parser.add_argument("--save-every", type=int, default=250,
                        help="Save rolling checkpoint every N steps (default: 250)")
    parser.add_argument("--keep-checkpoints", type=int, default=3,
                        help="Number of rolling checkpoints to keep (default: 3)")
    parser.add_argument("--permanent-save-every", type=int, default=2000,
                        help="Save permanent checkpoint every N steps (default: 2000)")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="resonance-network")
    args = parser.parse_args()

    # ── Distributed ──
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}"
    is_distributed = world_size > 1

    # ── Config ──
    config = load_config(args.config)
    mc = config["model"]
    tc = config["training"]
    dc = config["data"]

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

    log(f"{'=' * 60}", rank)
    log(f"  RESONANCE NETWORK — {args.stage.upper()}", rank)
    log(f"{'=' * 60}", rank)
    log(f"  World size:       {world_size}", rank)
    log(f"  Model:            dim={mc['dim']}, layers={mc['n_layers']}, heads={mc['num_heads']}", rank)
    log(f"  Seq len:          {tc['seq_len']}", rank)
    log(f"  Batch:            {tc['batch_size']} x {tc.get('gradient_accumulation', 1)} accum x {world_size} GPUs", rank)
    effective_batch = int(tc["batch_size"]) * int(tc.get("gradient_accumulation", 1)) * world_size
    log(f"  Effective batch:  {effective_batch}", rank)
    log(f"  Save every:       {args.save_every} steps (keep {args.keep_checkpoints})", rank)
    log(f"  Permanent save:   every {args.permanent_save_every} steps", rank)
    if torch.cuda.is_available():
        log(f"  GPU:              {torch.cuda.get_device_name()}", rank)
        log(f"  VRAM:             {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB", rank)
    log(f"{'=' * 60}", rank)

    # ── Data ──
    log("Loading data...", rank)

    if args.data_dir:
        if args.stage == "sft":
            train_loader, vocab_size = create_sft_loader(
                data_dir=args.data_dir,
                max_seq_len=tc["seq_len"],
                batch_size=tc["batch_size"],
                num_workers=dc.get("num_workers", 4),
            )
        else:
            train_loader, vocab_size = create_pretrain_loader(
                data_dir=args.data_dir,
                seq_len=tc["seq_len"],
                batch_size=tc["batch_size"],
                num_workers=dc.get("num_workers", 4),
                rank=rank,
                world_size=world_size,
            )
    else:
        # Streaming fallback
        dataset_map = {
            "slimpajama": "cerebras/SlimPajama-627B",
            "redpajama": "togethercomputer/RedPajama-Data-1T",
            "openwebtext": "Skylion007/openwebtext",
        }
        dataset_name = dataset_map.get(dc["dataset"], dc["dataset"])
        train_loader, vocab_size = create_dataloader(
            dataset_name=dataset_name, split="train",
            seq_len=tc["seq_len"], batch_size=tc["batch_size"],
            tokenizer_name=dc.get("tokenizer", "gpt2"),
            num_workers=dc.get("num_workers", 4),
        )

    log(f"  Vocab size: {vocab_size}", rank)

    # ── Model ──
    log("Building Resonance Network...", rank)
    model = ResonanceNetwork(
        vocab_size=vocab_size,
        dim=int(mc["dim"]),
        n_layers=int(mc["n_layers"]),
        max_seq_len=int(mc.get("max_seq_len", tc["seq_len"])),
        coupling_rank=int(mc.get("coupling_rank", mc["dim"] // 4)),
        num_heads=int(mc["num_heads"]),
        num_stored_patterns=int(mc.get("num_stored_patterns", 256)),
        hopfield_steps=int(mc.get("hopfield_steps", 1)),
        mag_expansion=int(mc.get("mag_expansion", 4)),
        dropout=float(mc.get("dropout", 0.1)),
        dt=float(mc.get("dt", 0.1)),
        use_sparsemax=bool(mc.get("use_sparsemax", True)),
        stability_weight=float(mc.get("stability_weight", 0.01)),
    ).to(device)

    n_params = model.get_num_params()
    log(f"  Parameters: {n_params:,} (non-embedding)", rank)
    log(f"  Parameters: {model.get_num_params(False):,} (total)", rank)

    # ── FSDP ──
    if is_distributed:
        log("Wrapping with FSDP...", rank)
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            # NOTE: do NOT set buffer_dtype — model has complex64 buffers
            # (RotaryEmbeddingComplex.rotation) that cannot be cast to bfloat16
        )
        auto_wrap = transformer_auto_wrap_policy(
            transformer_layer_cls={ResonanceLayer},
        )
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=local_rank,
            limit_all_gathers=True,
        )
        log("  FSDP ready", rank)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tc["learning_rate"]),
        weight_decay=float(tc.get("weight_decay", 0.1)),
        betas=(0.9, 0.95),
        fused=True,
    )

    # ── Resume ──
    start_step = 0
    if args.resume:
        log(f"Resuming from {args.resume}...", rank)
        start_step = load_checkpoint(model, optimizer, args.resume, device, rank, is_distributed)
        if start_step >= int(tc["max_steps"]):
            # Resuming from a different stage (e.g. pretrain -> SFT): reset step counter
            log(f"  Loaded weights from step {start_step}, resetting to step 0 for new stage", rank)
            start_step = 0
        else:
            log(f"  Resumed at step {start_step}", rank)

    # ── Wandb ──
    if args.wandb and rank == 0:
        import wandb
        wandb.init(project=args.wandb_project, config=config, name=f"{args.stage}_{mc['dim']}d_{mc['n_layers']}L")

    # ── Training loop ──
    grad_accum = int(tc.get("gradient_accumulation", 1))
    max_steps = int(tc["max_steps"])
    log_interval = 10
    amp_dtype = torch.bfloat16

    log(f"\nStarting {args.stage} training for {max_steps} steps...", rank)
    log(f"  Rolling checkpoints every {args.save_every} steps (keep last {args.keep_checkpoints})", rank)
    log(f"  Permanent checkpoints every {args.permanent_save_every} steps\n", rank)

    model.train()
    step = start_step
    accum_loss = 0.0
    accum_count = 0
    data_iter = iter(train_loader)
    t0 = time.time()

    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(grad_accum):
            try:
                input_ids, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                input_ids, targets = next(data_iter)

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Use no_sync for all micro-steps except the last to avoid
            # redundant all-reduce during gradient accumulation
            sync_context = model.no_sync() if (is_distributed and micro_step < grad_accum - 1) else nullcontext()
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits, loss, info = model(input_ids, targets)
                    loss = loss / grad_accum

                loss.backward()
            accum_loss += loss.item() * grad_accum
            accum_count += 1

        # LR schedule
        lr = get_lr(step, int(tc["warmup_steps"]), max_steps, float(tc["learning_rate"]), float(tc.get("min_lr", 1e-5)))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc.get("max_grad_norm", 1.0)))
        optimizer.step()
        step += 1

        # ── Logging ──
        if step % log_interval == 0:
            avg_loss = accum_loss / accum_count
            ppl = math.exp(min(avg_loss, 20))
            dt = time.time() - t0
            tokens_per_sec = (int(tc["batch_size"]) * int(tc["seq_len"]) * grad_accum * log_interval * world_size) / dt

            log(
                f"  step {step:>8d} | loss {avg_loss:.4f} | ppl {ppl:>8.1f} | "
                f"lr {lr:.2e} | grad {grad_norm:.2f} | "
                f"tok/s {tokens_per_sec:>10,.0f} | "
                f"gpu {torch.cuda.max_memory_allocated(device) / 1e9:.1f}GB",
                rank,
            )

            if args.wandb and rank == 0:
                import wandb
                wandb.log({
                    "loss": avg_loss, "perplexity": ppl, "lr": lr,
                    "grad_norm": float(grad_norm), "tokens_per_sec": tokens_per_sec,
                    "step": step,
                })

            accum_loss = 0.0
            accum_count = 0
            t0 = time.time()

        # ── Rolling checkpoint ──
        if step % args.save_every == 0:
            if is_distributed:
                dist.barrier()
            save_checkpoint(model, optimizer, step, config, output_dir, rank, is_distributed)
            manage_rolling_checkpoints(output_dir, args.keep_checkpoints, rank)

        # ── Permanent checkpoint ──
        if step % args.permanent_save_every == 0:
            if is_distributed:
                dist.barrier()
            save_permanent_checkpoint(model, optimizer, step, config, output_dir, rank, is_distributed)

    # ── Final save ──
    if is_distributed:
        dist.barrier()
    save_checkpoint(model, optimizer, step, config, output_dir, rank, is_distributed)
    save_permanent_checkpoint(model, optimizer, step, config, output_dir, rank, is_distributed)
    save_inference_checkpoint(model, config, output_dir, rank, is_distributed)

    log(f"\n{'=' * 60}", rank)
    log(f"  {args.stage.upper()} COMPLETE — {step} steps", rank)
    log(f"  Checkpoints: {output_dir}", rank)
    log(f"  Inference:   {output_dir}/inference_model.pt", rank)
    log(f"{'=' * 60}", rank)

    cleanup_distributed()


if __name__ == "__main__":
    main()
