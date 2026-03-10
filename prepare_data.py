#!/usr/bin/env python3
"""
One-command data preparation for Resonance Network.

Downloads, tokenizes, and shards ALL training data into binary format:
  Phase 1 - Pre-training corpus (SlimPajama + OpenWebText + StarCoder + Wikipedia)
  Phase 2 - Chat/instruction SFT data (OpenAssistant, Dolly, Alpaca, OpenOrca)

Usage:
    python prepare_data.py --output /workspace/data

Output:
    data/
      pretrain/
        shard_000000.bin   # 100M tokens each, uint16
        shard_000001.bin
        ...
        manifest.json
      sft/
        data.bin           # all SFT tokens packed, uint16
        index.json         # per-example offsets + loss mask starts
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

import tiktoken

# ── Config ──────────────────────────────────────────────────
SHARD_TOKENS = 100_000_000  # 100M tokens per shard (~200MB)

PRETRAIN_SOURCES = [
    {
        "name": "cerebras/SlimPajama-627B",
        "split": "train",
        "text_key": "text",
        "description": "627B token web corpus",
    },
    {
        "name": "Skylion007/openwebtext",
        "split": "train",
        "text_key": "text",
        "description": "OpenWebText (~9B tokens)",
    },
    {
        "name": "bigcode/starcoderdata",
        "split": "train",
        "text_key": "content",
        "description": "Code data for coding ability",
    },
    {
        "name": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "text_key": "text",
        "description": "English Wikipedia",
    },
]

SFT_SOURCES = [
    "OpenAssistant/oasst1",
    "databricks/databricks-dolly-15k",
    "yahma/alpaca-cleaned",
    "Open-Orca/OpenOrca",
]


# ── Shard writer ────────────────────────────────────────────
class ShardWriter:
    """Writes tokenized data into fixed-size binary shards."""

    def __init__(self, output_dir: Path, shard_tokens: int = SHARD_TOKENS):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = shard_tokens
        self.buffer = np.empty(shard_tokens, dtype=np.uint16)
        self.buf_pos = 0
        self.shard_idx = 0
        self.total_tokens = 0

        # Resume: skip already-written shards
        existing = sorted(self.output_dir.glob("shard_*.bin"))
        if existing:
            self.shard_idx = len(existing)
            self.total_tokens = sum(
                os.path.getsize(str(s)) // 2 for s in existing
            )
            print(f"  Resuming from shard {self.shard_idx} ({self.total_tokens:,} tokens already done)")

    def add_tokens(self, tokens: np.ndarray):
        """Add a numpy uint16 array of tokens to the buffer."""
        pos = 0
        while pos < len(tokens):
            space = self.shard_tokens - self.buf_pos
            chunk = min(len(tokens) - pos, space)
            self.buffer[self.buf_pos : self.buf_pos + chunk] = tokens[pos : pos + chunk]
            self.buf_pos += chunk
            pos += chunk
            if self.buf_pos >= self.shard_tokens:
                self._flush()

    def _flush(self):
        path = self.output_dir / f"shard_{self.shard_idx:06d}.bin"
        self.buffer[: self.buf_pos].tofile(str(path))
        self.total_tokens += self.buf_pos
        self.shard_idx += 1
        self.buf_pos = 0
        elapsed = time.time() - _start_time
        rate = self.total_tokens / max(elapsed, 1)
        print(
            f"  Shard {self.shard_idx - 1:>5d} | "
            f"{self.total_tokens / 1e9:.2f}B tokens | "
            f"{rate / 1e6:.1f}M tok/s"
        )

    def finalize(self) -> int:
        if self.buf_pos > 0:
            self._flush()
        manifest = {
            "num_shards": self.shard_idx,
            "total_tokens": int(self.total_tokens),
            "shard_tokens": self.shard_tokens,
            "dtype": "uint16",
        }
        with open(self.output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return self.total_tokens


# ── Phase 1: Pre-training data ─────────────────────────────
def prepare_pretrain(output_dir: Path, enc):
    """Stream, tokenize, and shard pre-training data."""
    from datasets import load_dataset

    writer = ShardWriter(output_dir / "pretrain")
    eot = np.array([enc.eot_token], dtype=np.uint16)

    for source in PRETRAIN_SOURCES:
        ds_name = source["name"]
        text_key = source["text_key"]
        desc = source["description"]
        split = source["split"]
        config = source.get("config", None)

        print(f"\n{'=' * 60}")
        print(f"  {ds_name}")
        print(f"  {desc}")
        print(f"{'=' * 60}")

        try:
            kwargs = dict(split=split, streaming=True, trust_remote_code=True)
            if config:
                ds = load_dataset(ds_name, config, **kwargs)
            else:
                ds = load_dataset(ds_name, **kwargs)
        except Exception as e:
            print(f"  WARN: Could not load {ds_name}: {e}")
            print(f"  Skipping...")
            continue

        doc_count = 0
        for example in ds:
            text = example.get(text_key, "")
            if not text or len(text) < 50:
                continue

            tokens = enc.encode(text, allowed_special=set())
            arr = np.array(tokens, dtype=np.uint16)
            writer.add_tokens(arr)
            writer.add_tokens(eot)  # document boundary
            doc_count += 1

            if doc_count % 100_000 == 0:
                print(f"    {doc_count:>10,} docs | {writer.total_tokens + writer.buf_pos:,} tokens")

        print(f"  Finished {ds_name}: {doc_count:,} documents")

    total = writer.finalize()
    print(f"\n  Pre-training complete: {total:,} tokens in {writer.shard_idx} shards")
    return total


# ── Phase 2: SFT / Chat data ───────────────────────────────
def prepare_sft(output_dir: Path, enc):
    """Download and prepare chat/instruction SFT data."""
    from datasets import load_dataset

    sft_dir = output_dir / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)

    all_examples = []  # list of {"user": str, "assistant": str}

    # ── OpenAssistant ──
    print(f"\n{'=' * 60}")
    print("  OpenAssistant/oasst1")
    print(f"{'=' * 60}")
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
        messages = {}
        children_map = {}
        for msg in ds:
            messages[msg["message_id"]] = msg
            pid = msg["parent_id"]
            if pid:
                children_map.setdefault(pid, []).append(msg)

        count = 0
        for msg in ds:
            if msg["parent_id"] is not None or msg["role"] != "prompter":
                continue
            kids = children_map.get(msg["message_id"], [])
            assistants = [c for c in kids if c["role"] == "assistant"]
            if not assistants:
                continue
            # Pick highest-quality response (lowest rank number)
            best = sorted(assistants, key=lambda m: m.get("rank") or 999)[0]
            user_text = msg["text"].strip()
            asst_text = best["text"].strip()
            if user_text and asst_text:
                all_examples.append({"user": user_text, "assistant": asst_text})
                count += 1
        print(f"  {count} conversations")
    except Exception as e:
        print(f"  WARN: {e}")

    # ── Dolly ──
    print(f"\n  databricks/databricks-dolly-15k")
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        count = 0
        for ex in ds:
            instruction = ex["instruction"].strip()
            context = ex.get("context", "").strip()
            response = ex["response"].strip()
            if not instruction or not response:
                continue
            user_msg = f"{instruction}\n\n{context}" if context else instruction
            all_examples.append({"user": user_msg, "assistant": response})
            count += 1
        print(f"  {count} examples")
    except Exception as e:
        print(f"  WARN: {e}")

    # ── Alpaca ──
    print(f"\n  yahma/alpaca-cleaned")
    try:
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        count = 0
        for ex in ds:
            instruction = ex["instruction"].strip()
            inp = ex.get("input", "").strip()
            output = ex["output"].strip()
            if not instruction or not output:
                continue
            user_msg = f"{instruction}\n\n{inp}" if inp else instruction
            all_examples.append({"user": user_msg, "assistant": output})
            count += 1
        print(f"  {count} examples")
    except Exception as e:
        print(f"  WARN: {e}")

    # ── OpenOrca (sample 500k) ──
    print(f"\n  Open-Orca/OpenOrca (sampling 500k)")
    try:
        ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
        count = 0
        for ex in ds:
            if count >= 500_000:
                break
            question = ex.get("question", "").strip()
            response = ex.get("response", "").strip()
            if not question or not response:
                continue
            all_examples.append({"user": question, "assistant": response})
            count += 1
        print(f"  {count} examples")
    except Exception as e:
        print(f"  WARN: {e}")

    print(f"\n  Total SFT examples: {len(all_examples)}")

    # ── Tokenize with loss masking ──
    print("  Tokenizing...")
    eot = enc.eot_token
    all_tokens = []
    index = []
    offset = 0

    for ex in all_examples:
        user_part = f"User: {ex['user']}\nAssistant:"
        full_text = f"{user_part} {ex['assistant']}"

        user_tokens = enc.encode(user_part, allowed_special=set())
        full_tokens = enc.encode(full_text, allowed_special=set())
        full_tokens.append(eot)

        all_tokens.extend(full_tokens)
        index.append({
            "offset": offset,
            "length": len(full_tokens),
            "mask_start": len(user_tokens),  # loss starts here (assistant response)
        })
        offset += len(full_tokens)

    # Save
    np.array(all_tokens, dtype=np.uint16).tofile(str(sft_dir / "data.bin"))
    with open(sft_dir / "index.json", "w") as f:
        json.dump({
            "num_examples": len(index),
            "total_tokens": len(all_tokens),
            "examples": index,
        }, f)

    print(f"  SFT data ready: {len(index)} examples, {len(all_tokens):,} tokens")
    print(f"  Saved to {sft_dir}")
    return len(all_tokens)


# ── Main ────────────────────────────────────────────────────
_start_time = time.time()


def main():
    global _start_time
    _start_time = time.time()

    parser = argparse.ArgumentParser(
        description="One-command data preparation for Resonance Network"
    )
    parser.add_argument(
        "--output", type=str, default="data",
        help="Output directory (use network volume path, e.g. /workspace/data)",
    )
    parser.add_argument(
        "--phase", type=str, default="all", choices=["all", "pretrain", "sft"],
        help="Which phase to run (default: all)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    enc = tiktoken.get_encoding("gpt2")

    print("=" * 60)
    print("  RESONANCE NETWORK — DATA PREPARATION")
    print("=" * 60)
    print(f"  Output:    {output_dir.resolve()}")
    print(f"  Tokenizer: tiktoken gpt2 ({enc.n_vocab} vocab)")
    print(f"  Phase:     {args.phase}")
    print("=" * 60)

    if args.phase in ("all", "pretrain"):
        print("\n\n  PHASE 1: PRE-TRAINING DATA")
        print("  " + "-" * 40)
        pretrain_tokens = prepare_pretrain(output_dir, enc)

    if args.phase in ("all", "sft"):
        print("\n\n  PHASE 2: SFT / CHAT DATA")
        print("  " + "-" * 40)
        sft_tokens = prepare_sft(output_dir, enc)

    elapsed = time.time() - _start_time
    print(f"\n{'=' * 60}")
    print(f"  ALL DATA READY — {elapsed / 3600:.1f} hours elapsed")
    print(f"  Output: {output_dir.resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
