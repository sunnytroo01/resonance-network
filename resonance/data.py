"""
Data pipeline for Resonance Network training.

Supports:
- ShardedPretrainDataset: reads pre-tokenized binary shards (prepare_data.py output)
- SFTDataset: reads tokenized chat data with loss masking
- StreamingTextDataset: on-the-fly streaming from HuggingFace (fallback)
"""

import json
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import tiktoken


# ── Pre-tokenized shard dataset (for large-scale training) ──────

class ShardedPretrainDataset(Dataset):
    """
    Reads pre-tokenized binary shards produced by prepare_data.py.
    Memory-maps shards for efficiency — works with terabytes of data.

    In distributed training, each rank gets a different subset of shards
    so there's no data overlap between GPUs.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 2048,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.seq_len = seq_len
        data_dir = Path(data_dir)

        with open(data_dir / "manifest.json") as f:
            self.manifest = json.load(f)

        all_shards = sorted(data_dir.glob("shard_*.bin"))
        if not all_shards:
            raise FileNotFoundError(f"No shards found in {data_dir}")

        # Distribute shards across ranks
        my_shards = all_shards[rank::world_size]
        if not my_shards:
            raise RuntimeError(
                f"Rank {rank} got 0 shards out of {len(all_shards)} "
                f"(world_size={world_size}). Need more shards than GPUs."
            )

        # Memory-map each shard
        self.shards = []
        self.shard_lengths = []
        self.cumulative = [0]
        total = 0
        for s in my_shards:
            mmap = np.memmap(str(s), dtype=np.uint16, mode="r")
            self.shards.append(mmap)
            self.shard_lengths.append(len(mmap))
            total += len(mmap)
            self.cumulative.append(total)

        self.total_tokens = total
        self.num_samples = total // (seq_len + 1)
        self._cumulative_arr = np.array(self.cumulative)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        chunk = self._read_range(start, self.seq_len + 1)
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y

    def _read_range(self, start: int, length: int) -> np.ndarray:
        """Read `length` tokens starting at global position `start`, spanning shards."""
        result = np.empty(length, dtype=np.uint16)
        pos = 0
        shard_idx = int(np.searchsorted(self._cumulative_arr[1:], start, side="right"))
        local_pos = start - self.cumulative[shard_idx]

        while pos < length:
            shard = self.shards[shard_idx]
            available = len(shard) - local_pos
            to_copy = min(length - pos, available)
            result[pos : pos + to_copy] = shard[local_pos : local_pos + to_copy]
            pos += to_copy
            shard_idx += 1
            local_pos = 0

        return result


# ── SFT dataset (chat/instruction fine-tuning) ─────────────────

class SFTDataset(Dataset):
    """
    Reads tokenized SFT data produced by prepare_data.py.
    Returns (input_ids, labels) where labels=-100 for user tokens (no loss).
    """

    def __init__(self, data_dir: str, max_seq_len: int = 2048):
        data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len

        self.data = np.memmap(str(data_dir / "data.bin"), dtype=np.uint16, mode="r")
        with open(data_dir / "index.json") as f:
            meta = json.load(f)
        self.examples = meta["examples"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        offset = ex["offset"]
        length = min(ex["length"], self.max_seq_len + 1)
        mask_start = ex["mask_start"]

        tokens = self.data[offset : offset + length].astype(np.int64)

        # Pad short sequences
        if len(tokens) < self.max_seq_len + 1:
            padded = np.full(self.max_seq_len + 1, -100, dtype=np.int64)
            padded[: len(tokens)] = tokens
            tokens = padded

        x = torch.from_numpy(tokens[:-1].copy())
        y = torch.from_numpy(tokens[1:].copy())

        # Mask loss on user tokens — only train on assistant response
        y[: max(0, mask_start - 1)] = -100

        return x, y


# ── Streaming dataset (fallback for quick experiments) ──────────

class StreamingTextDataset(IterableDataset):
    """
    Streams text from HuggingFace, tokenizes on-the-fly.
    Use ShardedPretrainDataset for production training.
    """

    def __init__(
        self,
        dataset_name: str = "cerebras/SlimPajama-627B",
        split: str = "train",
        seq_len: int = 2048,
        tokenizer_name: str = "gpt2",
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.seq_len = seq_len
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.vocab_size = self.enc.n_vocab

    def _token_generator(self):
        from datasets import load_dataset

        ds = load_dataset(
            self.dataset_name, split=self.split,
            streaming=True, trust_remote_code=True,
        )
        for example in ds:
            text = example.get("text", "")
            if not text:
                continue
            tokens = self.enc.encode(text, allowed_special=set())
            yield from tokens

    def __iter__(self):
        buffer = []
        for token in self._token_generator():
            buffer.append(token)
            if len(buffer) == self.seq_len + 1:
                x = torch.tensor(buffer[:-1], dtype=torch.long)
                y = torch.tensor(buffer[1:], dtype=torch.long)
                yield x, y
                buffer = []


# ── Factory functions ───────────────────────────────────────────

def create_pretrain_loader(
    data_dir: str,
    seq_len: int = 2048,
    batch_size: int = 8,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
) -> tuple:
    """Create dataloader from pre-tokenized shards."""
    dataset = ShardedPretrainDataset(
        data_dir=data_dir, seq_len=seq_len,
        rank=rank, world_size=world_size,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    enc = tiktoken.get_encoding("gpt2")
    return loader, enc.n_vocab


def create_sft_loader(
    data_dir: str,
    max_seq_len: int = 2048,
    batch_size: int = 8,
    num_workers: int = 4,
) -> tuple:
    """Create dataloader from tokenized SFT data."""
    dataset = SFTDataset(data_dir=data_dir, max_seq_len=max_seq_len)
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    enc = tiktoken.get_encoding("gpt2")
    return loader, enc.n_vocab


def create_dataloader(
    dataset_name: str = "cerebras/SlimPajama-627B",
    split: str = "train",
    seq_len: int = 2048,
    batch_size: int = 8,
    tokenizer_name: str = "gpt2",
    num_workers: int = 4,
) -> tuple:
    """Create a streaming dataloader (fallback)."""
    dataset = StreamingTextDataset(
        dataset_name=dataset_name, split=split,
        seq_len=seq_len, tokenizer_name=tokenizer_name,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    return loader, dataset.vocab_size
