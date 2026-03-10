"""
Streaming data pipeline for large-scale training.

Supports:
- SlimPajama (627B tokens) via HuggingFace streaming
- RedPajama, The Pile, etc.
- tiktoken tokenizer (GPT-2 compatible, 50257 vocab)
- Efficient chunking and packing for causal LM
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
import tiktoken
from typing import Optional
import itertools


class StreamingTextDataset(IterableDataset):
    """
    Streams text from HuggingFace datasets, tokenizes on-the-fly,
    and packs into fixed-length sequences for efficient training.
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
        """Yield tokens from streaming dataset."""
        from datasets import load_dataset

        ds = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )

        for example in ds:
            text = example.get("text", "")
            if not text:
                continue
            tokens = self.enc.encode(text, allowed_special=set())
            yield from tokens

    def __iter__(self):
        """Pack tokens into (input, target) pairs of length seq_len."""
        buffer = []
        for token in self._token_generator():
            buffer.append(token)
            if len(buffer) == self.seq_len + 1:
                x = torch.tensor(buffer[:-1], dtype=torch.long)
                y = torch.tensor(buffer[1:], dtype=torch.long)
                yield x, y
                buffer = []


def create_dataloader(
    dataset_name: str = "cerebras/SlimPajama-627B",
    split: str = "train",
    seq_len: int = 2048,
    batch_size: int = 8,
    tokenizer_name: str = "gpt2",
    num_workers: int = 4,
) -> tuple:
    """Create a streaming dataloader and return (loader, vocab_size)."""
    dataset = StreamingTextDataset(
        dataset_name=dataset_name,
        split=split,
        seq_len=seq_len,
        tokenizer_name=tokenizer_name,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return loader, dataset.vocab_size
