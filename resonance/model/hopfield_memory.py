"""
Modern Continuous Hopfield Network as Associative Memory.

Replaces the KV-cache / attention mechanism with energy-based associative memory.

Energy function: E = -log(sum_i exp(xi^T * z / sqrt(d)))
Update rule: z_new = softmax(Z^T * z / sqrt(d)) @ Z

Key properties:
- Operates as energy minimization, not a single forward pass
- Can iterate multiple times for harder retrievals
- Stored patterns can be LEARNED (not just from context)
- Has exponential storage capacity in dimension d
- Uses sparsemax (Hopfield-Fenchel-Young) for sparse retrieval

References:
- Ramsauer et al., "Hopfield Networks is All You Need" (2021)
- Martins & Astudillo, "From Softmax to Sparsemax" (2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .complex_ops import ComplexLinear, ComplexRMSNorm, complex_magnitude, complex_phase, make_complex


def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax activation function.
    Projects onto the probability simplex, producing sparse outputs.

    From "From Softmax to Sparsemax" (Martins & Astudillo, 2016).
    """
    z_sorted, _ = z.sort(dim=dim, descending=True)
    z_cumsum = z_sorted.cumsum(dim=dim)
    k = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)

    # Reshape k for broadcasting
    shape = [1] * z.ndim
    shape[dim] = -1
    k = k.view(shape)

    support = (1 + k * z_sorted > z_cumsum).to(z.dtype)
    k_max = support.sum(dim=dim, keepdim=True)

    # Threshold
    tau = (z_cumsum.gather(dim, (k_max - 1).long().clamp(min=0)) - 1) / k_max.clamp(min=1)
    output = (z - tau).clamp(min=0)

    return output


class HopfieldMemory(nn.Module):
    """
    Modern continuous Hopfield network for associative memory.

    Stores both:
    1. Learned patterns (nn.Parameter) -- long-term memory
    2. Dynamic patterns from the input sequence -- working memory

    Retrieval is via energy minimization using (sparse) softmax-weighted
    combination of stored patterns.
    """

    def __init__(
        self,
        dim: int,
        num_stored_patterns: int = 256,
        num_heads: int = 4,
        temperature: float = 1.0,
        retrieval_steps: int = 3,
        use_sparsemax: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_stored_patterns = num_stored_patterns
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = temperature
        self.retrieval_steps = retrieval_steps
        self.use_sparsemax = use_sparsemax

        assert dim % num_heads == 0

        # Learned stored patterns (long-term associative memory)
        self.stored_patterns = nn.Parameter(
            torch.randn(num_stored_patterns, dim) * 0.02
        )

        # Projection layers for query, pattern-key, pattern-value
        # These operate on REAL magnitudes extracted from complex states
        self.W_query = nn.Linear(dim, dim, bias=False)
        self.W_key = nn.Linear(dim, dim, bias=False)
        self.W_value = nn.Linear(dim, dim, bias=False)

        # Output projection back to complex space perturbation
        self.W_out = nn.Linear(dim, dim * 2)  # outputs real + imag components

        # Learnable temperature per head
        self.log_temp = nn.Parameter(torch.full((num_heads,), math.log(temperature)))

        # Layer norm on output
        self.norm = nn.LayerNorm(dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform associative memory retrieval.

        Args:
            z: complex tensor (batch, seq_len, dim) - current oscillatory state
            mask: optional causal mask (seq_len, seq_len)

        Returns:
            update: complex tensor (batch, seq_len, dim) - memory retrieval result
        """
        B, S, D = z.shape
        H = self.num_heads
        hd = self.head_dim

        # Extract magnitude as the query signal
        mag = complex_magnitude(z)  # (B, S, D) real

        # Combine learned patterns with dynamic patterns from input
        # Dynamic patterns: use the magnitudes of current states
        dynamic_patterns = mag  # (B, S, D)

        # Learned patterns expanded for batch: (1, P, D) -> (B, P, D)
        learned = self.stored_patterns.unsqueeze(0).expand(B, -1, -1)

        # Full pattern set: [learned; dynamic]
        patterns = torch.cat([learned, dynamic_patterns], dim=1)  # (B, P+S, D)
        P_total = patterns.shape[1]

        # Project queries, keys, values
        queries = self.W_query(mag).view(B, S, H, hd).transpose(1, 2)       # (B, H, S, hd)
        keys = self.W_key(patterns).view(B, P_total, H, hd).transpose(1, 2)  # (B, H, P+S, hd)
        values = self.W_value(patterns).view(B, P_total, H, hd).transpose(1, 2)

        # Temperature per head (clamp log_temp to prevent underflow/overflow)
        temp = self.log_temp.clamp(-4.0, 4.0).exp().view(1, H, 1, 1)  # (1, H, 1, 1)

        # Iterative Hopfield retrieval (multiple steps of energy minimization)
        state = queries  # initial retrieval state
        for step in range(self.retrieval_steps):
            # Compute association scores
            scores = torch.matmul(state, keys.transpose(-2, -1)) / (math.sqrt(hd) * temp)

            # Apply causal mask if provided (only for dynamic pattern portion)
            if mask is not None:
                # mask shape: (S, S), apply to last S columns of scores
                learned_scores = scores[..., :self.num_stored_patterns]
                dynamic_scores = scores[..., self.num_stored_patterns:]
                dynamic_scores = dynamic_scores + mask.unsqueeze(0).unsqueeze(0)
                scores = torch.cat([learned_scores, dynamic_scores], dim=-1)

            # Sparse or dense retrieval
            if self.use_sparsemax:
                weights = sparsemax(scores, dim=-1)
            else:
                weights = F.softmax(scores, dim=-1)

            # Retrieve
            state = torch.matmul(weights, values)  # (B, H, S, hd)

        # Reshape: (B, H, S, hd) -> (B, S, D)
        retrieved = state.transpose(1, 2).contiguous().view(B, S, D)

        # Project to complex perturbation (outputs 2*D: real and imag parts)
        out = self.W_out(retrieved)  # (B, S, 2*D)
        out = self.norm(out)
        out = self.dropout(out)

        # Convert to complex
        out_complex = make_complex(out[..., :D], out[..., D:])

        return out_complex
