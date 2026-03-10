"""
Resonance Network v2: Full sequence model for language modeling.

Architecture (v2 - layer stack, no DEQ):
1. Token embedding + rotary positional encoding (complex space)
2. Input projection to complex oscillatory space
3. N stacked ResonanceLayer blocks (each with own params):
   - Kuramoto coupling + value mixing (replaces attention)
   - Hopfield associative memory (optional learned patterns)
   - Magnitude FFN (SwiGLU)
4. Output projection: full complex state (real+imag) -> logits

Key difference from v1: no equilibrium solver. Each layer has its own
parameters, like a transformer. Gradient flows directly through the stack.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .complex_ops import (
    ComplexRMSNorm, RotaryEmbeddingComplex, complex_magnitude, make_complex,
)
from .oscillatory_block import OscillatoryBlock
from .hopfield_memory import HopfieldMemory


class ResonanceLayer(nn.Module):
    """
    One layer of the Resonance Network v2.

    Combines:
    1. OscillatoryBlock (Kuramoto coupling + value mixing + magnitude FFN)
    2. HopfieldMemory (learned associative memory for cross-pattern retrieval)
    3. Gated residual connection for memory contribution
    """

    def __init__(
        self,
        dim: int,
        coupling_rank: int = 64,
        num_heads: int = 4,
        num_stored_patterns: int = 192,
        hopfield_steps: int = 2,
        mag_expansion: int = 2,
        decay: float = 0.1,
        dropout: float = 0.1,
        dt: float = 0.1,
        use_sparsemax: bool = True,
    ):
        super().__init__()
        self.oscillatory = OscillatoryBlock(
            dim=dim,
            coupling_rank=coupling_rank,
            num_heads=num_heads,
            mag_expansion=mag_expansion,
            decay=decay,
            dropout=dropout,
            dt=dt,
        )
        self.memory = HopfieldMemory(
            dim=dim,
            num_stored_patterns=num_stored_patterns,
            num_heads=num_heads,
            retrieval_steps=hopfield_steps,
            use_sparsemax=use_sparsemax,
            dropout=dropout,
        )
        self.memory_gate = nn.Parameter(torch.tensor(0.1))
        self.norm = ComplexRMSNorm(dim)

    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Oscillatory update (Kuramoto coupling + value mixing + FFN)
        z = self.oscillatory(z, x, mask=mask)

        # Hopfield memory retrieval (gated)
        mem_update = self.memory(z, mask=mask)
        gate = torch.sigmoid(self.memory_gate)
        z = z + gate * mem_update

        # Normalize
        z = self.norm(z)
        return z


class ResonanceNetwork(nn.Module):
    """
    Full Resonance Network v2 for language modeling.

    Replaces v1's DEQ solver with a standard N-layer stack.
    Each layer has its own parameters (no weight tying).

    Novel aspects preserved:
    - Complex-valued hidden states (amplitude + phase)
    - Kuramoto phase coupling for inter-token synchronization
    - Value mixing through coupling weights (dual-duty)
    - Hopfield associative memory with learned patterns
    - Rotary positional embedding in complex space
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 384,
        n_layers: int = 6,
        max_seq_len: int = 256,
        coupling_rank: int = 48,
        num_heads: int = 6,
        num_stored_patterns: int = 192,
        hopfield_steps: int = 2,
        mag_expansion: int = 2,
        decay: float = 0.1,
        dropout: float = 0.1,
        dt: float = 0.1,
        use_sparsemax: bool = True,
        stability_weight: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.stability_weight = stability_weight

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, dim)

        # Rotary positional embedding for complex space
        self.rope = RotaryEmbeddingComplex(dim, max_seq_len)

        # Input projection: real embeddings -> complex oscillatory space
        self.input_proj = nn.Linear(dim, dim * 2)  # -> real + imag parts

        # Layer stack (each layer has own parameters)
        self.layers = nn.ModuleList([
            ResonanceLayer(
                dim=dim,
                coupling_rank=coupling_rank,
                num_heads=num_heads,
                num_stored_patterns=num_stored_patterns,
                hopfield_steps=hopfield_steps,
                mag_expansion=mag_expansion,
                decay=decay,
                dropout=dropout,
                dt=dt,
                use_sparsemax=use_sparsemax,
            )
            for _ in range(n_layers)
        ])

        # Output head: complex state -> logits
        # Uses BOTH magnitude AND phase (real+imag) so coupling can influence output
        self.output_norm = ComplexRMSNorm(dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),  # real + imag concatenated
            nn.GELU(),
            nn.Linear(dim, vocab_size),
        )

        # Initialize
        self._init_weights()

        # Causal mask buffer
        causal = torch.triu(torch.full((max_seq_len, max_seq_len), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', causal)

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        for m in self.output_mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_oscillatory_state(self, x: torch.Tensor) -> torch.Tensor:
        """Project real-valued embeddings into complex space with rotary encoding."""
        B, S, D = x.shape
        proj = self.input_proj(x)  # (B, S, 2*D)
        z = make_complex(proj[..., :D], proj[..., D:])
        z = self.rope(z)
        return z

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) token indices
            targets: (batch, seq_len) target token indices for loss computation

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar loss (if targets provided)
            info: dict with training info
        """
        B, S = input_ids.shape

        # Token embeddings (real-valued)
        x = self.token_embedding(input_ids)  # (B, S, dim)

        # Initialize oscillatory state
        z = self._init_oscillatory_state(x)

        # Causal mask for this sequence length
        mask = self.causal_mask[:S, :S]

        # Pass through layer stack
        for layer in self.layers:
            z = layer(z, x, mask=mask)

        # Decode: use full complex state (real + imag) for logits
        z_normed = self.output_norm(z)
        z_real_imag = torch.cat([z_normed.real, z_normed.imag], dim=-1)  # (B, S, 2*dim)
        logits = self.output_mlp(z_real_imag)  # (B, S, vocab_size)

        # Compute loss if targets provided
        loss = None
        info = {'n_layers': self.n_layers}

        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

            # Stability penalty: penalize large magnitudes
            stability_loss = torch.tensor(0.0, device=input_ids.device)
            if self.training and self.stability_weight > 0:
                mag = complex_magnitude(z)
                stability_loss = (mag ** 2).mean()

            loss = ce_loss + self.stability_weight * stability_loss

            info['ce_loss'] = ce_loss.item()
            info['stability_loss'] = stability_loss.item()
            info['total_loss'] = loss.item()

        return logits, loss, info

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count parameters, optionally excluding embeddings."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params
