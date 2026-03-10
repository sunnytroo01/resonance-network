"""
Oscillatory Block: Core layer of the Resonance Network v2.

Replaces dot-product attention with Kuramoto-style phase coupling that does
DUAL DUTY: (1) synchronize phases between related tokens, and (2) mix value
information through the same coupling weights.

Key equations:
    coupling_ij = softmax(Q_i^T K_j / sqrt(d))   [same as attention scores]
    phase_update_i = sum_j coupling_ij * sin(phase_j - phase_i)   [Kuramoto]
    value_mix_i = sum_j coupling_ij * V_j                         [content mixing]

The coupling weights serve as both phase synchronization AND content retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .complex_ops import (
    ComplexRMSNorm, ComplexDropout,
    complex_magnitude, complex_phase, make_complex,
)


class KuramotoCoupling(nn.Module):
    """
    Kuramoto coupling with value mixing.

    Computes coupling weights (like attention scores), then uses them for:
    1. Phase synchronization: sum_j K_ij * sin(phase_j - phase_i)
    2. Value retrieval: sum_j K_ij * V_j (content mixing between tokens)

    This is what makes tokens able to actually communicate content, not just
    synchronize their oscillation phases.
    """

    def __init__(self, dim: int, coupling_rank: int = 64, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.coupling_rank = coupling_rank
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        # Q/K for coupling weights (operates on real magnitudes)
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        # V for content mixing
        self.W_v = nn.Linear(dim, dim, bias=False)
        # Output projection (like attention output proj)
        self.W_o = nn.Linear(dim, dim, bias=False)

        # Coupling strength for phase component
        self.coupling_strength = nn.Parameter(torch.tensor(0.5))

    def forward(self, z: torch.Tensor, mask: torch.Tensor = None):
        """
        Returns:
            phase_delta: (B, S, D) real - phase update from Kuramoto coupling
            value_mix: (B, S, D) real - content mixed through coupling weights
        """
        B, S, D = z.shape
        H = self.num_heads
        hd = self.head_dim

        mag = complex_magnitude(z)  # (B, S, D)
        phase = complex_phase(z)    # (B, S, D)

        # Q, K, V projections from magnitudes
        q = self.W_q(mag).view(B, S, H, hd).transpose(1, 2)  # (B, H, S, hd)
        k = self.W_k(mag).view(B, S, H, hd).transpose(1, 2)
        v = self.W_v(mag).view(B, S, H, hd).transpose(1, 2)

        # Coupling weights (= attention scores)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
        weights = F.softmax(scores, dim=-1)  # (B, H, S, S)

        # === VALUE MIXING (content retrieval, like attention) ===
        value_out = torch.matmul(weights, v)  # (B, H, S, hd)
        value_out = value_out.transpose(1, 2).contiguous().view(B, S, D)
        value_mix = self.W_o(value_out)

        # === PHASE COUPLING (Kuramoto synchronization) ===
        phase_heads = phase.view(B, S, H, hd).transpose(1, 2)
        sin_ph = torch.sin(phase_heads)
        cos_ph = torch.cos(phase_heads)
        K_sin = torch.matmul(weights, sin_ph)
        K_cos = torch.matmul(weights, cos_ph)
        phase_update = cos_ph * K_sin - sin_ph * K_cos
        phase_update = phase_update.transpose(1, 2).contiguous().view(B, S, D)

        return self.coupling_strength * phase_update, value_mix


class OscillatoryBlock(nn.Module):
    """
    One layer of the Resonance Network.

    Combines:
    1. Kuramoto phase coupling + value mixing (replaces attention)
    2. Magnitude MLP (replaces FFN)
    3. Complex-valued residual connections + normalization
    """

    def __init__(
        self,
        dim: int,
        coupling_rank: int = 64,
        num_heads: int = 4,
        mag_expansion: int = 4,
        decay: float = 0.05,
        dropout: float = 0.1,
        dt: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.dt = dt

        # Natural frequencies per dimension
        self.omega = nn.Parameter(torch.randn(dim) * 0.01)

        # Kuramoto coupling + value mixing
        self.coupling = KuramotoCoupling(dim, coupling_rank, num_heads)

        # Pre-norm for coupling (on magnitudes)
        self.norm1 = ComplexRMSNorm(dim)

        # Magnitude FFN (SwiGLU-style, replaces transformer FFN)
        inner = dim * mag_expansion
        self.mag_gate = nn.Linear(dim, inner, bias=False)
        self.mag_up = nn.Linear(dim, inner, bias=False)
        self.mag_down = nn.Linear(inner, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)

        # Value mixing -> complex injection
        self.value_to_complex = nn.Linear(dim, dim * 2)  # -> real + imag

        # Dropout
        self.dropout = ComplexDropout(dropout)

    def forward(self, z: torch.Tensor, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z: complex (batch, seq_len, dim)
            x: real (batch, seq_len, dim) - original token embeddings
            mask: causal mask (seq_len, seq_len)
        Returns:
            z_new: complex (batch, seq_len, dim)
        """
        # Pre-norm
        z_normed = self.norm1(z)

        # Coupling: phase sync + value mixing
        phase_delta, value_mix = self.coupling(z_normed, mask=mask)

        # Phase update: natural frequency + coupling
        phase = complex_phase(z)
        new_phase = phase + self.dt * (self.omega + phase_delta)
        # Keep phase in [-pi, pi] to prevent precision loss in cos/sin with bfloat16
        new_phase = torch.remainder(new_phase + math.pi, 2 * math.pi) - math.pi

        # Inject value mix into complex state
        val_proj = self.value_to_complex(value_mix)
        val_complex = make_complex(val_proj[..., :self.dim], val_proj[..., self.dim:])

        # Update magnitude via FFN on current magnitudes
        mag = complex_magnitude(z)
        mag_normed = self.norm2(mag)
        mag_ffn = self.mag_down(F.silu(self.mag_gate(mag_normed)) * self.mag_up(mag_normed))
        new_mag = mag + mag_ffn
        new_mag = new_mag.clamp(-20.0, 20.0)  # prevent bfloat16 overflow
        new_mag = F.softplus(new_mag)  # keep positive

        # Reconstruct complex state
        z_new = make_complex(
            new_mag * torch.cos(new_phase),
            new_mag * torch.sin(new_phase),
        )

        # Add value injection + residual
        z_new = z + z_new - z.detach() + val_complex  # gradient-friendly residual
        z_new = self.dropout(z_new)

        return z_new
