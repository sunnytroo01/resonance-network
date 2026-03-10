"""
Complex-valued neural network operations for the Resonance Network.

Implements complex-valued linear layers, RMSNorm (magnitude-normalized, phase-preserved),
and utility functions for converting between real and complex representations.
"""

import torch
import torch.nn as nn
import math


def make_complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    """Create complex tensor, auto-casting bfloat16 to float32 (torch.complex requires float/double)."""
    if real.dtype == torch.bfloat16:
        real = real.float()
    if imag.dtype == torch.bfloat16:
        imag = imag.float()
    return torch.complex(real, imag)


def real_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert a real tensor of shape (..., 2*d) to complex tensor of shape (..., d)."""
    d = x.shape[-1] // 2
    return make_complex(x[..., :d], x[..., d:])


def complex_to_real(z: torch.Tensor) -> torch.Tensor:
    """Convert a complex tensor of shape (..., d) to real tensor of shape (..., 2*d)."""
    return torch.cat([z.real, z.imag], dim=-1)


def complex_magnitude(z: torch.Tensor) -> torch.Tensor:
    """Compute magnitude of complex tensor, numerically stable."""
    return torch.sqrt(z.real ** 2 + z.imag ** 2 + 1e-8)


def complex_phase(z: torch.Tensor) -> torch.Tensor:
    """Compute phase angle of complex tensor."""
    return torch.atan2(z.imag, z.real)


class ComplexRMSNorm(nn.Module):
    """
    RMSNorm adapted for complex-valued tensors.
    Normalizes the magnitude while preserving the phase.

    For complex z = r * exp(i*theta):
        - Compute RMS of magnitudes: rms = sqrt(mean(|z|^2))
        - Normalize: z_norm = z / rms * scale
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z is complex64, shape (..., dim)
        mag_sq = z.real ** 2 + z.imag ** 2  # |z|^2
        rms = torch.sqrt(mag_sq.mean(dim=-1, keepdim=True) + self.eps)
        z_normalized = z / rms
        # Scale is real-valued, applied to both real and imag parts
        return z_normalized * self.scale.unsqueeze(0).unsqueeze(0)


class ComplexLinear(nn.Module):
    """
    Linear layer for complex-valued tensors using real-valued parameterization.

    Computes: (W_real + i*W_imag) @ (z_real + i*z_imag) + (b_real + i*b_imag)

    Using real parameterization for better stability and optimizer compatibility.
    The real-valued decomposition:
        out_real = W_real @ z_real - W_imag @ z_imag + b_real
        out_imag = W_real @ z_imag + W_imag @ z_real + b_imag
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Real-valued parameterization
        self.W_real = nn.Parameter(torch.empty(out_features, in_features))
        self.W_imag = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.b_real = nn.Parameter(torch.zeros(out_features))
            self.b_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('b_real', None)
            self.register_parameter('b_imag', None)

        self._init_weights()

    def _init_weights(self):
        # Glorot initialization scaled for complex (each component gets 1/sqrt(2) factor)
        scale = 1.0 / math.sqrt(2.0)
        nn.init.kaiming_uniform_(self.W_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_imag, a=math.sqrt(5))
        self.W_real.data *= scale
        self.W_imag.data *= scale

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z is complex64
        z_real = z.real
        z_imag = z.imag

        out_real = nn.functional.linear(z_real, self.W_real) - nn.functional.linear(z_imag, self.W_imag)
        out_imag = nn.functional.linear(z_real, self.W_imag) + nn.functional.linear(z_imag, self.W_real)

        if self.b_real is not None:
            out_real = out_real + self.b_real
            out_imag = out_imag + self.b_imag

        return make_complex(out_real, out_imag)


class ComplexGELU(nn.Module):
    """
    GELU activation for complex tensors.
    Applies GELU separately to magnitude, preserves phase.

    z_out = GELU(|z|) * exp(i * angle(z))
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = complex_magnitude(z)
        phase = complex_phase(z)
        activated_mag = nn.functional.gelu(mag)
        return make_complex(
            activated_mag * torch.cos(phase),
            activated_mag * torch.sin(phase),
        )


class ComplexDropout(nn.Module):
    """Dropout for complex tensors - drops both real and imag together."""

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return z
        # Create real-valued mask, apply to both components
        mask = torch.bernoulli(torch.full_like(z.real, 1 - self.p)) / (1 - self.p)
        return make_complex(z.real * mask, z.imag * mask)


class RotaryEmbeddingComplex(nn.Module):
    """
    Rotary positional embedding adapted for complex-valued space.

    In standard RoPE, positions are encoded as rotations in 2D subspaces.
    For complex tensors, this is natural: multiply by exp(i * m * theta_k)
    where m is position and theta_k = 1/10000^(2k/d).
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, dtype=torch.float32) / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        angles = torch.outer(positions, self.inv_freq)  # (seq_len, dim)
        # Complex rotation factors: exp(i * angle)
        rotation = torch.complex(torch.cos(angles), torch.sin(angles))
        self.register_buffer('rotation', rotation)

    def forward(self, z: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Apply rotary embedding to complex tensor.
        z: (batch, seq_len, dim) complex tensor
        """
        seq_len = z.shape[1]
        if offset + seq_len > self.rotation.shape[0]:
            self._build_cache(offset + seq_len)
        rot = self.rotation[offset:offset + seq_len].unsqueeze(0)  # (1, seq_len, dim)
        return z * rot
