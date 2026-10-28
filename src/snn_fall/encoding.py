"""Spike encoders. All take z-scored inputs and return spike trains shaped [T, B, C]."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ttfs(strength: torch.Tensor, t_max: int, threshold: float) -> torch.Tensor:
    """Time-to-first-spike: stronger values (clamped to [0, 1]) fire earlier; one spike max.

    strength: [B, C] -> spikes [T, B, C]. Values <= threshold never fire.
    """
    s = strength.clamp(0, 1)
    t = ((1 - s) * (t_max - 1)).long()
    spikes = F.one_hot(t, t_max).float() * (s > threshold).unsqueeze(-1).float()
    return spikes.permute(2, 0, 1).contiguous()


class SegmentedTTFSEncoder(nn.Module):
    """Split the window into equal segments and TTFS-encode each segment's summary value.

    [B, L, C] -> [T, B, segments * C]; channel index = segment * C + channel.
    """

    def __init__(self, t_max: int = 20, num_segments: int = 4, threshold: float = 0.05, reduce: str = "max"):
        super().__init__()
        if reduce not in ("max", "mean"):
            raise ValueError(f"reduce must be 'max' or 'mean', got {reduce!r}")
        self.t_max, self.num_segments, self.threshold, self.reduce = t_max, num_segments, threshold, reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        seg_len = L // self.num_segments
        segs = x[:, : seg_len * self.num_segments].reshape(B, self.num_segments, seg_len, C)
        strength = segs.amax(dim=2) if self.reduce == "max" else segs.mean(dim=2)  # [B, S, C]
        return _ttfs(strength.reshape(B, -1), self.t_max, self.threshold)


class FS1TTFSEncoder(nn.Module):
    """TTFS encoding of the FS1 statistical features: [B, F] -> [T, B, F]."""

    def __init__(self, t_max: int = 20, threshold: float = 0.01):
        super().__init__()
        self.t_max, self.threshold = t_max, threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ttfs(x, self.t_max, self.threshold)


class LearnableLevelCrossingEncoder(nn.Module):
    """Level-crossing encoder with one learnable threshold per channel.

    The window is divided into ``t_max`` bins; bin t spikes on channel c if any
    sample-to-sample change |x[i+1] - x[i]| in that bin exceeds threshold[c].
    Forward pass emits hard 0/1 spikes; the backward pass uses a sigmoid
    soft-OR surrogate so the thresholds receive gradients (straight-through).
    """

    def __init__(self, channels: int, t_max: int = 20, init_threshold: float = 1.0, slope: float = 8.0):
        super().__init__()
        self.t_max, self.slope = t_max, slope
        self.threshold = nn.Parameter(torch.full((channels,), float(init_threshold)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        delta = (x[:, 1:] - x[:, :-1]).abs()                     # [B, L-1, C]
        p = torch.sigmoid(self.slope * (delta - self.threshold))  # per-sample spike probability
        bin_len = (L - 1) // self.t_max
        out = []
        for t in range(self.t_max):
            end = (t + 1) * bin_len if t < self.t_max - 1 else L - 1  # last bin takes the remainder
            p_any = 1 - torch.prod(1 - p[:, t * bin_len:end], dim=1)  # soft OR over the bin
            hard = (p_any > 0.5).float()
            out.append(p_any + (hard - p_any).detach())
        return torch.stack(out)
