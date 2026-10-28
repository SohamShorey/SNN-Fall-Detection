"""Three-pathway spiking convolutional network (SCNN) for binary fall detection.

    raw window ─┬─ Segmented TTFS ─ ConvPath ─ PathHead ─┐
                └─ Level crossing ─ ConvPath ─ PathHead ─┼─ concat ─ Classifier ─ fall / ADL
    FS1 feats ─── FS1 TTFS ──────── ConvPath ─ PathHead ─┘

Every spiking layer is a leaky integrate-and-fire (LIF) neuron simulated over
``t_max`` steps; the class logits are the output layer's final membrane potential.
"""
from __future__ import annotations

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate

from .config import Config
from .encoding import FS1TTFSEncoder, LearnableLevelCrossingEncoder, SegmentedTTFSEncoder

PATHS = ("ttfs", "lc", "fs1")


def _lif(beta: float, output: bool = False) -> snn.Leaky:
    return snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), output=output)


def run_lif(lif: snn.Leaky, currents_TBC: torch.Tensor) -> torch.Tensor:
    """Drive a LIF layer with a [T, B, C] input current and return its [T, B, C] spikes."""
    mem = lif.init_leaky()
    spikes = []
    for current in currents_TBC:
        spk, mem = lif(current, mem)
        spikes.append(spk)
    return torch.stack(spikes)


class SpikingConvPath(nn.Module):
    """Two (Conv1d over time -> GroupNorm -> LIF) stages: [T, B, C_in] -> [T, B, C]."""

    def __init__(self, in_ch: int, ch: int, k1: int, k2: int, gn_groups: int, beta: float):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, ch, k1, padding=k1 // 2)
        self.gn1 = nn.GroupNorm(gn_groups, ch)
        self.lif1 = _lif(beta)
        self.conv2 = nn.Conv1d(ch, ch, k2, padding=k2 // 2)
        self.gn2 = nn.GroupNorm(gn_groups, ch)
        self.lif2 = _lif(beta)

    @staticmethod
    def _stage(conv, gn, lif, spk_TBC):
        current = gn(conv(spk_TBC.permute(1, 2, 0)))           # [B, C, T]
        return run_lif(lif, current.permute(2, 0, 1))          # [T, B, C]

    def stage1(self, spk_TBC: torch.Tensor) -> torch.Tensor:
        return self._stage(self.conv1, self.gn1, self.lif1, spk_TBC)

    def forward(self, spk_TBC: torch.Tensor) -> torch.Tensor:
        return self._stage(self.conv2, self.gn2, self.lif2, self.stage1(spk_TBC))


class SpikingLinear(nn.Module):
    """Linear projection followed by a LIF neuron, applied at every time step."""

    def __init__(self, in_dim: int, out_dim: int, beta: float, output: bool = False):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.lif = _lif(beta, output=output)

    def forward(self, x_TBC: torch.Tensor):
        mem = self.lif.init_leaky()
        spikes = []
        for x in x_TBC:
            spk, mem = self.lif(self.fc(x), mem)
            spikes.append(spk)
        return torch.stack(spikes), mem


class SCNN(nn.Module):
    def __init__(self, cfg: Config, n_channels: int, n_fs1: int, num_classes: int = 2):
        super().__init__()
        e, m = cfg.encoding, cfg.model
        self.encoders = nn.ModuleDict({
            "ttfs": SegmentedTTFSEncoder(e.t_max, e.ttfs_segments, e.ttfs_threshold, e.ttfs_reduce),
            "lc": LearnableLevelCrossingEncoder(n_channels, e.t_max, e.lc_init_threshold, e.lc_surrogate_slope),
            "fs1": FS1TTFSEncoder(e.t_max, e.fs1_threshold),
        })
        in_ch = {"ttfs": n_channels * e.ttfs_segments, "lc": n_channels, "fs1": n_fs1}
        self.paths = nn.ModuleDict({
            p: SpikingConvPath(in_ch[p], m.conv_channels, m.kernel1, m.kernel2, m.gn_groups, m.beta) for p in PATHS
        })
        self.heads = nn.ModuleDict({p: SpikingLinear(m.conv_channels, m.head_dim, m.beta) for p in PATHS})
        self.hidden = SpikingLinear(len(PATHS) * m.head_dim, m.hidden_dim, m.beta)
        self.out = SpikingLinear(m.hidden_dim, num_classes, m.beta, output=True)

    def encode(self, x: torch.Tensor, fs1: torch.Tensor) -> dict[str, torch.Tensor]:
        """Raw window [B, L, C] and FS1 features [B, F] -> spike trains per pathway."""
        return {"ttfs": self.encoders["ttfs"](x), "lc": self.encoders["lc"](x), "fs1": self.encoders["fs1"](fs1)}

    def forward(self, x: torch.Tensor, fs1: torch.Tensor):
        """Returns (output spikes [T, B, classes], final output membrane [B, classes])."""
        spikes = self.encode(x, fs1)
        fused = torch.cat([self.heads[p](self.paths[p](spikes[p]))[0] for p in PATHS], dim=2)
        hidden, _ = self.hidden(fused)
        spk_out, mem_out = self.out(hidden)
        return spk_out, mem_out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
