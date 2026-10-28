"""Three-pathway spiking convolutional network (SCNN) for binary fall detection.

    raw window ─┬─ Segmented TTFS ─ ConvPath ─ PathHead ─┐
                └─ Level crossing ─ ConvPath ─ PathHead ─┼─ concat ─ Classifier ─ fall / ADL
    FS1 feats ─── FS1 TTFS ──────── ConvPath ─ PathHead ─┘

Every spiking layer is a leaky integrate-and-fire (LIF) neuron simulated over
``t_max`` steps; the class logits are the output layer's final membrane potential.

With ``model.kind: ann`` the same network is built as an "ANN twin": identical layers,
shapes and spike-encoded inputs, but each LIF neuron is replaced by a ReLU (no
state, real-valued activations) and the logits are the output averaged over the
``t_max`` positions. It isolates the effect of spiking neurons.
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
    """Two (Conv1d over time -> GroupNorm -> LIF) stages: [T, B, C_in] -> [T, B, C].

    spiking=False replaces each LIF with a ReLU (ANN twin); gn_groups=0 drops the GroupNorm.
    """

    def __init__(self, in_ch: int, ch: int, k1: int, k2: int, gn_groups: int, beta: float, spiking: bool = True):
        super().__init__()
        norm = (lambda: nn.GroupNorm(gn_groups, ch)) if gn_groups > 0 else nn.Identity  # gn_groups=0: no norm
        self.conv1 = nn.Conv1d(in_ch, ch, k1, padding=k1 // 2)
        self.gn1 = norm()
        self.lif1 = _lif(beta) if spiking else nn.ReLU()
        self.conv2 = nn.Conv1d(ch, ch, k2, padding=k2 // 2)
        self.gn2 = norm()
        self.lif2 = _lif(beta) if spiking else nn.ReLU()

    @staticmethod
    def _stage(conv, gn, act, spk_TBC):
        current = gn(conv(spk_TBC.permute(1, 2, 0))).permute(2, 0, 1)  # [T, B, C]
        return run_lif(act, current) if isinstance(act, snn.Leaky) else act(current)

    def stage1(self, spk_TBC: torch.Tensor) -> torch.Tensor:
        return self._stage(self.conv1, self.gn1, self.lif1, spk_TBC)

    def forward(self, spk_TBC: torch.Tensor) -> torch.Tensor:
        return self._stage(self.conv2, self.gn2, self.lif2, self.stage1(spk_TBC))


class SpikingLinear(nn.Module):
    """Linear projection followed by a LIF neuron, applied at every time step.

    Returns (spikes [T, B, out], final membrane [B, out]). With spiking=False (ANN twin)
    the LIF is replaced by a ReLU (identity for the output layer) and the second value is
    the activation averaged over time.
    """

    def __init__(self, in_dim: int, out_dim: int, beta: float, output: bool = False, spiking: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.lif = _lif(beta, output=output) if spiking else (nn.Identity() if output else nn.ReLU())

    def forward(self, x_TBC: torch.Tensor):
        if not isinstance(self.lif, snn.Leaky):
            y = self.lif(self.fc(x_TBC))
            return y, y.mean(0)
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
        self.spiking = m.kind == "scnn"
        s = self.spiking
        self.encoders = nn.ModuleDict({
            "ttfs": SegmentedTTFSEncoder(e.t_max, e.ttfs_segments, e.ttfs_threshold, e.ttfs_reduce),
            "lc": LearnableLevelCrossingEncoder(n_channels, e.t_max, e.lc_init_threshold, e.lc_surrogate_slope),
            "fs1": FS1TTFSEncoder(e.t_max, e.fs1_threshold),
        })
        in_ch = {"ttfs": n_channels * e.ttfs_segments, "lc": n_channels, "fs1": n_fs1}
        self.paths = nn.ModuleDict({
            p: SpikingConvPath(in_ch[p], m.conv_channels, m.kernel1, m.kernel2, m.gn_groups if m.norm == "group" else 0, m.beta, s)
            for p in PATHS
        })
        self.heads = nn.ModuleDict({p: SpikingLinear(m.conv_channels, m.head_dim, m.beta, spiking=s) for p in PATHS})
        self.hidden = SpikingLinear(len(PATHS) * m.head_dim, m.hidden_dim, m.beta, spiking=s)
        self.out = SpikingLinear(m.hidden_dim, num_classes, m.beta, output=True, spiking=s)

    def encode(self, x: torch.Tensor, fs1: torch.Tensor) -> dict[str, torch.Tensor]:
        """Raw window [B, L, C] and FS1 features [B, F] -> spike trains per pathway."""
        return {"ttfs": self.encoders["ttfs"](x), "lc": self.encoders["lc"](x), "fs1": self.encoders["fs1"](fs1)}

    def forward(self, x: torch.Tensor, fs1: torch.Tensor):
        """Returns (output spikes [T, B, classes], final output membrane [B, classes]);
        for the ANN twin, (output activations [T, B, classes], their time average)."""
        spikes = self.encode(x, fs1)
        fused = torch.cat([self.heads[p](self.paths[p](spikes[p]))[0] for p in PATHS], dim=2)
        hidden, _ = self.hidden(fused)
        spk_out, mem_out = self.out(hidden)
        return spk_out, mem_out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
