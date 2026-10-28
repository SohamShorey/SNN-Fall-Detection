"""Unsupervised STDP-style pretraining of the convolutional layers in each pathway.

For a conv layer with presynaptic spikes ``pre`` (mapped to {-1, +1}) the
postsynaptic activity is ``post = tanh(conv(pre))``. The Hebbian correlation
<pre, post> (averaged over batch and time) gives the LTP (positive) and LTD
(negative) terms; a decay term proportional to <post><pre> keeps weights from
growing without bound. Each update is rescaled to a target fraction of the
layer's weight std, capped per step, and weights are clipped to [-clip, clip].
The same update is applied to every kernel tap.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .config import STDPConfig
from .model import SCNN


class STDPTrainer:
    def __init__(self, model: SCNN, cfg: STDPConfig):
        self.model, self.cfg = model, cfg

    @torch.no_grad()
    def update(self, conv: nn.Conv1d, pre_TBC: torch.Tensor, layer: str) -> float:
        c = self.cfg
        T = pre_TBC.size(0)
        pre = pre_TBC.permute(1, 2, 0) * 2.0 - 1.0              # [B, C_in, T] in {-1, +1}
        post = torch.tanh(conv(pre))
        if post.size(2) != T:
            post = F.adaptive_avg_pool1d(post, T)               # [B, C_out, T]

        hebb = torch.einsum("bct,bkt->bkc", pre, post).mean(0) / T          # [C_out, C_in]
        decay = c.decay * torch.outer(post.mean((0, 2)), pre.mean((0, 2)))
        core = c.a_plus * hebb.clamp(min=0) - c.a_minus * hebb.clamp(max=0) - decay

        w_std = conv.weight.std().item() + 1e-8
        eta = c.lr * max(1.0, c.target_rel[layer] * w_std / (core.abs().mean().item() + 1e-12))
        max_step = c.max_step_frac[layer] * w_std
        dw = (eta * core).clamp(-max_step, max_step)

        conv.weight.add_(dw.unsqueeze(2)).clamp_(-c.clip, c.clip)
        return dw.abs().mean().item() / w_std

    @torch.no_grad()
    def run(self, loader, device: torch.device) -> dict[str, float]:
        """Run STDP over the loader; returns the mean relative update size per layer."""
        self.model.eval()
        totals: dict[str, float] = {}
        n = 0
        for epoch in range(self.cfg.epochs):
            for x, fs1, _ in tqdm(loader, desc=f"STDP {epoch + 1}/{self.cfg.epochs}", leave=False):
                spikes = self.model.encode(x.to(device), fs1.to(device))
                for p in self.model.path_names:
                    path = self.model.paths[p]
                    rel1 = self.update(path.conv1, spikes[p], "conv1")
                    # conv2's presynaptic spikes come from the freshly updated first stage
                    rel2 = self.update(path.conv2, path.stage1(spikes[p]), "conv2")
                    totals[f"{p}.conv1"] = totals.get(f"{p}.conv1", 0.0) + rel1
                    totals[f"{p}.conv2"] = totals.get(f"{p}.conv2", 0.0) + rel2
                n += 1
        return {k: v / max(n, 1) for k, v in totals.items()}
