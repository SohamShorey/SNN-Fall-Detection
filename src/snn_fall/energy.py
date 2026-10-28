"""Spike counts, operation counts and estimated energy per inference.

SNN: a layer whose input is a binary spike train only does work for input spikes, so its
synaptic operations are accumulates (AC): one per (input spike, weight it reaches).
Dense work that remains in the SNN (bias adds, GroupNorm, LIF membrane updates) is counted
separately.

Equivalent ANN: the same layers and shapes with real-valued activations (e.g. ReLU)
evaluated over the same t_max positions, so every weight is used at every position:
multiply-accumulates (MAC) equal to the dense count of each layer.

Energy = ops x energy-per-op (default: 45 nm CMOS, 32-bit float, Horowitz, ISSCC 2014:
MAC 4.6 pJ, AC 0.9 pJ). Memory access, encoding and control overheads are not modelled,
so the figures are a first-order, hardware-agnostic comparison, as is common in SNN papers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EnergyConfig
from .model import PATHS, SCNN, run_lif

GN_MACS_PER_ELEMENT = 2  # normalise (subtract mean, scale) + affine; statistics are not counted
LIF_MACS_PER_UPDATE = 1  # mem = beta * mem + input


def _conv_pairs(spk_TBC: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    """Per sample, the number of (nonzero input, kernel tap) pairs a Conv1d sees: [B]."""
    x = (spk_TBC > 0.5).float().permute(1, 2, 0)                     # [B, C_in, T]
    ones = torch.ones(1, conv.in_channels, conv.kernel_size[0], device=x.device)
    return F.conv1d(x, ones, padding=conv.padding[0]).sum(dim=(1, 2))


def _dense_conv_pairs(conv: nn.Conv1d, T: int, device) -> float:
    return _conv_pairs(torch.ones(T, 1, conv.in_channels, device=device), conv).item()


class _Tally:
    def __init__(self):
        self.layers: dict[str, dict[str, float]] = {}

    def add(self, layer: str, **values: float) -> None:
        row = self.layers.setdefault(layer, {})
        for k, v in values.items():
            row[k] = row.get(k, 0.0) + float(v)


@torch.no_grad()
def count_operations(model: SCNN, loader, device: torch.device, cfg_energy: EnergyConfig) -> dict:
    """Average spikes, operations and energy per inference over ``loader`` for the SNN and
    the architecture-matched ANN."""
    model.eval()
    snn, n = _Tally(), 0
    for x, fs1, _ in loader:
        x, fs1 = x.to(device), fs1.to(device)
        B = x.size(0)
        n += B
        spikes = model.encode(x, fs1)
        heads = []
        for p in PATHS:
            path = model.paths[p]
            s0 = spikes[p]
            snn.add(f"input.{p}", spikes=(s0 > 0.5).sum())
            s1 = path.stage1(s0)
            s2 = path._stage(path.conv2, path.gn2, path.lif2, s1)
            for name, conv, pre, post in (("conv1", path.conv1, s0, s1), ("conv2", path.conv2, s1, s2)):
                snn.add(f"{p}.{name}", synaptic_ac=_conv_pairs(pre, conv).sum() * conv.out_channels,
                        spikes_out=post.sum(), neurons=post.size(2) * post.size(0) * B)
            h, _ = model.heads[p](s2)
            snn.add(f"{p}.head", synaptic_ac=s2.sum() * model.heads[p].fc.out_features,
                    spikes_out=h.sum(), neurons=h.size(2) * h.size(0) * B)
            heads.append(h)
        fused = torch.cat(heads, dim=2)
        hidden, _ = model.hidden(fused)
        out, _ = model.out(hidden)
        snn.add("hidden", synaptic_ac=fused.sum() * model.hidden.fc.out_features,
                spikes_out=hidden.sum(), neurons=hidden.size(2) * hidden.size(0) * B)
        snn.add("output", synaptic_ac=hidden.sum() * model.out.fc.out_features,
                spikes_out=out.sum(), neurons=out.size(2) * out.size(0) * B)

    T = model.encoders["ttfs"].t_max
    layers, totals = {}, {"snn_synaptic_ac": 0.0, "snn_dense_mac": 0.0, "snn_dense_ac": 0.0,
                          "ann_synaptic_mac": 0.0, "ann_dense_mac": 0.0, "ann_dense_ac": 0.0}
    for name, row in snn.layers.items():
        avg = {k: v / n for k, v in row.items()}
        if name.startswith("input."):
            layers[name] = {"spikes": avg["spikes"]}
            continue
        module = _module(model, name)
        units = avg["neurons"]                                         # neuron-timesteps per inference
        if isinstance(module, nn.Conv1d):
            ann_mac = _dense_conv_pairs(module, T, device) * module.out_channels
            gn = GN_MACS_PER_ELEMENT * units
        else:
            ann_mac = module.in_features * module.out_features * T
            gn = 0.0
        layers[name] = {
            "spikes_out": avg["spikes_out"],
            "firing_rate": avg["spikes_out"] / units,
            "snn_synaptic_ac": avg["synaptic_ac"],
            "ann_mac": ann_mac,
            "synaptic_sparsity": 1.0 - avg["synaptic_ac"] / ann_mac,
        }
        totals["snn_synaptic_ac"] += avg["synaptic_ac"]
        totals["snn_dense_mac"] += gn + LIF_MACS_PER_UPDATE * units
        totals["snn_dense_ac"] += units                                 # one bias add per neuron-timestep
        totals["ann_synaptic_mac"] += ann_mac
        totals["ann_dense_mac"] += gn
        totals["ann_dense_ac"] += units                                 # bias adds (ReLU not counted)

    e_mac, e_ac = cfg_energy.e_mac_pj, cfg_energy.e_ac_pj
    snn_syn_nj = totals["snn_synaptic_ac"] * e_ac / 1e3
    snn_total_nj = snn_syn_nj + (totals["snn_dense_mac"] * e_mac + totals["snn_dense_ac"] * e_ac) / 1e3
    ann_syn_nj = totals["ann_synaptic_mac"] * e_mac / 1e3
    ann_total_nj = ann_syn_nj + (totals["ann_dense_mac"] * e_mac + totals["ann_dense_ac"] * e_ac) / 1e3
    return {
        "assumptions": {
            "e_mac_pj": e_mac, "e_ac_pj": e_ac, "t_max": T,
            "gn_macs_per_element": GN_MACS_PER_ELEMENT, "lif_macs_per_update": LIF_MACS_PER_UPDATE,
            "excluded": "memory access, spike encoding, GroupNorm statistics, control logic",
            "ann": "same layers and shapes, real-valued activations, evaluated at all t_max positions",
        },
        "per_inference": {
            "input_spikes": sum(v["spikes"] for k, v in layers.items() if k.startswith("input.")),
            "network_spikes": sum(v["spikes_out"] for k, v in layers.items() if not k.startswith("input.")),
            **totals,
            "snn_synaptic_energy_nj": snn_syn_nj,
            "snn_total_energy_nj": snn_total_nj,
            "ann_synaptic_energy_nj": ann_syn_nj,
            "ann_total_energy_nj": ann_total_nj,
            "synaptic_energy_ratio_ann_over_snn": ann_syn_nj / snn_syn_nj if snn_syn_nj else float("inf"),
            "total_energy_ratio_ann_over_snn": ann_total_nj / snn_total_nj,
        },
        "per_layer": layers,
        "inferences": n,
    }


def _module(model: SCNN, name: str) -> nn.Module:
    if name in ("hidden", "output"):
        return (model.hidden if name == "hidden" else model.out).fc
    path, layer = name.split(".")
    return model.heads[path].fc if layer == "head" else getattr(model.paths[path], layer)
