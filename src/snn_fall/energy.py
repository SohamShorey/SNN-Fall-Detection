"""Spike counts, operation counts and estimated energy per inference.

SNN: a layer whose input is a binary spike train only does work for input spikes, so its
synaptic operations are accumulates (AC): one per (input spike, weight it reaches).
Dense work that remains in the SNN (bias adds, GroupNorm, LIF membrane updates) is counted
separately.

Equivalent ANN: the same layers and shapes with real-valued activations (e.g. ReLU)
evaluated over the same t_max positions, so every weight is used at every position:
multiply-accumulates (MAC) equal to the dense count of each layer.

Energy = ops x energy-per-op (default: 45 nm CMOS, 32-bit float, Horowitz, ISSCC 2014:
MAC 4.6 pJ, AC 0.9 pJ; comparisons are charged as AC). The cost of encoding a window into
spikes is estimated separately (encoding_energy_nj). Memory access and control overheads are
not modelled, so the figures are a first-order, hardware-agnostic comparison, as is common in
SNN papers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EnergyConfig
from .model import SCNN, run_lif

GN_MACS_PER_ELEMENT = 2  # normalise (subtract mean, scale) + affine; statistics are not counted
LIF_MACS_PER_UPDATE = 1  # mem = beta * mem + input


def _conv_pairs(spk_TBC: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    """Per sample, the number of (nonzero input, kernel tap) pairs a Conv1d sees: [B]."""
    x = (spk_TBC > 0.5).float().permute(1, 2, 0)                     # [B, C_in, T]
    ones = torch.ones(1, conv.in_channels, conv.kernel_size[0], device=x.device)
    return F.conv1d(x, ones, padding=conv.padding[0]).sum(dim=(1, 2))


def _dense_conv_pairs(conv: nn.Conv1d, T: int, device) -> float:
    return _conv_pairs(torch.ones(T, 1, conv.in_channels, device=device), conv).item()


def _has_gn(model: SCNN) -> bool:
    return isinstance(model.paths[model.path_names[0]].gn1, nn.GroupNorm)


def encoding_cost(window_len: int, channels: int, n_fs1: int, segments: int, cfg_energy: EnergyConfig,
                  paths: tuple[str, ...] = ("ttfs", "lc", "fs1")) -> dict:
    """Approximate operations to turn one raw window into the spike trains of the active pathways.

    FS1 statistics: mean/variance accumulations (MAC) and running max/min (compare) per sample
    and channel; segmented TTFS: running max per sample and channel; level crossing: one
    subtraction (AC) and one threshold compare per sample and channel; TTFS time conversion:
    one MAC per encoded value.
    """
    L, C = window_len, channels
    mac = ac = cmp = 0
    if "fs1" in paths:
        mac += 2 * L * C + n_fs1
        cmp += 2 * L * C
    if "ttfs" in paths:
        mac += segments * C
        cmp += L * C
    if "lc" in paths:
        ac += L * C
        cmp += L * C
    return {"mac": mac, "ac": ac, "comparisons": cmp,
            "energy_nj": (mac * cfg_energy.e_mac_pj + (ac + cmp) * cfg_energy.e_ac_pj) / 1e3}


class _Tally:
    def __init__(self):
        self.layers: dict[str, dict[str, float]] = {}

    def add(self, layer: str, **values: float) -> None:
        row = self.layers.setdefault(layer, {})
        for k, v in values.items():
            row[k] = row.get(k, 0.0) + float(v)


def count_dense_operations(model: SCNN, device: torch.device, cfg_energy: EnergyConfig,
                           encoding: dict | None = None) -> dict:
    """Operations and energy per inference of the ANN twin (every weight used at every position)."""
    T = model.encoders["ttfs"].t_max
    gn_per = GN_MACS_PER_ELEMENT if _has_gn(model) else 0
    mac, gn, units = 0.0, 0.0, 0.0
    for p in model.path_names:
        for conv in (model.paths[p].conv1, model.paths[p].conv2):
            mac += _dense_conv_pairs(conv, T, device) * conv.out_channels
            gn += gn_per * conv.out_channels * T
            units += conv.out_channels * T
    for fc in [model.heads[p].fc for p in model.path_names] + [model.hidden.fc, model.out.fc]:
        mac += fc.in_features * fc.out_features * T
        units += fc.out_features * T
    e_mac, e_ac = cfg_energy.e_mac_pj, cfg_energy.e_ac_pj
    return {
        "assumptions": {"e_mac_pj": e_mac, "e_ac_pj": e_ac, "t_max": T, "gn_macs_per_element": gn_per,
                        "excluded": "memory access, GroupNorm statistics, ReLU, control logic"},
        "per_inference": {
            "ann_synaptic_mac": mac, "ann_dense_mac": gn, "ann_dense_ac": units,
            "ann_synaptic_energy_nj": mac * e_mac / 1e3,
            "ann_total_energy_nj": ((mac + gn) * e_mac + units * e_ac) / 1e3,
            "encoding_energy_nj": encoding["energy_nj"] if encoding else None,
        },
        "encoding": encoding,
    }


@torch.no_grad()
def count_operations(model: SCNN, loader, device: torch.device, cfg_energy: EnergyConfig) -> dict:
    """Average spikes, operations and energy per inference over ``loader`` for the SNN and
    the architecture-matched ANN (dense counts only, if ``model`` is the ANN twin)."""
    x0, fs0, _ = next(iter(loader))
    enc = encoding_cost(x0.shape[1], x0.shape[2], fs0.shape[1], model.encoders["ttfs"].num_segments, cfg_energy,
                        model.path_names)
    if not model.spiking:
        return count_dense_operations(model, device, cfg_energy, enc)
    gn_per = GN_MACS_PER_ELEMENT if _has_gn(model) else 0
    model.eval()
    snn, n = _Tally(), 0
    for x, fs1, _ in loader:
        x, fs1 = x.to(device), fs1.to(device)
        B = x.size(0)
        n += B
        spikes = model.encode(x, fs1)
        heads = []
        for p in model.path_names:
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
            gn = gn_per * units
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
            "gn_macs_per_element": gn_per, "lif_macs_per_update": LIF_MACS_PER_UPDATE,
            "excluded": "memory access, GroupNorm statistics, control logic",
            "encoding": "counted separately (encoding_energy_nj); identical for the SNN and the matched ANN",
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
            "encoding_energy_nj": enc["energy_nj"],
        },
        "encoding": enc,
        "per_layer": layers,
        "inferences": n,
    }


def _module(model: SCNN, name: str) -> nn.Module:
    if name in ("hidden", "output"):
        return (model.hidden if name == "hidden" else model.out).fc
    path, layer = name.split(".")
    return model.heads[path].fc if layer == "head" else getattr(model.paths[path], layer)
