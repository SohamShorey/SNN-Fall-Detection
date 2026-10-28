"""Experiment configuration: typed dataclasses loaded from YAML with CLI overrides."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    raw_dir: str = "data/SisFall_dataset"
    cache_path: str = "data/processed/sisfall_4s.npz"
    subject_prefixes: list[str] = field(default_factory=lambda: ["SA"])
    sample_rate: int = 200
    sensor_cols: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    peak_cols: list[int] = field(default_factory=lambda: [0, 1, 2])
    window_s: float = 4.0
    fall_pre_s: float = 3.0
    highpass_hz: float = 0.4
    highpass_order: int = 4
    highpass_causal: bool = False  # default.yaml enables it; False here keeps older run configs valid
    split_by: str = "subject"
    test_frac: float = 0.15
    val_frac: float = 0.15

    @property
    def window_len(self) -> int:
        return int(round(self.window_s * self.sample_rate))

    @property
    def fall_pre_len(self) -> int:
        return int(round(self.fall_pre_s * self.sample_rate))


@dataclass
class EncodingConfig:
    t_max: int = 20
    ttfs_segments: int = 4
    ttfs_reduce: str = "max"
    ttfs_threshold: float = 0.05
    lc_init_threshold: float = 1.0
    lc_surrogate_slope: float = 8.0
    fs1_threshold: float = 0.01


@dataclass
class ModelConfig:
    kind: str = "scnn"  # scnn (spiking) | ann (ANN twin: same layers with ReLU instead of LIF)
    conv_channels: int = 32
    kernel1: int = 5
    kernel2: int = 3
    gn_groups: int = 4
    head_dim: int = 64
    hidden_dim: int = 128
    beta: float = 0.9


@dataclass
class STDPConfig:
    enabled: bool = True
    epochs: int = 5
    lr: float = 1e-5
    decay: float = 0.005
    clip: float = 0.25
    a_plus: float = 1.0
    a_minus: float = 1.0
    target_rel: dict[str, float] = field(default_factory=lambda: {"conv1": 0.06, "conv2": 0.03})
    max_step_frac: dict[str, float] = field(default_factory=lambda: {"conv1": 0.015, "conv2": 0.008})


@dataclass
class TrainConfig:
    epochs: int = 200
    patience: int = 25
    batch_size: int = 32
    lr: float = 3e-4
    threshold_lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    label_smoothing: float = 0.05
    loss_mem_weight: float = 1.0
    loss_rate_weight: float = 0.0
    noise_std: float = 0.01
    fall_shift_s: list[float] | None = None  # [lo, hi]: random impact position (s) for train falls
    scheduler_factor: float = 0.3
    scheduler_patience: int = 10
    min_lr: float = 1e-6
    num_workers: int = 0


@dataclass
class EvalConfig:
    # Val/test falls are re-cut with the impact at each of these positions (s); val
    # (early stopping, threshold) and the headline test metrics use this set.
    # None evaluates only the aligned windows (impact at data.fall_pre_s).
    fall_offsets_s: list[float] | None = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5, 3.0])
    # Impact positions for the per-position recall diagnostic on test.
    diagnostic_offsets_s: list[float] = field(
        default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5])
    threshold_metric: str = "f1"  # metric maximised on val to pick the decision threshold


@dataclass
class EnergyConfig:
    # Energy per operation (pJ). Defaults: 45 nm CMOS, 32-bit float (Horowitz, ISSCC 2014).
    e_mac_pj: float = 4.6  # multiply-accumulate (ANN synapse, dense SNN work)
    e_ac_pj: float = 0.9   # accumulate (SNN synapse driven by a spike)


@dataclass
class TaskConfig:
    # binary:        ADL vs fall (the fall detector).
    # fall_category: fall windows only, labelled by the groups in fall_groups (ADLs dropped).
    kind: str = "binary"
    fall_groups: dict[str, list[str]] | None = None  # class name -> SisFall fall codes

    @property
    def class_names(self) -> list[str]:
        return ["ADL", "Fall"] if self.kind == "binary" else list(self.fall_groups)


@dataclass
class Config:
    seed: int = 42
    output_dir: str = "runs"
    task: TaskConfig = field(default_factory=TaskConfig)
    data: DataConfig = field(default_factory=DataConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    stdp: STDPConfig = field(default_factory=STDPConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_yaml(path: Path) -> dict:
    """Read a YAML config, resolving an optional `base:` key relative to the file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("base", None)
    if base is not None:
        raw = _deep_merge(_read_yaml(path.parent / base), raw)
    return raw


def _apply_override(raw: dict, assignment: str) -> None:
    """Apply a dotted `section.key=value` override; the value is parsed as YAML."""
    key, sep, value = assignment.partition("=")
    if not sep:
        raise ValueError(f"Override must look like key=value, got {assignment!r}")
    node = raw
    *parents, leaf = key.strip().split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = yaml.safe_load(value)


def _build(cls, raw: dict):
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise KeyError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in raw.items():
        sub = {"data": DataConfig, "encoding": EncodingConfig, "model": ModelConfig,
               "stdp": STDPConfig, "train": TrainConfig, "eval": EvalConfig, "task": TaskConfig,
               "energy": EnergyConfig}.get(name) if cls is Config else None
        kwargs[name] = _build(sub, value) if sub is not None else value
    return cls(**kwargs)


def validate(cfg: Config) -> None:
    """Check that settings are mutually consistent, with messages that say what to change."""
    d, e, m, t, ev = cfg.data, cfg.encoding, cfg.model, cfg.train, cfg.eval
    errors = []

    def check(ok: bool, msg: str) -> None:
        if not ok:
            errors.append(msg)

    W = d.window_s
    check(W > 0, "data.window_s must be positive.")
    check(0 <= d.fall_pre_s < W,
          f"data.fall_pre_s ({d.fall_pre_s}) must be in [0, data.window_s={W}) so the impact is inside the window.")
    check(d.split_by in ("subject", "activity"), "data.split_by must be 'subject' or 'activity'.")
    check(all(0 <= c <= 8 for c in d.sensor_cols + d.peak_cols),
          "data.sensor_cols / data.peak_cols must be SisFall column indices 0-8.")
    check(0 < d.highpass_hz < d.sample_rate / 2, "data.highpass_hz must be between 0 and half data.sample_rate.")
    check(0 < d.test_frac and 0 < d.val_frac and d.test_frac + d.val_frac < 1,
          "data.test_frac and data.val_frac must be positive and sum to less than 1.")
    if t.fall_shift_s:
        lo, hi = (t.fall_shift_s + [None, None])[:2]
        check(len(t.fall_shift_s) == 2 and 0 <= lo <= hi < W,
              f"train.fall_shift_s must be [lo, hi] with 0 <= lo <= hi < data.window_s ({W}).")
    check(all(0 <= o < W for o in ev.fall_offsets_s or []),
          f"eval.fall_offsets_s values must be in [0, data.window_s={W}).")
    check(ev.threshold_metric in ("f1", "accuracy"), "eval.threshold_metric must be 'f1' or 'accuracy'.")

    tk = cfg.task
    check(tk.kind in ("binary", "fall_category"), "task.kind must be 'binary' or 'fall_category'.")
    if tk.kind == "fall_category":
        groups = tk.fall_groups or {}
        codes = [c for cs in groups.values() for c in cs]
        valid = {f"F{i:02d}" for i in range(1, 16)}
        check(len(groups) >= 2, "task.fall_groups needs at least two classes for task.kind=fall_category.")
        check(all(c in valid for c in codes), f"task.fall_groups codes must be SisFall fall codes F01-F15, "
                                              f"got {sorted(set(codes) - valid)}.")
        check(len(codes) == len(set(codes)), "Each fall code may appear in only one task.fall_groups class.")

    L = d.window_len
    check(e.t_max >= 1 and (L - 1) // e.t_max >= 1,
          f"encoding.t_max ({e.t_max}) must be between 1 and {L - 1} (window length - 1 samples) "
          "so each level-crossing time bin holds at least one sample.")
    check(1 <= e.ttfs_segments <= L, f"encoding.ttfs_segments must be between 1 and {L}.")
    check(e.ttfs_reduce in ("max", "mean"), "encoding.ttfs_reduce must be 'max' or 'mean'.")
    check(m.kernel1 % 2 == 1 and m.kernel2 % 2 == 1,
          "model.kernel1 / model.kernel2 must be odd so the conv output keeps t_max time steps.")
    check(m.conv_channels % m.gn_groups == 0,
          f"model.conv_channels ({m.conv_channels}) must be divisible by model.gn_groups ({m.gn_groups}).")
    check(0 < m.beta <= 1, "model.beta must be in (0, 1].")
    check(m.kind in ("scnn", "ann"), "model.kind must be 'scnn' or 'ann'.")
    if m.kind == "ann":
        check(not cfg.stdp.enabled, "STDP pretraining applies to the spiking model only: set stdp.enabled=false "
                                    "for model.kind=ann.")
        check(t.loss_rate_weight == 0, "The spike-rate loss needs spiking outputs: set train.loss_rate_weight=0 "
                                       "for model.kind=ann.")
    for layer in ("conv1", "conv2"):
        check(layer in cfg.stdp.target_rel and layer in cfg.stdp.max_step_frac,
              f"stdp.target_rel and stdp.max_step_frac need a value for {layer}.")
    if errors:
        raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    raw = _read_yaml(Path(path)) if path else {}
    for o in overrides or []:
        _apply_override(raw, o)
    cfg = _build(Config, raw)
    validate(cfg)
    return cfg
