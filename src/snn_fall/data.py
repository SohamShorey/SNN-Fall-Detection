"""SisFall loading, filtering, windowing, subject-wise splitting and normalisation.

Pipeline (one recording -> zero or more 4 s windows):
  1. Parse the 9-column ADC recording at 200 Hz.
  2. 4th-order Butterworth high-pass at 0.4 Hz (removes gravity / drift); causal
     (forward-only) by default, optionally zero-phase.
  3. Falls (F**): one window placed so the peak acceleration magnitude sits 3 s
     after the window start, plus the surrounding context so the window can be
     re-cut with the impact elsewhere (training augmentation and shifted
     evaluation). ADLs (D**): non-overlapping 4 s windows.
  4. FS1 statistical features (mean, std, max, min per channel) per window.
  5. Train/val/test split by subject, z-score statistics fitted on train only.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import DataConfig, TaskConfig

SPLITS = ("train", "val", "test")
CACHE_FORMAT = 2  # bump when the cached arrays change so stale caches are rebuilt
_FILE_RE = re.compile(r"^(?P<activity>[DF]\d{2})_(?P<subject>S[AE]\d{2})_(?P<trial>R\d{2})\.txt$")


def read_recording(path: Path) -> np.ndarray:
    """Read one SisFall .txt file into a float array of shape (n_samples, 9).

    A few recordings contain truncated lines; only complete 9-column rows are kept.
    """
    lines = [ln for ln in path.read_text().replace(";", "").splitlines() if ln.count(",") >= 8]
    return np.loadtxt(io.StringIO("\n".join(lines)), delimiter=",", usecols=range(9), ndmin=2)


def highpass(x: np.ndarray, cutoff_hz: float, fs: int, order: int, causal: bool = False) -> np.ndarray:
    """Butterworth high-pass along time.

    causal=True filters forward only (realisable on a device in real time); the
    filter state starts at steady state for the first sample to avoid a start-up
    transient. causal=False is the zero-phase forward-backward filter.
    """
    sos = butter(order, cutoff_hz, btype="high", fs=fs, output="sos")
    if not causal:
        return sosfiltfilt(sos, x, axis=0)
    zi = sosfilt_zi(sos)[:, :, None] * x[0]  # (sections, 2, channels)
    return sosfilt(sos, x, axis=0, zi=zi)[0]


def fs1_features(window: np.ndarray) -> np.ndarray:
    """Mean, std, max and min per channel -> (4 * channels,)."""
    return np.concatenate([window.mean(0), window.std(0), window.max(0), window.min(0)])


def extract_windows(signal: np.ndarray, is_fall: bool, cfg: DataConfig) -> list[np.ndarray]:
    """Cut a filtered recording into fixed-length windows of the selected sensor columns."""
    L = cfg.window_len
    if is_fall:
        peak = fall_peak(signal, cfg)
        start = peak - cfg.fall_pre_len
        if start < 0 or start + L > len(signal):
            return []  # peak too close to either end of the recording
        starts = [start]
    else:
        starts = range(0, len(signal) - L + 1, L)
    return [signal[s:s + L, cfg.sensor_cols] for s in starts]


def fall_peak(signal: np.ndarray, cfg: DataConfig) -> int:
    return int(np.argmax(np.linalg.norm(signal[:, cfg.peak_cols], axis=1)))


def fall_context(signal: np.ndarray, cfg: DataConfig) -> tuple[np.ndarray, int, int]:
    """Signal from one window length before to one after the impact peak (zero-padded).

    The impact sits at index ``window_len`` of the returned (2 * window_len, C) array.
    A window with the impact ``pre`` samples after its start is
    ``ctx[window_len - pre : 2 * window_len - pre]``; also returns the range of
    ``pre`` values whose window lies fully inside the recording.
    """
    L, n = cfg.window_len, len(signal)
    peak = fall_peak(signal, cfg)
    ctx = np.zeros((2 * L, len(cfg.sensor_cols)))
    lo, hi = max(0, peak - L), min(n, peak + L)
    ctx[lo - (peak - L):hi - (peak - L)] = signal[lo:hi, cfg.sensor_cols]
    pre_min = max(0, peak + L - n)   # window must end inside the recording
    pre_max = min(L - 1, peak)       # window must start inside the recording
    return ctx, pre_min, pre_max


def list_recordings(cfg: DataConfig) -> list[dict]:
    root = Path(cfg.raw_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"SisFall directory not found: {root.resolve()}")
    recs = []
    for path in sorted(root.glob("S*/*.txt")):
        m = _FILE_RE.match(path.name)
        if m and m["subject"][:2] in cfg.subject_prefixes:
            recs.append({"path": path, **m.groupdict()})
    if not recs:
        raise FileNotFoundError(f"No SisFall recordings matched under {root.resolve()}")
    return recs


def split_groups(groups: list[str], cfg: DataConfig, seed: int) -> dict[str, list[str]]:
    """Split group ids (subjects, or activities for the legacy split) into train/val/test."""
    groups = sorted(set(groups))
    rest, test = train_test_split(groups, test_size=cfg.test_frac, random_state=seed)
    train, val = train_test_split(rest, test_size=cfg.val_frac / (1 - cfg.test_frac), random_state=seed)
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def preprocess(cfg: DataConfig, seed: int) -> dict:
    """Build the windowed, normalised dataset and write it to ``cache_file(cfg, seed)``."""
    if cfg.split_by not in ("subject", "activity"):
        raise ValueError(f"split_by must be 'subject' or 'activity', got {cfg.split_by!r}")
    recs = list_recordings(cfg)
    groups = split_groups([r[cfg.split_by] for r in recs], cfg, seed)
    group_to_split = {g: s for s, gs in groups.items() for g in gs}

    buf = {s: {"x": [], "fs1": [], "y": [], "subject": [], "activity": []} for s in SPLITS}
    ctx_buf = {s: {"ctx": [], "idx": [], "pre_min": [], "pre_max": []} for s in SPLITS}
    skipped = Counter()
    for rec in tqdm(recs, desc="Reading SisFall"):
        raw = read_recording(rec["path"])
        if len(raw) < cfg.window_len:
            skipped["too_short"] += 1
            continue
        is_fall = rec["activity"].startswith("F")
        filtered = highpass(raw, cfg.highpass_hz, cfg.sample_rate, cfg.highpass_order, cfg.highpass_causal)
        windows = extract_windows(filtered, is_fall, cfg)
        if not windows:
            skipped["fall_peak_at_edge"] += 1
        split = group_to_split[rec[cfg.split_by]]
        b = buf[split]
        if is_fall and windows:
            ctx, pre_min, pre_max = fall_context(filtered, cfg)
            c = ctx_buf[split]
            c["ctx"].append(ctx)
            c["idx"].append(len(b["y"]))
            c["pre_min"].append(pre_min)
            c["pre_max"].append(pre_max)
        for w in windows:
            b["x"].append(w)
            b["fs1"].append(fs1_features(w))
            b["y"].append(int(is_fall))
            b["subject"].append(rec["subject"])
            b["activity"].append(rec["activity"])

    arrays = {s: {k: np.asarray(v) for k, v in buf[s].items()} for s in SPLITS}

    # z-score using train statistics only
    x_mean = arrays["train"]["x"].mean(axis=(0, 1))
    x_std = arrays["train"]["x"].std(axis=(0, 1)) + 1e-8
    f_mean = arrays["train"]["fs1"].mean(axis=0)
    f_std = arrays["train"]["fs1"].std(axis=0) + 1e-8

    out = {}
    for s in SPLITS:
        out[f"x_{s}"] = ((arrays[s]["x"] - x_mean) / x_std).astype(np.float32)
        out[f"fs1_{s}"] = ((arrays[s]["fs1"] - f_mean) / f_std).astype(np.float32)
        out[f"y_{s}"] = arrays[s]["y"].astype(np.int64)
        out[f"subject_{s}"] = arrays[s]["subject"]
        out[f"activity_{s}"] = arrays[s]["activity"]
        c = ctx_buf[s]
        out[f"ctx_{s}"] = ((np.asarray(c["ctx"]) - x_mean) / x_std).astype(np.float32)
        out[f"ctx_idx_{s}"] = np.asarray(c["idx"], dtype=np.int64)
        out[f"ctx_pre_min_{s}"] = np.asarray(c["pre_min"], dtype=np.int64)
        out[f"ctx_pre_max_{s}"] = np.asarray(c["pre_max"], dtype=np.int64)
    meta = {
        "format": CACHE_FORMAT,
        "data_config": asdict(cfg),
        "seed": seed,
        "groups": groups,
        "skipped_recordings": dict(skipped),
        "norm": {"x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
                 "fs1_mean": f_mean.tolist(), "fs1_std": f_std.tolist()},
    }
    out["meta"] = np.array(json.dumps(meta))

    cache = cache_file(cfg, seed)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return load_processed(cache)


def cache_file(cfg: DataConfig, seed: int) -> Path:
    """Cache path tagged with a hash of the data settings, so each variant gets its own file."""
    settings = {k: v for k, v in asdict(cfg).items() if k != "cache_path"}
    digest = hashlib.sha1(json.dumps({**settings, "seed": seed}, sort_keys=True).encode()).hexdigest()[:8]
    base = Path(cfg.cache_path)
    return base.with_name(f"{base.stem}-{digest}{base.suffix}")


def load_processed(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files if k != "meta"}
        data["meta"] = json.loads(str(z["meta"]))
    return data


def load_or_preprocess(cfg: DataConfig, seed: int) -> dict:
    """Load the cached dataset, rebuilding it if missing or built with different settings."""
    cache = cache_file(cfg, seed)
    if cache.exists():
        data = load_processed(cache)
        if (data["meta"]["data_config"] == asdict(cfg) and data["meta"]["seed"] == seed
                and data["meta"].get("format") == CACHE_FORMAT):
            return data
        print(f"Cache {cache} does not match the current settings; rebuilding.")
    return preprocess(cfg, seed)


def task_view(data: dict, task: TaskConfig) -> dict:
    """Labels for the task. binary: unchanged (0 = ADL, 1 = fall). fall_category: keep only
    falls whose code is in task.fall_groups and label them by group index."""
    if task.kind == "binary":
        return data
    code_to_class = {code: i for i, codes in enumerate(task.fall_groups.values()) for code in codes}
    out = {"meta": data["meta"]}
    for s in SPLITS:
        keep = np.isin(data[f"activity_{s}"], list(code_to_class))
        new_index = np.cumsum(keep) - 1  # old window index -> index after filtering
        for k in ("x", "fs1", "subject", "activity"):
            out[f"{k}_{s}"] = data[f"{k}_{s}"][keep]
        out[f"y_{s}"] = np.array([code_to_class[a] for a in out[f"activity_{s}"]], dtype=np.int64)
        rows = keep[data[f"ctx_idx_{s}"]]
        out[f"ctx_idx_{s}"] = new_index[data[f"ctx_idx_{s}"][rows]]
        for k in ("ctx", "ctx_pre_min", "ctx_pre_max"):
            out[f"{k}_{s}"] = data[f"{k}_{s}"][rows]
    return out


def summarize(data: dict, class_names: list[str]) -> str:
    lines = []
    for s in SPLITS:
        y = data[f"y_{s}"]
        counts = " / ".join(f"{int((y == i).sum())} {name}" for i, name in enumerate(class_names))
        lines.append(f"{s:>5}: {len(y):5d} windows  ({counts})  groups={data['meta']['groups'][s]}")
    return "\n".join(lines)


def window_from_context(data: dict, split: str, row: int, pre: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalised window (impact ``pre`` samples after its start) and its normalised FS1 features."""
    ctx = data[f"ctx_{split}"][row]
    L = ctx.shape[0] // 2
    win = ctx[L - pre:2 * L - pre]
    norm = data["meta"]["norm"]
    raw = win * np.asarray(norm["x_std"]) + np.asarray(norm["x_mean"])
    fs1 = (fs1_features(raw) - np.asarray(norm["fs1_mean"])) / np.asarray(norm["fs1_std"])
    return win.astype(np.float32), fs1.astype(np.float32)


def shifted_eval_set(data: dict, split: str, offsets_s: list[float], sample_rate: int) -> dict:
    """Non-fall windows unchanged plus every fall re-cut with its impact at each offset (seconds).

    (fall, offset) pairs whose window would leave the recording are skipped. Falls keep
    their label. Returns arrays x, fs1, y and offset_s (NaN for non-fall windows).
    """
    y = data[f"y_{split}"]
    ctx_idx = data[f"ctx_idx_{split}"]
    other = np.ones(len(y), dtype=bool)
    other[ctx_idx] = False
    xs, fs, ys = [data[f"x_{split}"][other]], [data[f"fs1_{split}"][other]], [y[other]]
    offs = [np.full(other.sum(), np.nan)]
    for row, idx in enumerate(ctx_idx):
        lo, hi = data[f"ctx_pre_min_{split}"][row], data[f"ctx_pre_max_{split}"][row]
        for off in offsets_s:
            pre = int(round(off * sample_rate))
            if lo <= pre <= hi:
                w, f = window_from_context(data, split, row, pre)
                xs.append(w[None]); fs.append(f[None]); ys.append([y[idx]]); offs.append([off])
    return {"x": np.concatenate(xs).astype(np.float32), "fs1": np.concatenate(fs).astype(np.float32),
            "y": np.concatenate(ys).astype(np.int64), "offset_s": np.concatenate(offs)}


class FallDataset(Dataset):
    """Pre-normalised windows + FS1 features, with optional augmentation.

    noise_std: Gaussian noise added to the window and FS1 features.
    shift: (data, split, (pre_lo, pre_hi) in samples) to re-cut each fall window with its
        impact at a uniformly random position, drawn per access; FS1 is recomputed.
    """

    def __init__(self, x: np.ndarray, fs1: np.ndarray, y: np.ndarray, noise_std: float = 0.0,
                 shift: tuple[dict, str, tuple[int, int]] | None = None):
        self.x = torch.from_numpy(x)
        self.fs1 = torch.from_numpy(fs1)
        self.y = torch.from_numpy(y)
        self.noise_std = noise_std
        self.shift = shift
        if shift is not None:
            data, split, _ = shift
            self.ctx_row = {int(i): r for r, i in enumerate(data[f"ctx_idx_{split}"])}

    def __len__(self) -> int:
        return len(self.y)

    def _shifted(self, i: int):
        data, split, (lo, hi) = self.shift
        row = self.ctx_row[i]
        lo = max(lo, int(data[f"ctx_pre_min_{split}"][row]))
        hi = min(hi, int(data[f"ctx_pre_max_{split}"][row]))
        if lo > hi:  # no window in the requested range fits this recording: keep the aligned one
            return self.x[i], self.fs1[i]
        w, f = window_from_context(data, split, row, int(np.random.randint(lo, hi + 1)))
        return torch.from_numpy(w), torch.from_numpy(f)

    def __getitem__(self, i):
        if self.shift is not None and i in self.ctx_row:
            x, fs1 = self._shifted(i)
        else:
            x, fs1 = self.x[i], self.fs1[i]
        if self.noise_std > 0:
            x = x + self.noise_std * torch.randn_like(x)
            fs1 = fs1 + self.noise_std * torch.randn_like(fs1)
        return x, fs1, self.y[i]
