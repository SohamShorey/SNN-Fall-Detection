"""Non-neural baselines on exactly the same windows, splits and evaluation as the SCNN.

Fall detector (task.kind = binary):
  threshold_acc       fall if peak acceleration magnitude in the window exceeds a threshold
  threshold_acc_gyro  fall if peak acceleration AND peak angular-rate magnitudes exceed thresholds
Fall direction (task.kind = fall_category):
  rule_impact_dir     nearest class centroid of the impact direction (unit acceleration vector
                      at the peak) and the mean acceleration in the second after it
Both tasks:
  svm_rbf, random_forest, knn on hand-crafted window statistics

Training falls are re-cut at evenly spaced impact positions over train.fall_shift_s (the
networks draw them at random from the same range). Thresholds and hyper-parameters are chosen
on the validation subjects only; the test subjects are scored once with the chosen setting.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import Config
from .data import shifted_eval_set
from .evaluate import compute_metrics, multiclass_metrics
from .train import eval_sets

ACC_COLS, GYRO_COLS = {0, 1, 2, 6, 7, 8}, {3, 4, 5}


# ---------------------------------------------------------------- features
def _raw(x: np.ndarray, data: dict) -> np.ndarray:
    norm = data["meta"]["norm"]
    return x * np.asarray(norm["x_std"]) + np.asarray(norm["x_mean"])


def _channel_groups(cfg: Config) -> tuple[list[int], list[int]]:
    cols = cfg.data.sensor_cols
    acc = [i for i, c in enumerate(cols) if c in ACC_COLS][:3]
    gyro = [i for i, c in enumerate(cols) if c in GYRO_COLS][:3]
    return acc, gyro


def window_features(x: np.ndarray, data: dict, cfg: Config) -> np.ndarray:
    """Per channel mean/std/max/min, plus mean/std/max of the acceleration and angular-rate
    magnitudes: [N, 4 * C + 6]. Computed on the filtered signal in sensor units."""
    raw = _raw(x, data)
    acc, gyro = _channel_groups(cfg)
    feats = [raw.mean(1), raw.std(1), raw.max(1), raw.min(1)]
    for idx in (acc, gyro):
        if idx:
            mag = np.linalg.norm(raw[:, :, idx], axis=2)
            feats += [mag.mean(1, keepdims=True), mag.std(1, keepdims=True), mag.max(1, keepdims=True)]
    return np.concatenate(feats, axis=1)


def peak_magnitudes(x: np.ndarray, data: dict, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    raw = _raw(x, data)
    acc, gyro = _channel_groups(cfg)
    a = np.linalg.norm(raw[:, :, acc], axis=2).max(1)
    g = np.linalg.norm(raw[:, :, gyro], axis=2).max(1) if gyro else np.zeros(len(x))
    return a, g


def impact_direction_features(x: np.ndarray, data: dict, cfg: Config) -> np.ndarray:
    """Unit acceleration vector at the impact peak and the mean acceleration (unit vector)
    over the following second: [N, 6]."""
    raw = _raw(x, data)
    acc, _ = _channel_groups(cfg)
    a = raw[:, :, acc]
    peak = np.linalg.norm(a, axis=2).argmax(1)
    after = cfg.data.sample_rate
    out = np.empty((len(x), 6))
    for i, p in enumerate(peak):
        v = a[i, p]
        w = a[i, p:p + after].mean(0)
        out[i, :3] = v / (np.linalg.norm(v) + 1e-9)
        out[i, 3:] = w / (np.linalg.norm(w) + 1e-9)
    return out


# ---------------------------------------------------------------- helpers
def _train_set(data: dict, cfg: Config) -> dict:
    lo, hi = cfg.train.fall_shift_s or [cfg.data.fall_pre_s] * 2
    return shifted_eval_set(data, "train", list(np.linspace(lo, hi, 5)), cfg.data.sample_rate)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _best_threshold(y: np.ndarray, score: np.ndarray, metric: str) -> float:
    cands = np.unique(np.quantile(score, np.linspace(0, 1, 401)))
    vals = [compute_metrics(y, score, t)[metric] for t in cands]
    return float(cands[int(np.argmax(vals))])


def _ops_energy(mac: float, cmp: float, cfg: Config) -> dict:
    return {"mac": mac, "comparisons": cmp,
            "energy_nj": (mac * cfg.energy.e_mac_pj + cmp * cfg.energy.e_ac_pj) / 1e3}


def _model_cost(name: str, model, n_feat: int, cfg: Config) -> dict:
    """Stored parameters and per-inference operations of the classifier (features excluded)."""
    if name == "svm_rbf":
        n_sv = int(model.support_vectors_.shape[0])
        k = len(getattr(model, "classes_", [0, 1]))
        n_dec = 1 if k == 2 else k * (k - 1) // 2
        return {"parameters": n_sv * (n_feat + n_dec), "support_vectors": n_sv,
                **_ops_energy(n_sv * (n_feat + n_dec), n_sv, cfg)}  # kernel distances + weighting; n_sv exp()
    if name == "random_forest":
        nodes = sum(t.tree_.node_count for t in model.estimators_)
        depth = float(np.mean([t.tree_.max_depth for t in model.estimators_]))
        return {"parameters": nodes * 2, "trees": len(model.estimators_), "mean_max_depth": depth,
                **_ops_energy(0.0, depth * len(model.estimators_), cfg)}  # upper bound: max depth per tree
    if name == "knn":
        n = int(model.n_samples_fit_)
        return {"parameters": n * n_feat, "stored_samples": n, **_ops_energy(n * n_feat, n, cfg)}
    if name == "rule_impact_dir":
        k = len(model.classes_)
        return {"parameters": k * n_feat, **_ops_energy(k * n_feat, k, cfg)}
    return {"parameters": 1 if name == "threshold_acc" else 2, **_ops_energy(0.0, 1 if name == "threshold_acc" else 2, cfg)}


def feature_cost(cfg: Config) -> dict:
    """Approximate operations to compute the window statistics (shared by all classical models)."""
    L, C = cfg.data.window_len, len(cfg.data.sensor_cols)
    mac = L * C * 2 + L * 6       # mean + variance accumulations; squared magnitudes for acc and gyro
    cmp = L * C * 2 + L * 2       # running max/min per channel; magnitude maxima
    return _ops_energy(mac, cmp, cfg)


# ---------------------------------------------------------------- experiment
def run_baselines(data: dict, cfg: Config) -> dict:
    binary = cfg.task.kind == "binary"
    names = cfg.task.class_names
    tr = _train_set(data, cfg)
    sets = eval_sets(data, cfg)
    offsets = [o for o in cfg.eval.diagnostic_offsets_s if 0 <= o < cfg.data.window_s]
    sets["diag"] = shifted_eval_set(data, "test", offsets, cfg.data.sample_rate)
    results = {"task": cfg.task.kind, "classes": names, "train_windows": int(len(tr["y"])),
               "feature_extraction": feature_cost(cfg), "models": {}}

    def report(name: str, scores: dict, hyper: dict, model=None, n_feat: int = 0, threshold=None):
        r = {"hyperparameters": hyper, **_model_cost(name, model, n_feat, cfg)}
        if binary:
            r["selected_threshold"] = threshold
            for s in ("val", "test", "test_aligned"):
                r[s] = compute_metrics(sets[s]["y"], scores[s], threshold)
            ok = (scores["diag"] >= threshold).astype(int) == sets["diag"]["y"]
        else:
            for s in ("val", "test", "test_aligned"):
                r[s] = multiclass_metrics(sets[s]["y"], scores[s], names)
            ok = scores["diag"].argmax(1) == sets["diag"]["y"]
        r["test_fall_correct_by_impact_offset_s"] = {
            f"{o:g}": float(ok[sets["diag"]["offset_s"] == o].mean())
            for o in offsets if (sets["diag"]["offset_s"] == o).any()}
        results["models"][name] = r
        t = r["test"]
        print(f"{name:<20} test acc {t['accuracy']:.4f}  P {t['precision']:.4f}  R {t['recall']:.4f}  "
              f"F1 {t['f1']:.4f}  AUC {t['auc']:.4f}  | params {r['parameters']:,}  ops/inf "
              f"{r['mac']:,.0f} MAC + {r['comparisons']:,.0f} cmp")

    # ---- rule-based
    if binary:
        peaks = {s: peak_magnitudes(v["x"], data, cfg) for s, v in {**sets, "train": tr}.items()}
        acc = {s: p[0] for s, p in peaks.items()}
        thr = _best_threshold(sets["val"]["y"], acc["val"], cfg.eval.threshold_metric)
        report("threshold_acc", acc, {"acc_peak_threshold_sensor_units": thr}, threshold=thr)

        # 2-D grid: both peaks above their thresholds; score = min of the two normalised margins
        a_c = np.unique(np.quantile(peaks["val"][0], np.linspace(0, 1, 81)))
        g_c = np.unique(np.quantile(peaks["val"][1], np.linspace(0, 1, 81)))
        best = (-1.0, None)
        for ta in a_c:
            for tg in g_c:
                pred = ((peaks["val"][0] >= ta) & (peaks["val"][1] >= tg)).astype(float)
                m = compute_metrics(sets["val"]["y"], pred, 0.5)[cfg.eval.threshold_metric]
                if m > best[0]:
                    best = (m, (ta, tg))
        ta, tg = best[1]
        both = {s: np.minimum(p[0] / ta, p[1] / tg) for s, p in peaks.items()}
        report("threshold_acc_gyro", both, {"acc_peak_threshold": float(ta), "gyro_peak_threshold": float(tg)},
               threshold=1.0)
    else:
        f = {s: impact_direction_features(v["x"], data, cfg) for s, v in {**sets, "train": tr}.items()}
        nc = NearestCentroid().fit(f["train"], tr["y"])

        def nc_probs(z):
            d = np.linalg.norm(z[:, None, :] - nc.centroids_[None], axis=2)
            return _softmax(-d * 10)
        report("rule_impact_dir", {s: nc_probs(v) for s, v in f.items()}, {}, nc, f["train"].shape[1])

    # ---- classical ML on window statistics
    feats = {s: window_features(v["x"], data, cfg) for s, v in {**sets, "train": tr}.items()}
    scaler = StandardScaler().fit(feats["train"])
    feats = {s: scaler.transform(v) for s, v in feats.items()}
    n_feat = feats["train"].shape[1]
    results["features"] = n_feat
    grids = {
        "svm_rbf": [{"C": c} for c in (0.3, 1, 3, 10, 30)],
        "random_forest": [{"max_depth": d, "min_samples_leaf": l} for d in (None, 12) for l in (1, 4)],
        "knn": [{"n_neighbors": k} for k in (3, 5, 11, 21)],
    }
    makers = {
        "svm_rbf": lambda h: SVC(kernel="rbf", gamma="scale", class_weight="balanced", **h),
        "random_forest": lambda h: RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                          random_state=cfg.seed, n_jobs=-1, **h),
        "knn": lambda h: KNeighborsClassifier(weights="distance", **h),
    }
    for name, grid in grids.items():
        t0 = time.time()
        best = None
        for h in grid:
            model = makers[name](h).fit(feats["train"], tr["y"])
            if binary:
                score = lambda z, m=model: (m.decision_function(z) if name == "svm_rbf" else m.predict_proba(z)[:, 1])
                thr = _best_threshold(sets["val"]["y"], score(feats["val"]), cfg.eval.threshold_metric)
                val = compute_metrics(sets["val"]["y"], score(feats["val"]), thr)[cfg.eval.threshold_metric]
            else:
                score = lambda z, m=model: (_softmax(m.decision_function(z)) if name == "svm_rbf"
                                            else m.predict_proba(z))
                thr = None
                val = multiclass_metrics(sets["val"]["y"], score(feats["val"]), names)["f1"]
            if best is None or val > best[0]:
                best = (val, h, model, score, thr)
        _, h, model, score, thr = best
        report(name, {s: score(v) for s, v in feats.items() if s != "train"}, h, model, n_feat, thr)
        results["models"][name]["fit_seconds"] = round(time.time() - t0, 1)
    return results
