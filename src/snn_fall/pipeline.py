"""End-to-end experiment: data -> STDP pretraining -> supervised training -> evaluation."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import plotting
from .config import Config, load_config
from .data import FallDataset, load_or_preprocess, shifted_eval_set, summarize, task_view
from .evaluate import compute_metrics, multiclass_metrics, select_threshold, threshold_sweep
from .model import SCNN, count_parameters
from .stdp import STDPTrainer
from .train import fit, make_loaders, predict, seed_everything


def load_data(cfg: Config) -> dict:
    return task_view(load_or_preprocess(cfg.data, cfg.seed), cfg.task)


def build_model(cfg: Config, data: dict) -> SCNN:
    return SCNN(cfg, n_channels=data["x_train"].shape[2], n_fs1=data["fs1_train"].shape[1],
                num_classes=len(cfg.task.class_names))


def run_experiment(cfg: Config, name: str | None = None) -> Path:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.output_dir) / f"{datetime.now():%Y%m%d-%H%M%S}{'_' + name if name else ''}"
    run_dir.mkdir(parents=True)
    cfg.save(run_dir / "config.yaml")
    print(f"Run directory: {run_dir}\nDevice: {device}\nTask: {cfg.task.kind} {cfg.task.class_names}")

    data = load_data(cfg)
    print(summarize(data, cfg.task.class_names))
    loaders = make_loaders(data, cfg)
    model = build_model(cfg, data).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    if cfg.stdp.enabled and cfg.stdp.epochs > 0:
        rel = STDPTrainer(model, cfg.stdp).run(loaders["train"], device)
        print("STDP mean |dW| / std(W) per call: " + ", ".join(f"{k}={v:.2%}" for k, v in rel.items()))
        torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, run_dir / "stdp_pretrained.pt")

    fit(model, loaders, cfg, device, run_dir)
    evaluate_run(run_dir, data=data, device=device)
    return run_dir


def evaluate_run(run_dir: Path, data: dict | None = None, device: torch.device | None = None) -> dict:
    """Load run_dir/best.pt and report val + test metrics (threshold picked on val for binary)."""
    run_dir = Path(run_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(run_dir / "config.yaml")
    data = data or load_data(cfg)
    loaders = make_loaders(data, cfg)
    names = cfg.task.class_names
    binary = cfg.task.kind == "binary"

    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model = build_model(cfg, data).to(device)
    model.load_state_dict(ckpt["model"])

    preds = {s: predict(model, loaders[s], device, cfg)[:2] for s in ("val", "test", "test_aligned")}
    results = {"best_epoch": ckpt["epoch"], "parameters": count_parameters(model), "task": cfg.task.kind,
               "classes": names, "eval_fall_offsets_s": cfg.eval.fall_offsets_s}

    if binary:
        y_val, p_val = preds["val"]
        sweep = threshold_sweep(y_val, p_val[:, 1])
        sweep.to_csv(run_dir / "threshold_sweep_val.csv", index=False)
        thr = select_threshold(sweep, cfg.eval.threshold_metric)
        results["selected_threshold"] = thr
        for s, (y, p) in preds.items():
            results[s] = compute_metrics(y, p[:, 1], thr)
        y_test, p_test = preds["test"]
        results["test_at_0.5"] = compute_metrics(y_test, p_test[:, 1], 0.5)

        def correct(p, y):
            return (p[:, 1] >= thr).astype(int) == y
    else:
        for s, (y, p) in preds.items():
            results[s] = multiclass_metrics(y, p, names)

        def correct(p, y):
            return p.argmax(1) == y

    # share of test falls handled correctly (binary: detected; categories: right class)
    # as a function of where the impact sits in the window
    offsets = [o for o in cfg.eval.diagnostic_offsets_s if 0 <= o < cfg.data.window_s]
    diag = shifted_eval_set(data, "test", offsets, cfg.data.sample_rate)
    _, p_diag, _ = predict(model, DataLoader(FallDataset(diag["x"], diag["fs1"], diag["y"]),
                                             batch_size=cfg.train.batch_size * 4), device, cfg)
    ok = correct(p_diag, diag["y"])
    by_offset = {f"{o:g}": float(ok[diag["offset_s"] == o].mean()) for o in offsets if (diag["offset_s"] == o).any()}
    results["test_fall_correct_by_impact_offset_s"] = by_offset

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    plotting.plot_training_log(run_dir / "train_log.csv", run_dir / "training_curves.png")
    ylabel = "Fall recall (test)" if binary else "Category accuracy (test falls)"
    plotting.plot_by_offset(by_offset, cfg.train.fall_shift_s, ylabel, run_dir / "by_offset_test.png")
    if binary:
        t = results["test"]
        cm = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
        plotting.plot_threshold_sweep(sweep, thr, run_dir / "threshold_sweep_val.png")
        plotting.plot_confusion(cm, names, f"Test (threshold {thr:.2f})", run_dir / "confusion_test.png")
        print(f"\nThreshold chosen on val: {thr:.2f}")
    else:
        cm = np.array(results["test"]["confusion_matrix"])
        plotting.plot_confusion(cm, names, "Test", run_dir / "confusion_test.png")
        print()

    for s in ("test", "test_aligned"):
        t = results[s]
        extra = (f"(TN {t['tn']} FP {t['fp']} FN {t['fn']} TP {t['tp']})" if binary
                 else f"balanced acc {t['balanced_accuracy']:.4f}")
        print(f"{s.upper():<13} acc {t['accuracy']:.4f}  precision {t['precision']:.4f}  "
              f"recall {t['recall']:.4f}  F1 {t['f1']:.4f}  AUC {t['auc']:.4f}  {extra}")
    if not binary:
        for c, m in results["test"]["per_class"].items():
            print(f"  {c:<10} precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
                  f"F1 {m['f1']:.3f}  (n={m['support']})")
    print(f"Test falls handled correctly by impact position (s): "
          + "  ".join(f"{k}:{v:.0%}" for k, v in by_offset.items()))
    print(f"Results written to {run_dir}")
    return results
