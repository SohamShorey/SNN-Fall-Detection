"""End-to-end experiment: data -> STDP pretraining -> supervised training -> evaluation."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import plotting
from .config import Config, load_config
from .data import FallDataset, load_or_preprocess, shifted_eval_set, summarize
from .evaluate import compute_metrics, select_threshold, threshold_sweep
from .model import SCNN, count_parameters
from .stdp import STDPTrainer
from .train import fit, make_loaders, predict, seed_everything


def build_model(cfg: Config, data: dict) -> SCNN:
    return SCNN(cfg, n_channels=data["x_train"].shape[2], n_fs1=data["fs1_train"].shape[1])


def run_experiment(cfg: Config, name: str | None = None) -> Path:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.output_dir) / f"{datetime.now():%Y%m%d-%H%M%S}{'_' + name if name else ''}"
    run_dir.mkdir(parents=True)
    cfg.save(run_dir / "config.yaml")
    print(f"Run directory: {run_dir}\nDevice: {device}")

    data = load_or_preprocess(cfg.data, cfg.seed)
    print(summarize(data))
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
    """Load run_dir/best.pt, pick the threshold on val, and report val + test metrics."""
    run_dir = Path(run_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(run_dir / "config.yaml")
    data = data or load_or_preprocess(cfg.data, cfg.seed)
    loaders = make_loaders(data, cfg)

    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model = build_model(cfg, data).to(device)
    model.load_state_dict(ckpt["model"])

    y_val, p_val, _ = predict(model, loaders["val"], device, cfg)
    y_test, p_test, _ = predict(model, loaders["test"], device, cfg)
    y_al, p_al, _ = predict(model, loaders["test_aligned"], device, cfg)

    sweep = threshold_sweep(y_val, p_val)
    sweep.to_csv(run_dir / "threshold_sweep_val.csv", index=False)
    thr = select_threshold(sweep, cfg.eval.threshold_metric)

    # recall on test falls as a function of where the impact sits in the window
    offsets = [o for o in cfg.eval.diagnostic_offsets_s if 0 <= o < cfg.data.window_s]
    diag = shifted_eval_set(data, "test", offsets, cfg.data.sample_rate)
    diag_loader = DataLoader(FallDataset(diag["x"], diag["fs1"], diag["y"]), batch_size=cfg.train.batch_size * 4)
    _, p_diag, _ = predict(model, diag_loader, device, cfg)
    recall_by_offset = {f"{off:g}": float((p_diag[diag["offset_s"] == off] >= thr).mean())
                        for off in offsets if (diag["offset_s"] == off).any()}

    results = {
        "best_epoch": ckpt["epoch"],
        "parameters": count_parameters(model),
        "selected_threshold": thr,
        "eval_fall_offsets_s": cfg.eval.fall_offsets_s,
        "val": compute_metrics(y_val, p_val, thr),
        "test": compute_metrics(y_test, p_test, thr),
        "test_at_0.5": compute_metrics(y_test, p_test, 0.5),
        "test_aligned": compute_metrics(y_al, p_al, thr),
        "test_recall_by_impact_offset_s": recall_by_offset,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    plotting.plot_training_log(run_dir / "train_log.csv", run_dir / "training_curves.png")
    plotting.plot_threshold_sweep(sweep, thr, run_dir / "threshold_sweep_val.png")
    plotting.plot_confusion(results["test"], f"Test (threshold {thr:.2f})", run_dir / "confusion_test.png")
    plotting.plot_recall_by_offset(recall_by_offset, cfg.train.fall_shift_s, run_dir / "recall_by_offset_test.png")

    print(f"\nThreshold chosen on val: {thr:.2f}")
    for name in ("test", "test_aligned"):
        t = results[name]
        print(f"{name.upper():<13} acc {t['accuracy']:.4f}  precision {t['precision']:.4f}  "
              f"recall {t['recall']:.4f}  F1 {t['f1']:.4f}  AUC {t['auc']:.4f}  "
              f"(TN {t['tn']} FP {t['fp']} FN {t['fn']} TP {t['tp']})")
    print("Test fall recall by impact position (s): "
          + "  ".join(f"{k}:{v:.0%}" for k, v in recall_by_offset.items()))
    print(f"Results written to {run_dir}")
    return results
