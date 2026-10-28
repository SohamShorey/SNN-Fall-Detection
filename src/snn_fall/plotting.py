"""Figures for a training run (written to disk; never blocks on plt.show)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402


def plot_training_log(log_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(log_csv)
    panels = [
        ("Accuracy", [("train_acc", "Train"), ("val_acc", "Val")]),
        ("Loss", [("train_loss", "Train"), ("val_loss", "Val")]),
        ("F1 (val)", [("val_f1", None)]),
        ("Precision (val)", [("val_precision", None)]),
        ("Recall (val)", [("val_recall", None)]),
        ("AUC (val)", [("val_auc", None)]),
    ]
    fig, axs = plt.subplots(3, 2, figsize=(10, 10), sharex=True)
    for ax, (title, series) in zip(axs.flat, panels):
        for col, label in series:
            ax.plot(df["epoch"], df[col], label=label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if len(series) > 1:
            ax.legend()
    for ax in axs[-1]:
        ax.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_threshold_sweep(sweep: pd.DataFrame, chosen: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for col in ("f1", "precision", "recall", "accuracy"):
        ax.plot(sweep["threshold"], sweep[col], label=col.capitalize())
    ax.axvline(chosen, color="k", ls="--", lw=1, label=f"Chosen = {chosen:.2f}")
    ax.set_xlabel("Decision threshold on P(fall)")
    ax.set_ylabel("Score")
    ax.set_title("Threshold sweep (validation split)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_by_offset(score: dict[str, float], train_range: list[float] | None, ylabel: str, out_path: Path) -> None:
    offsets = [float(k) for k in score]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if train_range:
        ax.axvspan(*train_range, color="0.9", label="Training impact range")
    ax.plot(offsets, list(score.values()), "o-")
    ax.set_xlabel("Impact position in window (s)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    if train_range:
        ax.legend(loc="lower center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
    size = 3 + 0.6 * len(labels)
    fig, ax = plt.subplots(figsize=(size + 0.5, size))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Reds", ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
