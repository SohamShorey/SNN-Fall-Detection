"""Metrics and decision-threshold selection.

The operating threshold is chosen on the validation split (max F1) and then
applied unchanged to the held-out test split, so test metrics stay unbiased.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, prob)
    except ValueError:  # only one class present
        auc = float("nan")
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def threshold_sweep(y_true: np.ndarray, prob: np.ndarray, thresholds=np.linspace(0.01, 0.99, 99)) -> pd.DataFrame:
    return pd.DataFrame([compute_metrics(y_true, prob, th) for th in thresholds])


def select_threshold(sweep: pd.DataFrame, metric: str = "f1") -> float:
    """Best threshold by `metric`; ties broken towards 0.5 (the uncalibrated default)."""
    best = sweep[sweep[metric] == sweep[metric].max()]
    return float(best.iloc[(best["threshold"] - 0.5).abs().argmin()]["threshold"])
