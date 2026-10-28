"""Collect test results from run directories into one comparison table per task.

    python scripts/compare.py                       # every run under runs/
    python scripts/compare.py runs/A runs/B ...     # selected runs

Writes runs/comparison.csv and prints a table per task. Energy is per inference
(compute only); for the classical models the feature-extraction cost is listed separately.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def rows_for_run(run: Path) -> list[dict]:
    m = json.loads((run / "metrics.json").read_text())
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    base = {"run": run.name, "task": m.get("task", cfg.get("task", {}).get("kind", "binary"))}

    def scores(block: dict) -> dict:
        t = block["test"]
        by = block.get("test_fall_correct_by_impact_offset_s") or block.get("test_recall_by_impact_offset_s") or {}
        in_range = [v for k, v in by.items() if 1.0 <= float(k) <= 3.0]
        return {"test_f1": t["f1"], "test_precision": t["precision"], "test_recall": t["recall"],
                "test_accuracy": t["accuracy"], "test_auc": t["auc"],
                "aligned_f1": block.get("test_aligned", {}).get("f1"),
                "min_fall_score_1to3s": min(in_range) if in_range else None}

    if "models" in m:  # classical baselines
        feat = m["feature_extraction"]["energy_nj"]
        return [{**base, "model": name, **scores(b), "parameters": b["parameters"],
                 "mac": b["mac"], "ac_or_cmp": b["comparisons"], "energy_nj": b["energy_nj"],
                 "feature_energy_nj": feat} for name, b in m["models"].items()]

    eff = m.get("efficiency", {}).get("per_inference", {})
    spiking = "snn_synaptic_ac" in eff
    kind = cfg.get("model", {}).get("kind", "scnn")
    return [{**base, "model": "SCNN" if kind == "scnn" else "ANN twin", **scores(m), "parameters": m["parameters"],
             "mac": eff.get("snn_dense_mac") if spiking else eff.get("ann_synaptic_mac", 0) + eff.get("ann_dense_mac", 0),
             "ac_or_cmp": eff.get("snn_synaptic_ac") if spiking else eff.get("ann_dense_ac"),
             "energy_nj": eff.get("snn_total_energy_nj") if spiking else eff.get("ann_total_energy_nj"),
             "feature_energy_nj": None}]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="run directories (default: all under runs/)")
    args = p.parse_args()
    runs = [Path(r) for r in args.runs] or sorted(Path("runs").glob("*/"))
    rows = []
    for r in runs:
        if (r / "metrics.json").exists():
            try:
                rows += rows_for_run(r)
            except KeyError as e:  # older metrics.json formats
                print(f"Skipping {r.name}: missing {e}")
    df = pd.DataFrame(rows)
    df.to_csv(Path("runs") / "comparison.csv", index=False)
    cols = ["model", "test_f1", "test_precision", "test_recall", "test_auc", "aligned_f1",
            "min_fall_score_1to3s", "parameters", "mac", "ac_or_cmp", "energy_nj", "feature_energy_nj", "run"]
    for task, g in df.groupby("task"):
        print(f"\n=== {task}  (test subjects; F1 = macro-F1 for multi-class) ===")
        with pd.option_context("display.float_format", "{:.4g}".format, "display.width", 250):
            print(g.sort_values("test_f1", ascending=False)[cols].to_string(index=False))
    print("\nWritten to runs/comparison.csv")
