"""Compile every finished run into paper-ready result tables.

    python scripts/paper_tables.py

Reads runs/*/metrics.json and writes, in results/:
    all_results.csv   one row per model (every run), all metrics and costs
    tables.md         Markdown tables (to read or paste)
    tables.tex        LaTeX booktabs tables (\\usepackage{booktabs})

Tables: fall detection, fall direction, SNN efficiency variants, encoding-stream ablation.
All scores are on the held-out test subjects with falls at impact positions 1-3 s; F1,
precision and recall are macro-averaged for fall direction. When several runs share a
label, the most recent one is used.
"""
from pathlib import Path

import pandas as pd
import yaml
from compare import rows_for_run

OUT = Path("results")
CLASSICAL = {"svm_rbf": "SVM (RBF)", "random_forest": "Random Forest", "knn": "k-NN",
             "threshold_acc": "Threshold (acc. peak)", "threshold_acc_gyro": "Threshold (acc. + gyro peaks)",
             "rule_impact_dir": "Impact-direction rule"}
PATH_NAMES = {"ttfs": "TTFS", "lc": "LC", "fs1": "FS1"}
TINY = {"norm": "none", "t_max": 10, "head_dim": 32, "hidden_dim": 32}


def describe(run: Path, model: str) -> dict:
    """Readable label, group and variant for one result row, derived from the run's config."""
    if model in CLASSICAL:
        group = "Rule-based" if model.startswith(("threshold", "rule")) else "Classical ML"
        return {"label": CLASSICAL[model], "group": group, "size": "", "streams": ""}
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    m, e, d, t = cfg.get("model", {}), cfg.get("encoding", {}), cfg.get("data", {}), cfg.get("train", {})
    knobs = {"norm": m.get("norm", "group"), "t_max": e.get("t_max", 20),
             "head_dim": m.get("head_dim", 64), "hidden_dim": m.get("hidden_dim", 128)}
    paths = m.get("paths", ["ttfs", "lc", "fs1"])
    streams = "+".join(PATH_NAMES[p] for p in ("ttfs", "lc", "fs1") if p in paths)
    if knobs == TINY:
        size = "tiny"
    else:
        changes = []
        if knobs["head_dim"] != 64 or knobs["hidden_dim"] != 128:
            changes.append(f"head {knobs['head_dim']}, hidden {knobs['hidden_dim']}")
        if knobs["norm"] == "none":
            changes.append("no GroupNorm")
        if knobs["t_max"] != 20:
            changes.append(f"T={knobs['t_max']}")
        size = "full" if not changes else "; ".join(changes)
    name = "SCNN" if m.get("kind", "scnn") == "scnn" else "ANN twin"
    group = "SNN (ours)" if name == "SCNN" else "ANN twin"
    if not d.get("highpass_causal", False) or not t.get("fall_shift_s"):
        size = "original pipeline"
        group = "SNN (original)"
    label = name if size == "full" else f"{name} ({size})"
    if group == "SNN (original)":
        label = f"{name}, original pipeline (aligned windows, zero-phase filter)"
    if streams != "TTFS+LC+FS1":
        label += f" [{streams}]"
    return {"label": label, "group": group, "size": size, "streams": streams}


def collect() -> pd.DataFrame:
    rows = []
    for run in sorted(Path("runs").glob("*/")):
        if not (run / "metrics.json").exists():
            continue
        try:
            for r in rows_for_run(run):
                rows.append({**r, **describe(run, r["model"])})
        except KeyError as err:  # older metrics.json formats
            print(f"Skipping {run.name}: missing {err}")
    df = pd.DataFrame(rows)
    df["total_energy_nj"] = df["energy_nj"] + df["feature_energy_nj"]
    df["task"] = df["task"].map({"binary": "Fall detection", "fall_category": "Fall direction"})
    # newest run wins when the same model/configuration was trained more than once
    return df.sort_values("run").drop_duplicates(["task", "label"], keep="last").reset_index(drop=True)


def table(df: pd.DataFrame, direction: bool) -> pd.DataFrame:
    f1 = "Macro-F1 (%)" if direction else "F1 (%)"
    out = pd.DataFrame({
        "Model": df["label"],
        "Params": df["parameters"].map(lambda v: f"{v / 1e3:.1f}K" if v >= 1000 else f"{v:.0f}"),
        f1: df["test_f1"] * 100,
        "Precision (%)": df["test_precision"] * 100,
        "Recall (%)": df["test_recall"] * 100,
        "AUC (%)": df["test_auc"] * 100,
        "Model energy (nJ)": df["energy_nj"],
        "Input energy (nJ)": df["feature_energy_nj"],
        "Total energy (nJ)": df["total_energy_nj"],
    })
    return out


SECTIONS = [
    ("Fall detection: all models", "Fall detection",
     lambda d: d[(d.streams.isin(["", "TTFS+LC+FS1"])) & (d["size"].isin(["", "full", "tiny"])
                 | d.group.eq("SNN (original)"))]),
    ("Fall direction: all models", "Fall direction", lambda d: d),
    ("Fall detection: SNN efficiency variants", "Fall detection",
     lambda d: d[d.group.eq("SNN (ours)") & d.streams.eq("TTFS+LC+FS1")]),
    ("Fall detection: encoding-stream ablation", "Fall detection",
     lambda d: d[d.group.eq("SNN (ours)") & d["size"].isin(["full", "tiny"])]),
]


def to_latex(t: pd.DataFrame, caption: str) -> str:
    cols = "l" + "r" * (len(t.columns) - 1)
    body = t.to_latex(index=False, escape=True, na_rep="--", float_format="%.2f", column_format=cols)
    body = body.replace("\\begin{tabular}", "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}").replace(
        "\\end{tabular}", "\\end{tabular}}")
    return (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n{body}\\end{{table}}\n")


if __name__ == "__main__":
    df = collect()
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "all_results.csv", index=False)
    md, tex = ["# Results (test subjects, falls at impact positions 1-3 s)\n"], []
    for title, task, pick in SECTIONS:
        sub = pick(df[df.task == task]).sort_values("test_f1", ascending=False)
        if sub.empty:
            continue
        t = table(sub, direction=task == "Fall direction")
        md += [f"## {title}\n", t.to_string(index=False, float_format=lambda v: f"{v:.2f}", na_rep="--"), "\n"]
        tex.append(to_latex(t, title))
        print(f"\n=== {title} ===")
        print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}", na_rep="--"))
    (OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "tables.tex").write_text("\n".join(tex), encoding="utf-8")
    print(f"\nWrote {OUT / 'all_results.csv'}, {OUT / 'tables.md'}, {OUT / 'tables.tex'}")
