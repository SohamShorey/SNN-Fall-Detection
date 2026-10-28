"""Threshold, rule-based and classical ML baselines on the same data and evaluation as the SCNN.

    python scripts/baselines.py                                   # fall detector
    python scripts/baselines.py --config configs/fall_direction.yaml   # fall direction
"""
import json
from datetime import datetime
from pathlib import Path

from _common import config_parser

from snn_fall.baselines import run_baselines
from snn_fall.config import load_config
from snn_fall.data import summarize
from snn_fall.pipeline import load_data
from snn_fall.train import seed_everything

if __name__ == "__main__":
    p = config_parser(__doc__)
    p.add_argument("--name", help="suffix for the run directory name")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)

    run_dir = Path(cfg.output_dir) / f"{datetime.now():%Y%m%d-%H%M%S}_classical{'_' + args.name if args.name else ''}"
    run_dir.mkdir(parents=True)
    cfg.save(run_dir / "config.yaml")
    data = load_data(cfg)
    print(f"Run directory: {run_dir}\nTask: {cfg.task.kind} {cfg.task.class_names}")
    print(summarize(data, cfg.task.class_names))

    results = run_baselines(data, cfg)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {run_dir}")
