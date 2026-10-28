"""Re-evaluate a finished run directory (threshold on val, metrics on test)."""
import argparse

import _common  # noqa: F401  (sets up the import path)

from snn_fall.pipeline import evaluate_run

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", help="e.g. runs/20261027-210000")
    evaluate_run(p.parse_args().run_dir)
