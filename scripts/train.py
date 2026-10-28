"""Run the full experiment: STDP pretraining, supervised training, and val/test evaluation."""
from _common import config_parser

from snn_fall.config import load_config
from snn_fall.pipeline import run_experiment

if __name__ == "__main__":
    p = config_parser(__doc__)
    p.add_argument("--name", help="suffix for the run directory name")
    args = p.parse_args()
    run_experiment(load_config(args.config, args.overrides), name=args.name)
