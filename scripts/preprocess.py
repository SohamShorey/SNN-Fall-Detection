"""Build the windowed, normalised SisFall dataset cache."""
from _common import config_parser

from snn_fall.config import load_config
from snn_fall.data import cache_file, preprocess, summarize

if __name__ == "__main__":
    args = config_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.overrides)
    data = preprocess(cfg.data, cfg.seed)
    print(summarize(data))
    print(f"Skipped recordings: {data['meta']['skipped_recordings']}")
    print(f"Saved to {cache_file(cfg.data, cfg.seed)}")
