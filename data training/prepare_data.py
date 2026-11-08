"""Run from any working directory: python 'data training/prepare_data.py'."""

import argparse
import json
from pathlib import Path

from common.preprocessing import prepare, read_config


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Prepare one history-only MovieLens baseline split; no model training")
    parser.add_argument("--data-dir", type=Path, default=base.parent / "Lab3" / "ml-100k")
    parser.add_argument("--config", type=Path, default=base / "configs" / "preprocessing.json")
    parser.add_argument("--output-dir", type=Path, default=base / "outputs" / "preprocessed")
    parser.add_argument("--split-dir", type=Path, default=None)
    args = parser.parse_args()
    config = read_config(args.config)
    split_dir = args.split_dir or base / "splits" / config["split_name"]
    manifest = prepare(args.data_dir, args.output_dir, split_dir, args.config)
    print(json.dumps({key: manifest[key] for key in
                      ("dataset", "partition_counts", "eligible_counts", "excluded_counts", "history_movies")}, indent=2))
    print(f"Prepared inputs: {args.output_dir.resolve()}")
    print("No model trained; no test performance computed.")


if __name__ == "__main__":
    main()
