"""Evaluate the nonpersonalized history-frequency reference, without model fitting."""

import argparse
import json

from common.baseline_run import configuration, load_development, make_output, save_result


def run(output_dir=None):
    config = configuration()
    _, validation = load_development()
    output = make_output("movie_high_rating_rate", output_dir)
    return save_result("movie_high_rating_rate", validation,
                       validation["frequency_score"].astype(float), config, output,
                       {"fit_seconds": 0, "tuning": "none; fixed history high-rating proportion"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))
