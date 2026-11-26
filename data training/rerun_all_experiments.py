"""Reproduce the fixed development experiments from scratch, never opening test.

Archive old generated outputs before invoking this script. It refuses any
pre-existing formal model output. This is reproducibility, not extra samples,
new seeds, new tuning, or an independent unseen test.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent
CONTROL = BASE / "outputs/full_rerun_20261125"

STAGES = [
    ("tests", None, []),
    ("floor", "movie_high_rating_rate.py", []),
    ("lr", "logistic_regression.py", []),
    ("knn_original_grid", "knn.py", []),
    ("gnn", "gnn.py", ["--device", "cuda"]),
    ("gnn_repeat", "gnn.py", ["--device", "cuda", "--output-dir", str(BASE / "outputs/gnn_repeat_verified")]),
    ("knn_comparison", "compare_knn.py", []),
    ("gnn_comparison", "compare_gnn.py", []),
    ("genre_training_and_sanity", "audit_training_sanity.py", ["--run-genre", "--output-dir", str(BASE / "outputs/genre_sanity_verified")]),
    ("genre_comparison", "compare_genre.py", []),
    ("genre_report", "build_genre_report.py", []),
    ("gnn_capacity", "gnn_extensions.py", ["--device", "cuda"]),
    ("gnn_capacity_comparison", "compare_gnn_extensions.py", []),
    ("knn_extended_grid", "knn_grid_extension.py", []),
    ("knn_extended_comparison", "compare_knn_grid_extension.py", []),
]

# Every child, including tests, gets an audit hook before importing project code.
# Synthetic test fixtures outside the real data root remain allowed.
GUARDED = r'''
import json, pathlib, runpy, sys
script, log = sys.argv[1:3]
root = pathlib.Path(sys.argv[3]).resolve()
opened = set()
blocked = []
def guard(event, args):
    if event == "open" and isinstance(args[0], (str, bytes)):
        p = pathlib.Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
        if root in p.parents:
            rel = p.relative_to(root).as_posix()
            if rel == "test_inputs.npz" or rel.startswith("sealed/"):
                blocked.append(rel)
                raise PermissionError("Full rerun development stage blocked real test access")
            opened.add(rel)
sys.addaudithook(guard)
sys.path.insert(0, str(pathlib.Path(script).parent))
arguments = sys.argv[4:]
try:
    if script.endswith("__unittest__"):
        import unittest
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(arguments[0]))
        if not result.wasSuccessful():
            raise SystemExit(1)
    else:
        sys.argv = [script] + arguments
        runpy.run_path(script, run_name="__main__")
finally:
    pathlib.Path(log).write_text(json.dumps({"opened": sorted(opened), "blocked": blocked,
        "test_accessed": False}, indent=2) + "\n", encoding="utf-8")
'''


def main():
    if CONTROL.exists():
        raise FileExistsError(f"Refusing to restart/overwrite rerun: {CONTROL}")
    for name in ("movie_high_rating_rate", "logistic_regression", "knn", "gnn",
                 "gnn_repeat_verified", "logistic_regression_genre", "knn_genre",
                 "gnn_extensions", "knn_grid_extension"):
        if (BASE / "outputs" / name).exists():
            raise FileExistsError(f"Archive old output first: {name}")
    CONTROL.mkdir(parents=True)
    state = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "RUNNING",
             "purpose": "Fixed-protocol full reproducibility run; test was already observed in prior run",
             "data_or_split_rebuilt": False, "stages": []}
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    for name, script, arguments in STAGES:
        script_path = str(BASE / (script or "__unittest__"))
        if script is None:
            arguments = [str(BASE / "tests")]
        command = [sys.executable, "-u", "-c", GUARDED, script_path,
                   str(CONTROL / f"{name}_access.json"), str(BASE / "outputs/preprocessed"), *arguments]
        stage = {"name": name, "started_utc": datetime.now(timezone.utc).isoformat()}
        print(f"START {name}", flush=True)
        with (CONTROL / f"{name}.log").open("x", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=PROJECT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        stage.update(exit_code=result.returncode, finished_utc=datetime.now(timezone.utc).isoformat())
        state["stages"].append(stage)
        state["status"] = "FAIL" if result.returncode else "RUNNING"
        (CONTROL / "status.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        print(f"END {name}: exit={result.returncode}", flush=True)
        if result.returncode:
            raise SystemExit(f"Stopped; inspect {CONTROL / (name + '.log')}; no automatic retries")
    state.update(status="PASS", finished_utc=datetime.now(timezone.utc).isoformat())
    (CONTROL / "status.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
