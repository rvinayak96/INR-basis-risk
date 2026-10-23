# result folders

import json
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


# new run folder
def new_run(results_dir=None, name=None):
    root = Path(results_dir) if results_dir is not None else RESULTS
    root.mkdir(parents=True, exist_ok=True)
    if name is None:
        name = f"basis_run_{datetime.now():%Y%m%d_%H%M%S}"
    run = root / name
    run.mkdir(exist_ok=False)
    return run


# latest run
def latest_run(results_dir=None):
    root = Path(results_dir) if results_dir is not None else RESULTS
    runs = sorted(root.glob("basis_run_*"))
    if not runs:
        raise FileNotFoundError(f"No basis run found in {root}")
    return runs[-1]


# step folder
def step_folder(run, step):
    out = Path(run) / step
    out.mkdir(parents=True, exist_ok=False)
    return out


# save settings
def save_settings(run, settings):
    file = Path(run) / "settings.json"
    with file.open("x", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
