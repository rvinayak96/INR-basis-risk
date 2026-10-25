# result folders and saving

from datetime import datetime
from pathlib import Path
from shutil import copy2
from hashlib import sha256
from importlib.metadata import version
import json
import sys

import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


# new run folder
def new_run(results_dir=None, name=None):
    root = Path(results_dir) if results_dir is not None else RESULTS
    root.mkdir(parents=True, exist_ok=True)
    name = name or f"basis_run_{datetime.now():%Y%m%d_%H%M%S_%f}"
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
    folder = Path(run) / step
    folder.mkdir(exist_ok=False)
    return folder


# save settings
def save_settings(run, settings):
    with (Path(run) / "settings.json").open("x", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


# file fingerprint (sha256)
def file_hash(file):
    h = sha256()
    with Path(file).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# save everything from a run
def save_run(run, settings, outputs, input_files, project):
    run = Path(run)
    project = Path(project)
    save_settings(run, settings)

    for name, value in outputs.items():
        if value is None:
            continue
        if isinstance(value, (pd.DataFrame, pd.Series)):
            value.to_csv(run / f"{name}.csv", index=True)
        else:
            with (run / f"{name}.json").open("x", encoding="utf-8") as f:
                json.dump(value, f, indent=2)

    # copy of the code used
    code_folder = run / "code"
    code_folder.mkdir()
    scripts = sorted(project.glob("*.py"))
    notebook = project / "USDINR_Basis_risk.ipynb"
    if notebook.exists():
        scripts.append(notebook)
    for file in scripts:
        copy2(file, code_folder / file.name)

    # package versions
    versions = {"python": sys.version.split()[0]}
    for package in ["numpy", "pandas", "scipy", "statsmodels", "matplotlib"]:
        versions[package] = version(package)

    # what went in: versions, input and code hashes
    manifest = {
        "versions": versions,
        "inputs": {
            str(Path(file).resolve()): file_hash(file)
            for file in dict.fromkeys(input_files)
        },
        "code": {file.name: file_hash(file) for file in scripts}
    }
    with (run / "manifest.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return run
