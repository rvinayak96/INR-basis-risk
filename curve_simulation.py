# simulation check

import json
from pathlib import Path

import numpy as np
import pandas as pd

import curve_model as cm


SCENARIOS = {
    "baseline": {},
    "no_turn": {"mu_D": 0.0, "sigma_w": 0.0, "sigma_xi": 0.0},
    "one_charge": {"mu_D": 10.0, "sigma_w": 0.0, "sigma_xi": 3.0}
}


# true parameters
def true_params(estimated, tenors, scenario="baseline"):
    names = cm.parameter_names(tenors)
    truth = np.asarray(estimated, dtype=float).copy()

    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if len(truth) != len(names):
        raise ValueError("Parameter vector does not match tenors")

    for name, value in SCENARIOS[scenario].items():
        truth[names.index(name)] = value
    return truth


# state errors
def state_errors(truth, estimated, which, rep):
    rows = []
    for state in cm.STATES:
        error = estimated[state] - truth[state]
        lower, upper = estimated[f"{state}_lo"], estimated[f"{state}_hi"]
        covered = (truth[state] >= lower) & (truth[state] <= upper)
        rows.append({
            "rep": rep, "estimate": which, "state": state,
            "rmse": float(np.sqrt((error ** 2).mean())),
            "bias": float(error.mean()),
            "coverage_90": float(covered.mean()),
            "band_width": float((upper - lower).mean()),
            "band_excludes_zero": float(((lower > 0) | (upper < 0)).mean())
        })
    return rows


# turn errors by month
def turn_by_month(truth, estimated, which, rep):
    error = estimated["turn"] - truth["turn"]
    covered = ((truth["turn"] >= estimated["turn_lo"])
               & (truth["turn"] <= estimated["turn_hi"]))

    df = pd.DataFrame({
        "month": estimated.index.month,
        "squared_error": error.to_numpy() ** 2,
        "covered": covered.to_numpy(),
        "width": (estimated["turn_hi"] - estimated["turn_lo"]).to_numpy()
    })
    out = df.groupby("month").agg(
        mse=("squared_error", "mean"),
        coverage_90=("covered", "mean"),
        band_width=("width", "mean")
    ).reset_index()
    out["rmse"] = np.sqrt(out.pop("mse"))
    out.insert(0, "estimate", which)
    out.insert(0, "rep", rep)
    return out


# simulate and refit
def run(data, estimated, scenario="baseline", reps=10, seed=1000,
        lam=cm.LAM, maxiter=1000, two_routes=True, folder=None):
    if reps < 1:
        raise ValueError("reps must be positive")

    names = cm.parameter_names(data["tenors"])
    truth = true_params(estimated, data["tenors"], scenario)
    index = data["basis"].index

    settings = {
        "scenario": scenario, "reps": reps, "seed": seed,
        "lambda": lam, "tenors": list(data["tenors"]),
        "n_dates": len(index),
        "sample_start": str(index.min().date()),
        "sample_end": str(index.max().date()),
        "maxiter": maxiter, "two_routes": two_routes
    }

    states, months, params = [], [], []
    # One re-estimation per simulated sample.
    for rep in range(reps):
        rng = np.random.default_rng(seed + rep)
        simulated, true_states = cm.simulate(truth, data, rng, lam=lam)
        _, fitted = cm.fit(simulated, lam=lam, maxiter=maxiter,
                           two_routes=two_routes)
        p = np.asarray(fitted.params)
        for which in ("filtered", "smoothed"):
            estimates = cm.states_table(fitted, index, which)
            states.extend(state_errors(true_states, estimates, which, rep))
            months.append(turn_by_month(true_states, estimates, which, rep))

        bound = [name for name in names
                 if (name.startswith("phi_") or name in ("rho", "kappa"))
                 and abs(p[names.index(name)]) > 0.998]
        params.append({
            "rep": rep, "loglik": float(fitted.llf),
            "converged": bool(fitted.mle_retvals.get("converged", False)),
            "at_bound": " ".join(bound), **dict(zip(names, p))
        })

    result = {
        "settings": settings, "truth": truth,
        "params": pd.DataFrame(params),
        "states": pd.DataFrame(states),
        "turn_by_month": pd.concat(months, ignore_index=True)
    }
    if folder is not None:
        save_results(result, folder)
    return result


# save
def save_results(result, folder):
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {folder}")
    folder.mkdir(parents=True, exist_ok=True)

    names = cm.parameter_names(result["settings"]["tenors"])
    pd.DataFrame({"param": names, "true": result["truth"]}).to_csv(
        folder / "truth.csv", index=False)
    for key in ("params", "states", "turn_by_month"):
        result[key].to_csv(folder / f"{key}.csv", index=False)
    with (folder / "settings.json").open("x", encoding="utf-8") as file:
        json.dump(result["settings"], file, indent=2)
    return folder
