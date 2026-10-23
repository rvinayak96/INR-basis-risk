# LR test for the turn

import numpy as np
import pandas as pd
from pathlib import Path

from scipy.stats import chi2

import curve_model as cm


NO_TURN = {
    "mu_D": 0.0,
    "sigma_w": 1e-6,
    "sigma_xi": 1e-6
}


# no-turn fit
def restricted_fit(data, start, lam=cm.LAM, maxiter=1000):

    _, result = cm.mle(
        data,
        start=start,
        fixed=NO_TURN,
        lam=lam,
        maxiter=maxiter
    )

    return result


# LR
def compare(
    data,
    full_loglik,
    start,
    label="real data",
    lam=cm.LAM,
    maxiter=1000,
    full_converged=True
):

    restricted = restricted_fit(
        data, start, lam=lam, maxiter=maxiter
    )

    raw_lr = 2 * (full_loglik - restricted.llf)

    lr = max(0.0, raw_lr)

    return {
        "sample": label,
        "loglik_turn": float(full_loglik),
        "loglik_no_turn": float(restricted.llf),
        "LR": float(lr),
        "raw_LR": float(raw_lr),
        "full_converged": (
            np.nan if full_converged is None else bool(full_converged)
        ),
        "restricted_converged": bool(
            restricted.mle_retvals.get("converged", False)
        )
    }


# LR test
def run(data, full_params, full_loglik, null_simulations=None,
        seed=1000, lam=cm.LAM, maxiter=1000,
        full_converged=True, folder=None):
    rows = [compare(
        data, full_loglik, full_params,
        lam=lam, maxiter=maxiter,
        full_converged=full_converged
    )]

    if null_simulations is not None:
        settings = null_simulations["settings"]
        if settings["scenario"] != "no_turn":
            raise ValueError("Supply no_turn simulation results")
        if settings["seed"] != seed or settings["lambda"] != lam:
            raise ValueError("Simulation seed or lambda differs")
        if settings["tenors"] != list(data["tenors"]):
            raise ValueError("Simulation tenor selection differs")
        if settings["n_dates"] != len(data["basis"]):
            raise ValueError("Simulation sample length differs")
        if settings["sample_start"] != str(data["basis"].index.min().date()):
            raise ValueError("Simulation sample start differs")
        if settings["sample_end"] != str(data["basis"].index.max().date()):
            raise ValueError("Simulation sample end differs")

        names = cm.parameter_names(data["tenors"])
        truth = np.asarray(null_simulations["truth"], dtype=float)
        fitted = null_simulations["params"]

        if len(truth) != len(names):
            raise ValueError("Simulation parameter vector has wrong length")
        if any(abs(truth[names.index(name)]) > 1e-12 for name in NO_TURN):
            raise ValueError("Supply simulations from the no_turn scenario")

        for _, rep_row in fitted.iterrows():
            rep = int(rep_row["rep"])
            simulated, _ = cm.simulate(
                truth, data, np.random.default_rng(seed + rep), lam=lam
            )
            starting = rep_row[names].to_numpy(dtype=float)
            rows.append(compare(
                simulated,
                full_loglik=float(rep_row["loglik"]),
                start=starting,
                label=f"simulated no-turn rep {rep}",
                lam=lam,
                maxiter=maxiter,
                full_converged=bool(rep_row["converged"]),
            ))

    result = pd.DataFrame(rows)
    cutoff = chi2.ppf(0.95, 3)
    result["chi2_3_reference_5pct"] = cutoff
    result["above_chi2_reference"] = result["LR"] > cutoff


    result["mc_p_value"] = np.nan
    result["mc_reps"] = 0
    if len(result) > 1:
        simulated_lr = result.loc[1:, "LR"]
        real_lr = result.loc[0, "LR"]
        p_mc = (1 + (simulated_lr >= real_lr).sum()) / (len(simulated_lr) + 1)
        result.loc[0, "mc_p_value"] = float(p_mc)
        result.loc[0, "mc_reps"] = len(simulated_lr)
        result.attrs["mc_p_value"] = float(p_mc)

    if folder is not None:
        folder = Path(folder)
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"Output folder is not empty: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
        result.to_csv(folder / "turn_lr_test.csv", index=False)

    return result
