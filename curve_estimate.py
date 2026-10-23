# estimation and model checks

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import curve_model as cm


# fiscal year
def fiscal_year(index):

    return np.where(
        index.month <= 3,
        index.year,
        index.year + 1
    )


# reset and rollover days
def turn_days(data):

    idx = data["basis"].index

    resets = list(idx[data["reset"]])

    rollovers = [
        idx[idx.get_loc(date) - 1]
        for date in resets
    ]

    return resets, rollovers


# turn price from FBD April
def observed_turn_price(data, max_days=20):

    if "1M" not in data["tenors"]:
        raise ValueError(
            "Turn-price proxy requires the 1M tenor"
        )

    one_month = data["basis"]["1M"].where(
        data["x"]["1M"] == 0
    )

    carry = (
        one_month.ffill()
        .rolling(20, min_periods=5)
        .median()
    )

    price = (
        data["fbd"]
        - data["fbd_n"] / 365 * carry
    )

    return price.where(data["fbd_n"] <= max_days)


# main fit + comparisons
def estimate(
    train,
    lam=cm.LAM,
    start=None,
    maxiter=1000,
    two_routes=True,
    other_lambdas=(0.2, 0.4),
    comparisons=True,
    fixed=None,
    phi_cap=0.95,
    turn_walk=True,
    use_fbd=True
):

    initial = (
        cm.start_values(train, lam, phi_cap)
        if start is None
        else np.asarray(start, dtype=float)
    )

    model, main = cm.fit(
        train,
        lam=lam,
        start=initial,
        fixed=fixed,
        turn_walk=turn_walk,
        use_fbd=use_fbd,
        maxiter=maxiter,
        two_routes=two_routes
    )

    fits = {"main": main}
    params = np.asarray(main.params)

    if comparisons:

        if turn_walk:
            fits["one_charge_per_season"] = cm.fit(
                train, lam=lam, turn_walk=False, use_fbd=use_fbd,
                start=params, fixed=fixed, maxiter=maxiter,
                two_routes=False
            )[1]

        if use_fbd:
            fits["curve_only"] = cm.fit(
                train, lam=lam, turn_walk=turn_walk, use_fbd=False,
                start=params, fixed=fixed, maxiter=maxiter,
                two_routes=False
            )[1]

        for value in other_lambdas:

            fits[f"lambda_{value}"] = cm.fit(
                train,
                lam=value,
                start=params,
                fixed=fixed,
                turn_walk=turn_walk,
                use_fbd=use_fbd,
                maxiter=maxiter,
                two_routes=False
            )[1]

    names = cm.parameter_names(train["tenors"])

    table = pd.DataFrame(
        {
            name: np.asarray(res.params)
            for name, res in fits.items()
        },
        index=names
    )

    summary = pd.DataFrame([
        {
            "fit": name,
            "loglik": res.llf,
            "converged": bool(
                res.mle_retvals.get("converged", False)
            )
        }
        for name, res in fits.items()
    ])

    summary["max_abs_gradient"] = np.nan

    summary.loc[0, "max_abs_gradient"] = (
        np.abs(model.score(params)).max()
    )

    return fits, table, summary


# forecast error check
def innovation_check(result, tenors, burn=50):

    m = len(tenors)

    v = result.forecasts_error[:m, burn:].T
    F = result.forecasts_error_cov[:m, :m, burn:]

    keep = np.isfinite(v).all(axis=1)

    v = v[keep]
    F = F[:, :, keep]

    if len(v) < 3:
        raise ValueError(
            "Too few complete innovations after burn-in"
        )

    actual = np.corrcoef(v.T)

    average_F = F.mean(axis=2)

    sd = np.sqrt(np.diag(average_F))
    implied = average_F / np.outer(sd, sd)

    whitened = np.array([
        np.linalg.solve(
            np.linalg.cholesky(F[:, :, t]),
            v[t]
        )
        for t in range(len(v))
    ])

    corr_white = np.corrcoef(whitened.T)

    upper = corr_white[
        np.triu_indices(m, 1)
    ]

    corr = pd.DataFrame({
        "pair": [
            f"{tenors[i]}-{tenors[i + 1]}"
            for i in range(m - 1)
        ],
        "corr_actual": [
            actual[i, i + 1]
            for i in range(m - 1)
        ],
        "corr_model": [
            implied[i, i + 1]
            for i in range(m - 1)
        ]
    })

    white = {
        "whitened_mean_abs_corr": float(
            np.abs(upper).mean()
        ),
        "whitened_max_abs_corr": float(
            np.abs(upper).max()
        ),
        "whitened_lag1_autocorr_mean": float(
            np.mean([
                pd.Series(whitened[:, i]).autocorr()
                for i in range(m)
            ])
        )
    }

    return corr, white


# turn by year
def turn_by_year(
    filtered, smoothed, two_step,
    observed, rollover_days, turn_year
):

    rows = []
    fy = turn_year.reindex(filtered.index).to_numpy()

    for year in sorted(set(fy)):
        roll = [day for day in rollover_days if day.year == year]
        if not roll:
            continue

        rollover = roll[0]
        pos = filtered.index.get_loc(rollover)
        if pos == 0:
            continue
        pre_rollover = filtered.index[pos - 1]

        feb = filtered[(fy == year) & (filtered.index.month == 2)]
        window = (fy == year) & (
            (filtered.index.month >= 10) | (filtered.index.month <= 3)
        )

        rows.append({
            "fiscal_year": year,
            "rollover_date": rollover,
            "pre_rollover_date": pre_rollover,
            "filtered_end_feb": feb["turn"].iloc[-1] if len(feb) else np.nan,
            "filtered_pre_rollover": filtered.loc[pre_rollover, "turn"],
            "smoothed_median_oct_mar": smoothed.loc[window, "turn"].median(),
            "observed_price_median": observed[fy == year].median(),
            "two_step": two_step[window].median()
        })

    return pd.DataFrame(rows)


# filter and checks
def analyse(
    data,
    params,
    curve_only_params=None,
    lam=cm.LAM,
    use_fbd=True,
    observed_turn_days=20,
    burn=50
):

    idx = data["basis"].index
    tenors = data["tenors"]
    m = len(tenors)

    resets, rollovers = turn_days(data)

    full = cm.run_filter(params, data, lam, use_fbd)

    smoothed = cm.states_table(
        full, idx, "smoothed"
    )

    filtered = cm.states_table(
        full, idx, "filtered"
    )

    one_step = pd.concat([
        pd.DataFrame({
            "date": idx,
            "tenor": tenor,
            "observed": data["basis"][tenor].values,
            "forecast_mean": full.forecasts[i],
            "forecast_var": full.forecasts_error_cov[i, i],
            "x": data["x"][tenor].values
        })
        for i, tenor in enumerate(tenors)
    ], ignore_index=True)

    common = np.einsum(
        "kst,st->tk",
        full.model["design"][:m, :4],
        full.smoothed_state[:4]
    )

    fit_by_tenor = pd.DataFrame({
        "tenor": tenors,
        "rmse_bp": np.sqrt(
            np.nanmean(
                (data["basis"].values - common) ** 2,
                axis=0
            )
        ),
        "sd_daily_change_bp": (
            data["basis"].diff().std().values
        )
    })

    corr, white = innovation_check(
        full, tenors, burn
    )

    observed = observed_turn_price(
        data, observed_turn_days
    )

    curve_only = None
    turn_validation = pd.DataFrame()

    if curve_only_params is not None:

        no_fbd = cm.run_filter(
            curve_only_params,
            data,
            lam,
            use_fbd=False
        )

        curve_only = cm.states_table(
            no_fbd, idx, "filtered"
        )

        valid = observed.notna()

        turn_validation = pd.DataFrame([
            {
                "estimate": label,
                "days": int(valid.sum()),
                "corr_with_observed": (
                    series[valid].corr(observed[valid])
                ),
                "rmse_bp": np.sqrt(
                    ((series[valid] - observed[valid]) ** 2).mean()
                ),
                "mean_gap_bp": (
                    series[valid] - observed[valid]
                ).mean()
            }
            for label, series in [
                (
                    "main model, filtered (uses FBD April)",
                    filtered["turn"]
                ),
                (
                    "curve-only model, filtered (excludes FBD April)",
                    curve_only["turn"]
                )
            ]
        ])

    preliminary_turn = (
        cm.cross_section(data, lam)[0]["turn"]
    )

    years = turn_by_year(
        filtered, smoothed,
        preliminary_turn,
        observed, rollovers, data["turn_year"]
    )

    return {
        "smoothed": smoothed,
        "filtered": filtered,
        "curve_only": curve_only,
        "one_step": one_step,
        "fit_by_tenor": fit_by_tenor,
        "innovation_corr": corr,
        "innovation_white": white,
        "observed_turn": observed,
        "turn_validation": turn_validation,
        "turn_by_year": years,
        "reset_days": resets,
        "rollover_days": rollovers
    }


# states plot
def plot_states(smoothed, observed, file):

    fig, axes = plt.subplots(
        4, 1, figsize=(10, 11), sharex=True
    )

    labels = {
        "level": "Level (annualized bp)",
        "slope": "Slope (annualized bp)",
        "curvature": "Curvature (annualized bp)",
        "turn": "Financial-year turn (bp of notional)"
    }

    for ax, state in zip(axes, cm.STATES):

        ax.plot(
            smoothed.index,
            smoothed[state],
            linewidth=0.8
        )

        ax.fill_between(
            smoothed.index,
            smoothed[f"{state}_lo"],
            smoothed[f"{state}_hi"],
            alpha=0.2
        )

        ax.set_title(labels[state])
        ax.axhline(0, color="grey", linewidth=0.5)

    axes[3].plot(
        observed.index,
        observed,
        "o",
        markersize=2.5
    )

    fig.tight_layout()
    fig.savefig(file, dpi=140)
    plt.close(fig)


# save
def save_outputs(
    folder, table, summary, profile,
    results, settings=None
):

    folder = Path(folder)

    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {folder}"
        )

    folder.mkdir(parents=True, exist_ok=True)

    table.to_csv(folder / "params.csv")

    summary.to_csv(
        folder / "fit_summary.csv", index=False
    )

    if profile is not None:
        profile.to_csv(
            folder / "phi_profile.csv", index=False
        )

    files = {
        "smoothed": "states_smoothed.csv",
        "filtered": "states_filtered.csv",
        "one_step": "one_step_forecasts.csv",
        "fit_by_tenor": "fit_by_tenor.csv",
        "innovation_corr": "innovation_correlation.csv",
        "turn_validation": "turn_validation.csv",
        "turn_by_year": "turn_by_fiscal_year.csv"
    }

    for key, file in files.items():

        results[key].to_csv(
            folder / file,
            index=(key in {"smoothed", "filtered"})
        )

    with (
        folder / "innovation_whitened.json"
    ).open("x", encoding="utf-8") as f:

        json.dump(
            results["innovation_white"],
            f,
            indent=2
        )

    if settings is not None:

        with (
            folder / "estimate_settings.json"
        ).open("x", encoding="utf-8") as f:

            json.dump(settings, f, indent=2)

    plot_states(
        results["smoothed"],
        results["observed_turn"],
        folder / "fig_states_smoothed.png"
    )

    return folder
