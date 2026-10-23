# VaR / ES models and backtests

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


TENORS = ["1M", "3M", "12M"]
TRAIN_END, TEST_START = "2023-12-31", "2024-01-01"
LEVELS = [0.95, 0.99]
WINDOW = 250
RM_LAMBDA = 0.94
EWMA_GRID = [0.60, 0.70, 0.80, 0.85, 0.90, 0.94, 0.97]
MODELS = ["RW-normal", "HS", "EWMA-normal", "EWMA-t-cal", "SS-normal", "SS-t-EWMA", "SS-FHS"]


# EWMA variance
def ewma_var(x, lam, start):
    v, prev = np.empty(len(x)), start
    for i in range(len(x)):
        v[i] = prev
        if not np.isnan(x[i]):
            prev = lam * prev + (1 - lam) * x[i] ** 2
    return v


# unit-variance t
def t_unit(nu):
    return stats.t(nu, scale=np.sqrt((nu - 2) / nu))


# t degrees of freedom
def fit_nu(z):
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) < 30 or z.std() <= 0:
        raise ValueError("Need at least 30 variable training residuals")
    return float(np.clip(stats.t.fit(z / z.std(), floc=0)[0], 2.5, 50))


# forecast mean and sd
def build_forecasts(one, turn_avg, tenors=TENORS, train_end=TRAIN_END,
                    window=WINDOW, rm_lambda=RM_LAMBDA):
    out = {}
    for ten in tenors:
        d = one[one["tenor"] == ten].set_index("date").sort_index()
        dy = d["observed"].diff()
        start = dy.loc[:train_end].dropna().iloc[:60].var()
        if not np.isfinite(start) or start <= 0:
            raise ValueError(f"Insufficient training changes for {ten}")
        f = pd.DataFrame({"change": dy})
        f["RW-normal_mean"], f["RW-normal_sd"] = 0.0, dy.rolling(window).std().shift()
        f["EWMA-normal_mean"], f["EWMA-normal_sd"] = 0.0, np.sqrt(ewma_var(dy.values, rm_lambda, start))
        f["EWMA-t-cal_mean"] = turn_avg * d["x"].diff()
        f["EWMA-t-cal_sd"] = np.sqrt(ewma_var((dy - f["EWMA-t-cal_mean"]).values, rm_lambda, start))
        f["SS-normal_mean"] = d["forecast_mean"] - d["observed"].shift()
        f["SS-normal_sd"] = np.sqrt(d["forecast_var"])
        out[ten] = f
    return out


# scaled SS forecasts
def add_ss_t_ewma(fc, weights):
    for tenor, f in fc.items():
        lam = weights[tenor] if isinstance(weights, dict) else weights
        z = (f["change"] - f["SS-normal_mean"]) / f["SS-normal_sd"]
        scale = ewma_var(z.to_numpy(), lam, 1.0)
        f["SS-t-EWMA_mean"] = f["SS-normal_mean"]
        f["SS-t-EWMA_sd"] = f["SS-normal_sd"] * np.sqrt(scale)
        f["SS-FHS_mean"] = f["SS-t-EWMA_mean"]
        f["SS-FHS_sd"] = f["SS-t-EWMA_sd"]


# pick EWMA weight and nu
def choose_settings(fc, train_end=TRAIN_END, grid_weights=EWMA_GRID):
    rows, nus, weights = [], {}, {}

    for tenor, f in fc.items():
        for lam in grid_weights:
            add_ss_t_ewma({tenor: f}, lam)
            z = ((f["change"] - f["SS-t-EWMA_mean"]) / f["SS-t-EWMA_sd"]).loc[:train_end]
            sd = f["SS-t-EWMA_sd"].loc[:train_end]
            good = np.isfinite(z) & np.isfinite(sd) & (sd > 0)
            nu = fit_nu(z[good])
            ll = (t_unit(nu).logpdf(z[good]) - np.log(sd[good])).sum()
            rows.append({"tenor": tenor, "ewma_weight": lam,
                         "nu": nu, "train_loglik": ll})

        grid_tenor = pd.DataFrame(rows)
        grid_tenor = grid_tenor[grid_tenor["tenor"] == tenor]
        best = grid_tenor.loc[grid_tenor["train_loglik"].idxmax()]
        weights[tenor] = float(best["ewma_weight"])

        zc = ((f["change"] - f["EWMA-t-cal_mean"]) / f["EWMA-t-cal_sd"]).loc[:train_end]
        nus[tenor] = {
            "SS-t-EWMA": float(best["nu"]),
            "EWMA-t-cal": fit_nu(zc)
        }

    add_ss_t_ewma(fc, weights)
    return nus, weights, pd.DataFrame(rows)


# VaR and ES
def var_es(mean, sd, level, nu=None):
    a = 1 - level
    if nu is None:
        z = stats.norm.ppf(level)
        return mean + sd * z, mean + sd * stats.norm.pdf(z) / a
    q, s = stats.t.ppf(level, nu), np.sqrt((nu - 2) / nu)
    return mean + sd * q * s, mean + sd * s * (nu + q ** 2) / (nu - 1) * stats.t.pdf(q, nu) / a


# prob of a bigger change
def upper_prob(y, mean, sd, nu=None):
    z = (y - mean) / sd
    return stats.norm.sf(z) if nu is None else t_unit(nu).sf(z)


# HS / FHS
def empirical_forecast(x, level, window=None, min_obs=WINDOW):
    v, e, u = np.full(len(x), np.nan), np.full(len(x), np.nan), np.full(len(x), np.nan)
    for i in range(min_obs + 1, len(x)):
        past = x[max(0, i - window):i] if window else x[:i]
        past = past[~np.isnan(past)]
        if len(past) < min_obs or np.isnan(x[i]):
            continue
        v[i] = np.quantile(past, level)
        e[i] = past[past >= v[i]].mean()
        u[i] = (past > x[i]).mean()
    return v, e, u


# Kupiec
def kupiec(hits, a):
    n, x = len(hits), hits.sum()
    ll0 = (n - x) * np.log(1 - a) + x * np.log(a)
    ll1 = 0.0 if x in (0, n) else (n - x) * np.log(1 - x / n) + x * np.log(x / n)
    return 1 - stats.chi2.cdf(-2 * (ll0 - ll1), 1)


# Christoffersen
def christoffersen(hits):
    h = hits.astype(int)
    n00, n01 = ((h[:-1] == 0) & (h[1:] == 0)).sum(), ((h[:-1] == 0) & (h[1:] == 1)).sum()
    n10, n11 = ((h[:-1] == 1) & (h[1:] == 0)).sum(), ((h[:-1] == 1) & (h[1:] == 1)).sum()
    ll = lambda q, a, b: a * np.log(1 - q) + b * np.log(q) if 0 < q < 1 else 0.0
    p01, p11, p = n01 / max(n00 + n01, 1), n11 / max(n10 + n11, 1), (n01 + n11) / max(len(h) - 1, 1)
    lr = -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    return 1 - stats.chi2.cdf(lr, 1)


# Basel zone
def traffic_light(breaches, days, a):
    p = stats.binom.cdf(breaches, days, a)
    return "green" if p < 0.95 else ("yellow" if p < 0.9999 else "red")


# ES test
def du_escanciano(u, a):
    u = u[~np.isnan(u)]
    H = (a - u) * (u < a) / a
    stat = (H.mean() - a / 2) / np.sqrt(a * (1 / 3 - a / 4) / len(H))
    return stat, 2 * stats.norm.sf(abs(stat))


# tick loss
def tick_loss(y, var, level):
    return (level - (y < var)) * (y - var)


# FZ0 loss
def fz0_loss(y, var, es, a):
    return (y > var) * (y - var) / (a * es) + var / es + np.log(es) - 1


# DM test
def diebold_mariano(loss, bench):
    d = bench - loss
    fit = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return d.mean(), fit.pvalues[0]


# VaR/ES for one model
def forecasts_for(full, f, model, level, nus, window=WINDOW):
    pos = full.index.get_indexer(f.index)
    y = f["change"].values
    if model == "HS":
        v, e, u = empirical_forecast(full["change"].values, level, window=window, min_obs=window)
        return v[pos], e[pos], u[pos]
    if model == "SS-FHS":
        z = ((full["change"] - full["SS-FHS_mean"]) / full["SS-FHS_sd"]).values
        q, tail, u = empirical_forecast(z, level, min_obs=window)
        mean, sd = f["SS-FHS_mean"].values, f["SS-FHS_sd"].values
        return mean + sd * q[pos], mean + sd * tail[pos], u[pos]
    mean, sd, nu = f[f"{model}_mean"].values, f[f"{model}_sd"].values, nus.get(model)
    v, e = var_es(mean, sd, level, nu)
    return v, e, upper_prob(y, mean, sd, nu)


# backtests and comparisons
def evaluate(fc, nus, reset_days, levels=LEVELS, models=MODELS,
             test_start=TEST_START, window=WINDOW):
    rows, comparisons = [], []

    for tenor, full in fc.items():
        test = full.loc[test_start:].dropna(subset=["change"])
        y = test["change"].to_numpy()
        dates = test.index

        turns = np.zeros(len(test), dtype=bool)
        for day in reset_days:
            i = dates.searchsorted(pd.Timestamp(day))
            turns[max(i - 1, 0):min(i + 1, len(test))] = True

        for level in levels:
            a = 1 - level
            forecasts = {}

            for name in models:
                var, es, prob = forecasts_for(
                    full, test, name, level, nus[tenor], window
                )
                forecasts[name] = (var, es)

                good = (np.isfinite(y) & np.isfinite(var)
                        & np.isfinite(es) & np.isfinite(prob))
                if good.sum() < 5:
                    raise ValueError(f"Too few test forecasts: {tenor}, {name}")

                hits = y[good] > var[good]
                stat, p_es = du_escanciano(prob[good], a)
                rows.append({
                    "tenor": tenor, "level": level, "model": name,
                    "days": int(good.sum()), "expected": a * good.sum(),
                    "breaches": int(hits.sum()),
                    "zone": traffic_light(int(hits.sum()), int(good.sum()), a),
                    "kupiec_p": kupiec(hits, a),
                    "independence_p": christoffersen(hits),
                    "es_test_stat": stat, "es_test_p": p_es,
                    "breaches_turn_window": int(hits[turns[good]].sum()),
                    "mean_var": float(np.mean(var[good]))
                })

            benchmarks = {}
            for benchmark in ("EWMA-normal", "HS"):
                if benchmark in models:
                    benchmarks[benchmark] = [n for n in models if n != benchmark]
            if "EWMA-t-cal" in models:
                benchmarks["EWMA-t-cal"] = [
                    n for n in ("SS-t-EWMA", "SS-FHS") if n in models
                ]

            for bench, alternatives in benchmarks.items():
                if bench not in forecasts:
                    continue
                bv, be = forecasts[bench]
                for name in alternatives:
                    v, e = forecasts[name]
                    good = (np.isfinite(y) & np.isfinite(v)
                            & np.isfinite(e) & np.isfinite(bv) & np.isfinite(be))
                    if good.sum() < 10:
                        continue
                    tick_gain, tick_p = diebold_mariano(
                        tick_loss(y[good], v[good], level),
                        tick_loss(y[good], bv[good], level)
                    )
                    # Keep the sample specific to this pair of models.
                    es_good = good & (e > 0) & (be > 0)
                    if es_good.sum() >= 10:
                        fz_gain, fz_p = diebold_mariano(
                            fz0_loss(y[es_good], v[es_good], e[es_good], a),
                            fz0_loss(y[es_good], bv[es_good], be[es_good], a)
                        )
                    else:
                        fz_gain, fz_p = np.nan, np.nan
                    comparisons.append({
                        "tenor": tenor, "level": level, "model": name,
                        "benchmark": bench, "tick_gain": tick_gain,
                        "tick_p": tick_p, "fz0_gain": fz_gain,
                        "fz0_p": fz_p, "tick_days": int(good.sum()),
                        "fz0_days": int(es_good.sum()),
                        "fz0_days_dropped": int(good.sum() - es_good.sum())
                    })

    return pd.DataFrame(rows), pd.DataFrame(comparisons)

# VaR plot
def plot_var(fc, nus, path, test_start=TEST_START, window=WINDOW, models=MODELS):
    fig, axes = plt.subplots(len(fc), 1, figsize=(10, 2.7 * len(fc)), sharex=True)
    for ax, (tenor, full) in zip(np.atleast_1d(axes), fc.items()):
        test = full.loc[test_start:].dropna(subset=["change"])
        ax.plot(test.index, test["change"], color="grey", lw=0.6, label="Daily change")
        for name, color in [("EWMA-normal", "#eb6834"), ("SS-FHS", "#2a78d6")]:
            if name not in models:
                continue
            v, _, _ = forecasts_for(full, test, name, 0.99, nus[tenor], window)
            ax.plot(test.index, v, color=color, lw=0.9, label=f"99% VaR: {name}")
        ax.set_title(f"{tenor}: one-day basis change and upper-tail VaR (bp)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# run everything
def run(one_step, turn_avg, reset_days, tenors=TENORS,
        train_end=TRAIN_END, test_start=TEST_START,
        levels=LEVELS, window=WINDOW, rm_lambda=RM_LAMBDA,
        ewma_grid=EWMA_GRID, models=MODELS):
    if pd.Timestamp(test_start) <= pd.Timestamp(train_end):
        raise ValueError("Test period must start after training ends")
    forecasts = build_forecasts(
        one_step, turn_avg, tenors, train_end, window, rm_lambda
    )
    nus, weight, tuning = choose_settings(forecasts, train_end, ewma_grid)
    table, comparisons = evaluate(
        forecasts, nus, reset_days, levels, models, test_start, window
    )
    settings = {
        "train_end": train_end, "test_start": test_start,
        "tenors": list(tenors), "turn_avg_bp": float(turn_avg),
        "ewma_weight": weight, "rm_lambda": rm_lambda,
        "window": window, "nus": nus, "models": list(models),
        "ewma_grid": list(ewma_grid)
    }
    return {
        "backtests": table, "loss_comparison": comparisons,
        "tuning": tuning, "forecasts": forecasts,
        "settings": settings
    }


# save
def save_results(result, folder):
    import json
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    result["backtests"].to_csv(folder / "backtests.csv", index=False)
    result["loss_comparison"].to_csv(folder / "loss_comparison.csv", index=False)
    result["tuning"].to_csv(folder / "ewma_weight_choice.csv", index=False)
    with (folder / "settings.json").open("x", encoding="utf-8") as f:
        json.dump(result["settings"], f, indent=2)
    plot_var(
        result["forecasts"], result["settings"]["nus"],
        folder / "fig_var_99.png", result["settings"]["test_start"],
        result["settings"]["window"], result["settings"]["models"]
    )
    return folder
