# rates vs basis, half-lives, local projections

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR


PERIODS = {
    "2018-19": ("2018", "2019"),
    "2020-21": ("2020", "2021"),
    "2022-24": ("2022", "2024"),
    "2025-26": ("2025", "2026"),
    "full": ("2018", "2026"),
}

DRIVERS = ["rbi_spot", "d_fwd", "d_liq", "d_usd"]


# rates vs basis
def covariance_allocation(data, tenors=("1M", "3M", "6M", "12M"),
                          periods=None):
    periods = PERIODS if periods is None else periods
    rows = []

    for freq, rule in [("daily", None), ("weekly", "W-FRI"),
                       ("month-end", "ME")]:
        prem = data["prem"]
        implied = data["implied"]
        basis = data["basis"]

        if rule is not None:
            prem = prem.resample(rule).last()
            implied = implied.resample(rule).last()
            basis = basis.resample(rule).last()

        dp = prem.diff()
        di = implied.diff()
        db = basis.diff()

        for period, (start, end) in periods.items():
            for tenor in tenors:
                p = dp.loc[start:end, tenor]
                i = di.loc[start:end, tenor]
                b = db.loc[start:end, tenor]

                valid = p.notna() & i.notna() & b.notna()
                p, i, b = p[valid], i[valid], b[valid]

                variance = p.var()

                if len(p) < 3 or not np.isfinite(variance) or variance <= 0:
                    rate_share = np.nan
                    basis_share = np.nan
                    sd = np.nan
                else:
                    rate_share = i.cov(p) / variance
                    basis_share = b.cov(p) / variance
                    sd = np.sqrt(variance)

                rows.append({
                    "changes": freq,
                    "period": period,
                    "tenor": tenor,
                    "share_rates": rate_share,
                    "share_basis": basis_share,
                    "sd_premium_change_bp": sd,
                })

    return pd.DataFrame(rows)


# half-lives
def half_lives(params):
    rows = []

    for factor, name in [("level", "phi_L"),
                         ("slope", "phi_S"),
                         ("curvature", "phi_C")]:
        phi = float(params[name])
        days = np.log(0.5) / np.log(phi) if 0 < phi < 1 else np.nan

        rows.append({
            "factor": factor,
            "phi_daily": phi,
            "half_life_trading_days": days,
            "at_upper_limit": phi > 0.998,
        })

    return pd.DataFrame(rows)


# monthly drivers
def prepare_drivers(monthly):
    monthly = monthly.copy()
    monthly.index = pd.to_datetime(monthly.index)
    monthly = monthly.sort_index()

    if "n_days" in monthly.columns:
        monthly = monthly[monthly["n_days"] >= 10]

    required = [
        "net_purchase_usd_mn",
        "chg_net_forward_usd_mn",
        "net_liq_inj_lakh_cr",
        "broad_usd_index",
    ]

    missing = [col for col in required if col not in monthly.columns]
    if missing:
        raise ValueError(f"Missing macro columns: {missing}")

    if (monthly["broad_usd_index"].dropna() <= 0).any():
        raise ValueError("Broad USD index must be positive")

    monthly.index = monthly.index.to_period("M").to_timestamp()
    if monthly.index.has_duplicates:
        raise ValueError("Multiple rows for the same month")

    drivers = pd.DataFrame(index=monthly.index)
    drivers["rbi_spot"] = monthly["net_purchase_usd_mn"] / 1000
    drivers["d_fwd"] = monthly["chg_net_forward_usd_mn"] / 1000
    drivers["d_liq"] = monthly["net_liq_inj_lakh_cr"].diff()
    drivers["d_usd"] = np.log(monthly["broad_usd_index"]).diff() * 100

    return drivers


# lag order
def _choose_lags(data, min_lags=2, max_lags=6):
    chosen = VAR(data).select_order(maxlags=max_lags).aic
    if chosen is None:
        raise ValueError("VAR could not select a lag order")
    return max(int(chosen), min_lags)


# one LP
def _lp_one(data, driver, horizon, lags):
    lagged = pd.concat({
        f"{col}_lag{j}": data[col].shift(j)
        for col in data.columns
        for j in range(1, lags + 1)
    }, axis=1)

    months = pd.get_dummies(
        data.index.month, prefix="month", drop_first=True
    ).astype(float)
    months.index = data.index

    X = sm.add_constant(pd.concat([data[[driver]], lagged, months], axis=1))
    df = pd.concat([data["basis"].shift(-horizon).rename("y"), X], axis=1)
    df = df.dropna()

    if len(df) <= X.shape[1]:
        raise ValueError("Too few observations for the chosen horizon and lags")

    regression = sm.OLS(df["y"], df.drop(columns="y"))
    hc1 = regression.fit(cov_type="HC1")
    hac = regression.fit(
        cov_type="HAC", cov_kwds={"maxlags": horizon + 1}
    )

    return float(hc1.params[driver]), float(hc1.bse[driver]), \
        float(hac.bse[driver]), int(hc1.nobs)


# local projections
def projections(smoothed, drivers, factors=("level", "slope"),
                focal_drivers=("rbi_spot", "d_liq"),
                horizons=range(7), lags=None,
                min_lags=2, max_lags=6, se_type="HC1"):
    if se_type not in {"HC1", "HAC"}:
        raise ValueError("se_type must be 'HC1' or 'HAC'")

    missing = [name for name in DRIVERS if name not in drivers.columns]
    if missing:
        raise ValueError(f"Missing predictor columns: {missing}")

    index = pd.to_datetime(drivers.index).to_period("M").to_timestamp()
    drivers = drivers.copy()
    drivers.index = index
    drivers = drivers.sort_index()

    if drivers.index.has_duplicates:
        raise ValueError("Multiple rows for the same month")

    rows = []

    for factor in factors:
        monthly_factor = smoothed[factor].resample("MS").mean()
        df = pd.concat([
            monthly_factor.rename("basis"), drivers[DRIVERS]
        ], axis=1).sort_index()


        df = df.asfreq("MS")
        complete = df.notna().all(axis=1)
        if not complete.any():
            raise ValueError(f"No overlapping observations for {factor}")

        first, last = df.index[complete][[0, -1]]
        df = df.loc[first:last]
        if df.isna().any().any():
            raise ValueError(
                f"Missing months inside the {factor} estimation sample"
            )

        p = _choose_lags(df, min_lags, max_lags) if lags is None else lags
        if not isinstance(p, int) or p < 1:
            raise ValueError("lags must be a positive integer")

        for driver in focal_drivers:
            if driver not in DRIVERS:
                raise ValueError(f"Unknown focal driver: {driver}")

            for horizon in horizons:
                if not isinstance(horizon, int) or horizon < 0:
                    raise ValueError("Horizons must be nonnegative integers")

                beta, se_hc1, se_hac, n = _lp_one(
                    df, driver, horizon, p
                )
                se = se_hc1 if se_type == "HC1" else se_hac

                rows.append({
                    "factor": factor,
                    "driver": driver,
                    "h_months": horizon,
                    "response_bp": beta,
                    "se_HC1": se_hc1,
                    "se_HAC": se_hac,
                    "lower_90": beta - 1.645 * se,
                    "upper_90": beta + 1.645 * se,
                    "N": n,
                    "lags": p,
                })

    return pd.DataFrame(rows)


# save
def save_results(folder, allocation, persistence, local_projections=None):
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {folder}")
    folder.mkdir(parents=True, exist_ok=True)

    allocation.to_csv(folder / "covariance_allocation.csv", index=False)
    persistence.to_csv(folder / "factor_half_lives.csv", index=False)

    if local_projections is not None:
        local_projections.to_csv(
            folder / "factor_local_projections.csv", index=False
        )

    return folder
