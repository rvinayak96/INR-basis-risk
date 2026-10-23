# building the daily basis curve

import numpy as np
import pandas as pd


# rate for each contract
def interpolate(curve, dates, n):

    rates = np.full(len(n), np.nan)

    for i, (date, days) in enumerate(zip(dates, n)):

        if pd.isna(date) or pd.isna(days):
            continue

        if date not in curve.index:
            continue

        row = curve.loc[date].dropna()

        if row.empty:
            continue

        rates[i] = np.interp(
            days,
            row.index.to_numpy(dtype=float),
            row.to_numpy(dtype=float)
        )

    return rates

# CIP premium
def implied_bp(i_in, i_us, n):

    inr_growth = 1 + i_in / 100 * n / 365
    usd_growth = 1 + i_us / 100 * n / 365

    forward_ratio = inr_growth / usd_growth

    return (forward_ratio - 1) * 365 / n * 10000


# clean forwards, contract days
def prepare_forwards(forwards):

    required = [
        "date", "tenor", "settle",
        "spot_settle", "prem_pct"
    ]

    missing = [
        col for col in required
        if col not in forwards.columns
    ]

    if missing:
        raise ValueError(f"Missing forward columns: {missing}")

    fwd = forwards.copy()

    for col in ["date", "settle", "spot_settle"]:
        fwd[col] = pd.to_datetime(fwd[col])

    fwd["tenor"] = fwd["tenor"].str.strip()

    fwd["prem_pct"] = pd.to_numeric(
        fwd["prem_pct"], errors="raise"
    )

    if fwd[required].isna().any().any():
        raise ValueError("Missing required forward data")

    if fwd.duplicated(["date", "tenor"]).any():
        raise ValueError("Duplicate date and tenor")

    fwd["n"] = (
        fwd["settle"] - fwd["spot_settle"]
    ).dt.days

    if (fwd["n"] <= 0).any():
        raise ValueError("Forward maturity must be positive")

    return fwd


# FBD April dates
def turn_calendar(forwards, years, fallback_april1=False):

    fbd = forwards.loc[
        forwards["tenor"] == "FBD April", "settle"
    ]

    fbd = fbd.groupby(fbd.dt.year).agg(
        lambda dates: dates.mode().iloc[0]
    )

    rows = []

    for year in years:
        from_fbil = year in fbd.index

        if not from_fbil and not fallback_april1:
            raise ValueError(
                f"No FBIL FBD April date for {year}"
            )

        date = (
            fbd.loc[year] if from_fbil
            else pd.Timestamp(f"{year}-04-01")
        )

        rows.append({
            "year": year,
            "fbd_april": date,
            "from_fbil": from_fbil
        })

    return pd.DataFrame(rows).set_index("year")


# year-end exposure
def add_turn_exposure(forwards, calendar):

    fwd = forwards.copy()

    year = fwd["spot_settle"].dt.year

    this_year_fbd = year.map(calendar["fbd_april"])

    if this_year_fbd.isna().any():
        raise ValueError("Missing FBD date for spot-settlement year")

    year = year.where(
        fwd["spot_settle"] < this_year_fbd,
        year + 1
    )

    fwd["turn_year"] = year

    fwd["fbd_april"] = year.map(calendar["fbd_april"])

    fwd["fbd_from_fbil"] = year.map(
        calendar["from_fbil"]
    )

    if fwd["fbd_april"].isna().any():
        raise ValueError("Missing next FBD April date")

    # crosses the year end?
    fwd["x_fy"] = np.where(
        fwd["settle"] >= fwd["fbd_april"],
        365 / fwd["n"],
        0.0
    )

    return fwd


# previous US date
def match_us_dates(forwards, usd_curve):

    dates = pd.DataFrame({
        "date": np.sort(forwards["date"].unique())
    })

    valid = usd_curve.dropna(how="all").index

    us_dates = pd.DataFrame({
        "date": pd.DatetimeIndex(valid).sort_values()
    })

    us_dates["us_date"] = us_dates["date"]

    link = pd.merge_asof(
        dates,
        us_dates,
        on="date",
        direction="backward",
        allow_exact_matches=False
    )

    out = forwards.merge(
        link,
        on="date",
        how="left",
        validate="many_to_one"
    )

    out["us_lag_days"] = (
        out["date"] - out["us_date"]
    ).dt.days

    return out


# quick checks
def checks(curve):

    summary = curve.assign(
        missing_basis=curve["basis_bp"].isna(),
        fallback_turn=~curve["fbd_from_fbil"].astype(bool)
    )

    summary = summary.groupby("tenor").agg(
        observations=("basis_bp", "size"),
        missing_basis=("missing_basis", "sum"),
        mean_basis_bp=("basis_bp", "mean"),
        sd_basis_bp=("basis_bp", "std"),
        max_us_lag_days=("us_lag_days", "max"),
        fallback_turn_rows=("fallback_turn", "sum")
    )

    return summary.reset_index()


# basis curve
def build(forwards, inr_curve, usd_curve, ois_curve=None, tbill_curve=None,
          start="2018-07-30", end=None, tenors=None, include_fbd=True,
          calendar=None, fallback_april1=False):
    if tenors is None:
        tenors = [f"{m}M" for m in range(1, 13)]

    selected = list(tenors)
    source = forwards[forwards["tenor"].isin(selected + ["FBD April"])].copy()

    fwd = source.copy()
    if start is not None:
        fwd = fwd[fwd["date"] >= pd.Timestamp(start)]
    if end is not None:
        fwd = fwd[fwd["date"] <= pd.Timestamp(end)]
    fwd = prepare_forwards(fwd)
    if fwd.empty:
        raise ValueError("No forwards in the selected sample")

    if calendar is None:
        first = int(fwd["spot_settle"].dt.year.min())
        last = int(fwd["spot_settle"].dt.year.max()) + 1

        history = source.copy()
        history["settle"] = pd.to_datetime(history["settle"])
        calendar = turn_calendar(history, range(first, last + 1), fallback_april1=True)

    fwd = add_turn_exposure(fwd, calendar)
    if not fallback_april1 and (~fwd["fbd_from_fbil"].astype(bool)).any():
        years = sorted(fwd.loc[~fwd["fbd_from_fbil"].astype(bool), "turn_year"].unique())
        raise ValueError(f"Missing published FBD April dates for turn years: {years}")

    final_tenors = selected + (["FBD April"] if include_fbd else [])
    fwd = fwd[fwd["tenor"].isin(final_tenors)].copy()
    if fwd.empty:
        raise ValueError("No selected forward observations")

    # previous US day, no look-ahead
    fwd = match_us_dates(fwd, usd_curve)
    n = fwd["n"].to_numpy()
    fwd["i_us"] = interpolate(usd_curve, fwd["us_date"], n)
    fwd["i_in"] = interpolate(inr_curve, fwd["date"], n)
    fwd["i_in_ois"] = (interpolate(ois_curve, fwd["date"], n)
                        if ois_curve is not None else np.nan)
    fwd["i_in_tbill"] = (interpolate(tbill_curve, fwd["date"], n)
                          if tbill_curve is not None else np.nan)
    fwd["prem_bp"] = fwd["prem_pct"] * 100
    fwd["implied_bp"] = implied_bp(fwd["i_in"], fwd["i_us"], n)
    fwd["basis_bp"] = fwd["prem_bp"] - fwd["implied_bp"]
    fwd["basis_ois_bp"] = fwd["prem_bp"] - implied_bp(fwd["i_in_ois"], fwd["i_us"], n)
    fwd["basis_tbill_bp"] = fwd["prem_bp"] - implied_bp(fwd["i_in_tbill"], fwd["i_us"], n)
    fwd["basis_notional_bp"] = fwd["basis_bp"] * n / 365
    fwd["tau_months"] = n / (365 / 12)

    for col in ["spot", "prem_rs"]:
        if col not in fwd:
            fwd[col] = np.nan

    cols = [
        "date", "tenor", "settle", "spot_settle", "n", "tau_months",
        "spot", "prem_rs", "prem_bp", "i_in", "i_in_ois", "i_in_tbill",
        "i_us", "us_date", "implied_bp", "basis_bp", "basis_ois_bp",
        "basis_tbill_bp", "basis_notional_bp", "x_fy", "turn_year",
        "fbd_april", "fbd_from_fbil", "us_lag_days"
    ]
    curve = fwd[cols].sort_values(["date", "n"]).reset_index(drop=True)
    return curve, checks(curve)


# rate cross-checks
def rate_checks(curve, usd_curve=None, us_bills=None):
    rows = []
    for tenor in ["1M", "3M", "6M", "12M"]:
        c = curve[curve["tenor"] == tenor]
        d = c["i_in_ois"] - c["i_in_tbill"]
        rows.append({"check": f"INR OIS - T-bill {tenor}, pct pts",
                     "n": int(d.notna().sum()), "mean": d.mean(), "sd": d.std()})

    if usd_curve is not None and us_bills is not None:
        for cmt, bill in [(30, 28), (91, 91), (182, 182), (365, 364)]:
            diff = (us_bills[bill] - usd_curve[cmt]).dropna()
            rows.append({"check": f"US bill {bill}d - CMT {cmt}d, pct pts",
                         "n": len(diff), "mean": diff.mean(), "sd": diff.std()})
    return pd.DataFrame(rows)


# US bill yields
def load_us_bills(files):
    frames = []
    for file in files:
        df = pd.read_csv(file)
        df.index = pd.to_datetime(df["Date"], format="mixed")
        out = pd.DataFrame(index=df.index)
        for days, label in [(28, "4 WEEKS COUPON EQUIVALENT"),
                            (91, "13 WEEKS COUPON EQUIVALENT"),
                            (182, "26 WEEKS COUPON EQUIVALENT")]:
            out[days] = pd.to_numeric(df[label], errors="coerce")
        i = pd.to_numeric(df["52 WEEKS COUPON EQUIVALENT"], errors="coerce") / 100
        out[364] = (i * 364 / 365 + i ** 2 * (364 / 730 - 0.25)) * 365 / 364 * 100
        frames.append(out)
    return pd.concat(frames).sort_index()
