# loading raw rate and forward files

import pandas as pd

# tidy a rate table
def standardize_curve(curve):


    curve = curve.copy()

    curve.index = pd.to_datetime(curve.index)
    curve.columns=curve.columns.astype(float)


    if curve.index.has_duplicates:
        raise ValueError("Duplicate dates in rate curve")

    if curve.columns.has_duplicates:
        raise ValueError("duplicate maturities in rate curve")

    if any(day<=0 for day in curve.columns):
        raise ValueError("maturities must be positive")

    curve = curve.apply(pd.to_numeric, errors="raise")
    curve.index.name = "date"
    return curve.sort_index().sort_index(axis=1)


# curve from one file per maturity
def load_rate_files(files):


    if not files:
        raise ValueError("no rate files specified")

    series = {}

    for days, file in files.items():

        df =pd.read_csv(file)

        df["date"]=pd.to_datetime(
            df["date"], format="%Y-%m-%d"

        )

        df["rate"]=pd.to_numeric(
            df["rate"],errors="raise"
        )

        if df["date"].duplicated().any():
            raise ValueError(f"Duplicate dates in {file}")

        series[days]=df.set_index("date")["rate"]

    curve = pd.DataFrame(series)

    return standardize_curve(curve)


# FBIL OIS / T-bill files
def load_long_curve(file, tenor_days, tenor_col="Tenor", skiprows=0):

    df = pd.read_csv(file, skiprows=skiprows, dtype=str)

    df.columns = df.columns.str.strip()

    df[tenor_col] = df[tenor_col].str.strip()

    df = df[df[tenor_col].isin(tenor_days)].copy()

    if df.empty:
        raise ValueError("No matching tenors found")

    df["date"] = pd.to_datetime(
        df["Date"].str.strip(),
        format="mixed",
        dayfirst=True
    )

    df["days"] = df[tenor_col].map(tenor_days)

    df["rate"] = pd.to_numeric(
        df["Rate(%)"], errors="coerce"
    )

    df = df.dropna(subset=["rate"])

    curve = df.pivot_table(
        index="date", columns="days", values="rate", aggfunc="last"
    )

    return standardize_curve(curve)


# own curve file
def load_wide_curve(file):

    df = pd.read_csv(file)

    df["date"] = pd.to_datetime(df["date"])

    return standardize_curve(df.set_index("date"))


# FBIL forward premia
def load_fbil_forwards(file, skiprows=2):

    df = pd.read_csv(file, skiprows=skiprows, dtype=str)
    df.columns = df.columns.str.strip()

    required = [
        "Date", "Tenor", "Settlement Date",
        "USD/INR Premia(%)", "USD/INR Premia(Rs.)",
        "Comments"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing FBIL columns: {missing}")

    df = df.rename(columns={
        "Date": "date",
        "Tenor": "tenor",
        "Settlement Date": "settle",
        "USD/INR Premia(%)": "prem_pct",
        "USD/INR Premia(Rs.)": "prem_rs",
        "Comments": "comment"
    })

    df["date"] = pd.to_datetime(
        df["date"].str.strip(), format="%d-%b-%y"
    )

    df["settle"] = pd.to_datetime(
        df["settle"].str.strip(),
        format="%d-%b-%y",
        errors="coerce"
    )

    df["tenor"] = df["tenor"].str.strip()

    for col in ["prem_pct", "prem_rs"]:
        df[col] = pd.to_numeric(
            df[col].str.replace(",", "", regex=False),
            errors="coerce"
        )

    # drop repeated (stale) fixings
    repeated = df.loc[
        df["comment"].fillna("").str.contains(
            "Repeated", regex=False
        ),
        "date"
    ].unique()

    df = df[~df["date"].isin(repeated)].copy()

    spot = df[df["tenor"] == "Spot"].set_index("date")

    if spot.index.has_duplicates:
        raise ValueError("Duplicate Spot fixing dates")

    fwd = df[df["tenor"] != "Spot"].dropna(
        subset=["prem_pct", "settle"]
    ).copy()

    fwd["spot_settle"] = fwd["date"].map(spot["settle"])
    fwd["spot"] = fwd["date"].map(spot["prem_rs"])

    cols = [
        "date", "tenor", "settle", "spot_settle",
        "prem_pct", "prem_rs", "spot"
    ]

    return fwd[cols]


# FRED series
def read_fred(file):
    df = pd.read_csv(file)

    if len(df.columns) != 2:
        raise ValueError("Expected two columns in FRED file")

    df.columns = ["date", "rate"]

    df["date"] = pd.to_datetime(df["date"])

    df["rate"] = pd.to_numeric(
        df["rate"], errors="coerce"
    )

    return df.set_index("date")["rate"]


# 1Y par yield to zero rate
def zero_1y(y6, y1):

    p6 = 1 / (1 + y6 / 100 * 182.5 / 365)

    p1 = (1 - y1 / 200 * p6) / (1 + y1 / 200)

    return (1 / p1 - 1) * 100


# US Treasury curve
def load_us_treasury(
    par_file,
    fred_files,
    switch_date="2024-01-01",
    par_start="2017-12-01"
):

    par = pd.read_csv(par_file)

    par["date"] = pd.to_datetime(
        par["date"], format="%m/%d/%Y"
    )

    par = par.set_index("date")[
        ["1 mo", "3 mo", "6 mo", "1 yr"]
    ]

    par.columns = ["1M", "3M", "6M", "12M"]

    switch = pd.Timestamp(switch_date)

    par = par.loc[
        (par.index >= pd.Timestamp(par_start))
        & (par.index < switch)
    ]

    fred = pd.DataFrame({
        tenor: read_fred(file)
        for tenor, file in fred_files.items()
    })

    fred = fred.loc[fred.index >= switch]

    cmt = pd.concat([par, fred])
    cmt = cmt.sort_index().dropna(how="all")

    curve = pd.DataFrame({
        30: cmt["1M"],
        91: cmt["3M"],
        182: cmt["6M"],
        365: zero_1y(cmt["6M"], cmt["12M"])
    })

    return standardize_curve(curve)
