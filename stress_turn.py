# turn charge, event study, stress

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import curve_estimate as ce
import curve_model as cm


EPISODES = {
    "Covid, Mar-Apr 2020": ("2020-03-01", "2020-04-30"),
    "Dec 2025 liquidity squeeze": ("2025-12-01", "2025-12-31"),
    "Mar-Apr 2026 position cap and NDF curbs": ("2026-03-16", "2026-04-30")
}
HORIZON = 10
NOTIONAL_USD = 10_000_000


# bp to paise
def paise(bp, spot):
    return bp / 10_000 * spot * 100


# turn by year
def turn_by_year(filtered, observed, spot, rollover_days):
    rows = []
    for rollover in rollover_days:
        rollover = pd.Timestamp(rollover)
        if rollover not in filtered.index:
            continue
        pos = filtered.index.get_loc(rollover)
        if pos == 0:
            continue

        last = filtered.index[pos - 1]
        feb = filtered.loc[
            (filtered.index.year == rollover.year)
            & (filtered.index.month == 2)
        ]
        if feb.empty:
            continue

        feb_date = feb.index[-1]
        recent = observed.loc[
            (observed.index.year == rollover.year)
            & (observed.index <= last)
        ]
        rows.append({
            "turn_year": rollover.year,
            "rollover_date": rollover,
            "last_full_turn_date": last,
            "end_feb_bp": filtered.loc[feb_date, "turn"],
            "end_feb_lo": filtered.loc[feb_date, "turn_lo"],
            "end_feb_hi": filtered.loc[feb_date, "turn_hi"],
            "end_feb_paise": paise(filtered.loc[feb_date, "turn"], spot.loc[feb_date]),
            "last_full_turn_bp": filtered.loc[last, "turn"],
            "last_full_turn_lo": filtered.loc[last, "turn_lo"],
            "last_full_turn_hi": filtered.loc[last, "turn_hi"],
            "last_full_turn_paise": paise(filtered.loc[last, "turn"], spot.loc[last]),
            "observed_median_bp": recent.median(),
            "observed_days": int(recent.notna().sum())
        })
    return pd.DataFrame(rows)


# turn now
def turn_now(filtered, spot, turn_year):
    date = filtered.index[-1]
    f = filtered.loc[date]
    year = int(turn_year.loc[date])
    return pd.DataFrame([{
        "as_of": date,
        "turn": f"31 March {year}",
        "estimate_bp": f["turn"],
        "lo90_bp": f["turn_lo"],
        "hi90_bp": f["turn_hi"],
        "estimate_paise": paise(f["turn"], spot.loc[date]),
        "lo90_paise": paise(f["turn_lo"], spot.loc[date]),
        "hi90_paise": paise(f["turn_hi"], spot.loc[date])
    }])


# event study
def event_study(data, rollover_days, before=8, after=3,
                tenors=("1M", "3M", "12M")):
    basis = data["basis"]
    index = basis.index
    rows = []
    for date in rollover_days:
        date = pd.Timestamp(date)
        if date not in index:
            continue
        i = index.get_loc(date)
        if i - before - 1 < 0 or i + after >= len(index):
            continue
        for k in range(-before, after + 1):
            change = basis[list(tenors)].iloc[i + k] - basis[list(tenors)].iloc[i + k - 1]
            rows.append({
                "year": date.year, "day": k,
                **change.to_dict()
            })
    events = pd.DataFrame(rows)
    if events.empty:
        raise ValueError("No complete rollover event windows")
    grouped = events.groupby("day")[list(tenors)]
    summary = pd.concat({
        "mean": grouped.mean(),
        "median": grouped.median(),
        "years_with_fall": grouped.agg(lambda s: int((s < 0).sum()))
    }, axis=1)
    summary.columns = [f"{stat}_{tenor}" for stat, tenor in summary.columns]
    return summary, events


# stress replay
def stress(smoothed, data, episodes=EPISODES, horizon=HORIZON,
           lam=cm.LAM, notional_usd=NOTIONAL_USD):
    idx = data["basis"].index
    last = idx[-1]
    tenors = data["tenors"]
    n_now = data["n"].loc[last].to_numpy()
    tau_now = data["tau"].loc[last].to_numpy()
    spot_now = float(data["spot"].loc[last])
    if not np.isfinite(spot_now):
        raise ValueError("Missing latest spot rate for INR stress conversion")

    g_now, h_now = cm.ns_params(tau_now, lam)
    g3, h3 = cm.ns_params(3.0, lam)
    factors = smoothed[["level", "slope", "curvature"]]
    curve_3m = (
        factors["level"]
        + factors["slope"] * g3
        + factors["curvature"] * h3
    )
    rows, raw = [], []

    for name, (first, final) in episodes.items():
        period = curve_3m.loc[first:final]
        if len(period) <= horizon:
            raise ValueError(f"Episode too short for {horizon}-day horizon: {name}")
        moves = period.shift(-horizon) - period
        start = moves.dropna().idxmax()
        end = period.index[period.index.get_loc(start) + horizon]

        difference = factors.loc[end] - factors.loc[start]
        shock = (
            difference["level"]
            + difference["slope"] * g_now
            + difference["curvature"] * h_now
        )
        for i, tenor in enumerate(tenors):
            rupees_per_usd = shock[i] / 10_000 * n_now[i] / 365 * spot_now
            rows.append({
                "episode": name,
                "window": f"{start.date()} to {end.date()}",
                "tenor": tenor,
                "change_bp": shock[i],
                "paise_per_usd": rupees_per_usd * 100,
                "inr_lakh_for_notional": rupees_per_usd * notional_usd / 100_000
            })

        obs = data["basis"].loc[first:final]
        raw_change = (obs.shift(-horizon) - obs).max()
        raw.append({"episode": name, **raw_change.to_dict()})

    return pd.DataFrame(rows), pd.DataFrame(raw), last


# turn plot
def plot_turn(filtered, observed, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(filtered.index, filtered["turn"], lw=0.9, label="Filtered turn")
    ax.fill_between(filtered.index, filtered["turn_lo"], filtered["turn_hi"], alpha=0.2)
    ax.plot(observed.index, observed, "o", markersize=2.5, label="FBD proxy")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("Basis points of notional")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# run everything
def run(data, filtered, smoothed, observed=None, rollover_days=None,
        episodes=EPISODES, horizon=HORIZON, lam=cm.LAM,
        notional_usd=NOTIONAL_USD, event_tenors=("1M", "3M", "12M")):
    if observed is None:
        observed = ce.observed_turn_price(data)
    if rollover_days is None:
        _, rollover_days = ce.turn_days(data)

    years = turn_by_year(filtered, observed, data["spot"], rollover_days)
    now = turn_now(filtered, data["spot"], data["turn_year"])
    timing, events = event_study(data, rollover_days, tenors=event_tenors)
    scenarios, raw, last = stress(
        smoothed, data, episodes, horizon, lam, notional_usd
    )
    return {
        "turn_by_year": years, "turn_now": now,
        "event_summary": timing, "event_by_year": events,
        "stress_scenarios": scenarios, "stress_raw": raw,
        "stress_date": last, "horizon": horizon,
        "filtered": filtered, "observed_turn": observed
    }


# save
def save_results(result, folder):
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    for key, name in {
        "turn_by_year": "turn_by_year.csv",
        "turn_now": "turn_now.csv",
        "event_summary": "turn_timing_event_study.csv",
        "event_by_year": "turn_timing_by_year.csv",
        "stress_scenarios": "stress_scenarios.csv",
        "stress_raw": f"stress_raw_{result['horizon']}day_rises.csv"
    }.items():
        result[key].to_csv(folder / name, index=(key == "event_summary"))
    plot_turn(
        result["filtered"], result["observed_turn"],
        folder / "fig_turn_realtime.png"
    )
    return folder
