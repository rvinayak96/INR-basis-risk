# tables and figures for the report
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

import curve_model as cm
import curve_estimate as ce
import risk_backtest as rb
import stress_turn as st


NAMES = {"HS": "HS", "EWMA-normal": "RiskMetrics", "EWMA-t-cal": "EWMA-t",
         "SS-normal": "SS-normal", "SS-t-EWMA": "SS-t", "SS-FHS": "SS-FHS",
         "RW-normal": "RW-normal"}
SHOW = ["1M", "3M", "12M"]


# show and save a table
def keep(table, name, folder):
    print(name)
    display(table)
    if folder is not None:
        table.to_csv(folder / f"table_{name}.csv")


# show and save a figure
def save(fig, name, folder):
    fig.tight_layout()
    if folder is not None:
        fig.savefig(folder / f"{name}.png", dpi=150)
    plt.show()


# tables
def tables(S, folder, data, train, params, names, summary, out, persistence,
           allocation, lp=None, risk=None, stress=None, sim=None, turn_test=None):
    T = S["tenors"]
    basis = data["basis"]

    keep(pd.DataFrame({"Mean": basis.mean(), "SD": basis.std(), "Min": basis.min(),
                       "Max": basis.max(), "SD daily change": basis.diff().std()}
                      ).round(1), "basis_summary", folder)
    keep(summary, "fit_summary", folder)

    # estimates and standard errors
    model = cm.BasisCurveModel(train, S["lambda"], S["use_fbd"])
    model._start = params
    se = model.smooth(params, cov_type="approx").bse
    keep(pd.DataFrame({"Estimate": params, "Std. error": se}, index=names).round(4),
         "parameters", folder)

    keep(out["fit_by_tenor"].round(2), "fit_by_tenor", folder)
    keep(persistence.round(3), "half_lives", folder)
    keep(out["turn_by_year"].round(2), "turn_by_year", folder)
    keep(out["turn_validation"].round(3), "turn_validation", folder)
    keep((100 * allocation.pivot_table(index="tenor", columns=["changes", "period"],
                                       values="share_basis").loc[T]).round(1),
         "basis_share_pct", folder)

    if lp is not None:
        keep(lp.round(2), "local_projections", folder)

    if risk is not None:
        bt = risk["backtests"].replace({"model": NAMES})
        models = [NAMES[m] for m in S["risk_models"]]
        for level in S["risk_levels"]:
            part = bt[bt["level"] == level]
            for value in ["breaches", "zone", "es_test_p"]:
                table = part.pivot(index="tenor", columns="model", values=value)
                keep(table.loc[T, models], f"{value}_{level}", folder)
        keep(risk["loss_comparison"].replace({"model": NAMES, "benchmark": NAMES}).round(3),
             "loss_comparison", folder)

    if stress is not None:
        keep(stress["turn_now"].round(2), "turn_now", folder)
        keep(stress["event_summary"].round(1), "rollover_event", folder)
        for value in ["change_bp", "inr_lakh_for_notional"]:
            table = stress["stress_scenarios"].pivot(index="tenor", columns="episode",
                                                     values=value)
            keep(table.loc[T].round(1), f"stress_{value}", folder)

    if sim is not None:
        p = sim["params"]
        keep(pd.DataFrame({"True": params, "Mean": p[names].mean(), "SD": p[names].std(),
                           "Min": p[names].min(), "Max": p[names].max()},
                          index=names).round(4), "mc_parameters", folder)
        keep(pd.DataFrame({"Replications": [len(p)],
                           "Converged": [int(p["converged"].sum())],
                           "Any parameter at bound": [int(p["at_bound"].notna().sum())]}),
             "mc_convergence", folder)
        keep(sim["states"].groupby(["state", "estimate"])[
            ["rmse", "bias", "coverage_90"]].mean().round(3), "mc_states", folder)

    if turn_test is not None:
        keep(turn_test[["sample", "LR", "raw_LR"]].round(3), "turn_lr_test", folder)


# figures
def figures(S, folder, data, out, allocation, lp=None, risk=None, stress=None):
    T = S["tenors"]
    basis = data["basis"]

    fig, ax = plt.subplots(figsize=(10, 4))
    basis[SHOW].plot(ax=ax, lw=0.7)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title("CIP basis (bp)")
    save(fig, "fig_basis", folder)

    fig, ax = plt.subplots(figsize=(8, 4))
    daily = allocation[allocation["changes"] == "daily"]
    share = daily.pivot(index="tenor", columns="period", values="share_basis").loc[T]
    (100 * share).plot(ax=ax, marker="o", ms=3)
    ax.set_ylabel("%")
    ax.set_title("Basis share of daily forward premium variance")
    save(fig, "fig_basis_share", folder)

    if folder is not None:
        ce.plot_states(out["smoothed"], out["observed_turn"], folder / "fig_states.png")
        st.plot_turn(out["filtered"], out["observed_turn"], folder / "fig_turn.png")

    if lp is not None:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
        for i, factor in enumerate(["level", "slope"]):
            for j, driver in enumerate(["rbi_spot", "d_liq"]):
                part = lp[(lp["factor"] == factor) & (lp["driver"] == driver)]
                ax = axes[i, j]
                ax.plot(part["h_months"], part["response_bp"], marker="o", ms=3)
                ax.fill_between(part["h_months"], part["lower_90"], part["upper_90"],
                                alpha=0.2)
                ax.axhline(0, color="grey", lw=0.5)
                ax.set_title(f"{factor}: {driver}")
        save(fig, "fig_local_projections", folder)

    if risk is not None:
        bt = risk["backtests"].replace({"model": NAMES})
        part = bt[bt["level"] == 0.99]
        models = [NAMES[m] for m in S["risk_models"]]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        part.pivot(index="tenor", columns="model", values="breaches").loc[T, models].plot(
            ax=ax, marker="o", ms=3)
        ax.axhline(part["expected"].iloc[0], color="black", ls="--", lw=0.8)
        ax.axhline(10.5, color="goldenrod", lw=0.6)
        ax.axhline(17.5, color="firebrick", lw=0.6)
        ax.set_title("99% VaR breaches, test period")
        save(fig, "fig_breaches_99", folder)

        if folder is not None:
            rb.plot_var({t: risk["forecasts"][t] for t in SHOW}, risk["settings"]["nus"],
                        folder / "fig_var.png", test_start=S["test_start"],
                        window=S["risk_window"], models=S["risk_models"])

    if stress is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        event = stress["event_summary"]
        event[["median_1M", "median_3M", "median_12M"]].plot(ax=ax, marker="o", ms=3)
        ax.axvline(0, color="grey", ls="--", lw=0.5)
        ax.set_title("Median basis change around the year-end rollover (bp)")
        save(fig, "fig_rollover", folder)

        fig, ax = plt.subplots(figsize=(8, 4))
        scen = stress["stress_scenarios"].pivot(index="tenor", columns="episode",
                                                values="change_bp")
        scen.loc[T].plot(ax=ax, marker="o", ms=3)
        ax.set_title("Historical stress scenarios (bp)")
        save(fig, "fig_stress", folder)
