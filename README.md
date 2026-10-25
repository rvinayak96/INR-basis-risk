# USD/INR basis risk

Code for building a daily USD/INR forward basis curve (1M–12M), modelling it with a dynamic Nelson–Siegel state-space model that includes a 31 March year-end factor, and backtesting VaR/ES models for basis risk. The research note is available on request.

## Setup

```
pip install -r requirements.txt
jupyter notebook USDINR_Basis_risk.ipynb
```

Tested with Python 3.12, numpy 2.3.5, pandas 2.2.3, scipy 1.17.0, statsmodels 0.15.0, matplotlib 3.11.1.

## Folder layout

```
project/
├── USDINR_Basis_risk.ipynb   runs everything
├── *.py                          code (below)
├── raw/                          input files (not included)
└── results/                      one new folder per run (created automatically)
```

Keep the notebook, the `.py` files and `raw/` in the same folder.

## How to run

1. Put the data files in `raw/`.
2. Open the notebook. In the **Settings** cell, set the file names in `FILES` and the options in `S`.
3. Run all cells. Each run is saved to `results/basis_run_<date>_<time>/` with its settings, outputs, a copy of the code, package versions and file hashes. Nothing is overwritten.
4. The last cell calls `report.py`, which shows the main tables and figures and saves them in the same run folder.

### Main settings (`S`)

|Setting|What it does|
|-|-|
|`data_mode`|`"raw"` builds the basis from the raw files; `"csv"` loads a saved `basis_curve_long.csv`|
|`param_mode`|`"fit"` estimates the model (about 25 minutes); `"saved"` loads `params.csv`|
|`start`, `end`, `train_end`, `test_start`|sample, estimation and test periods|
|`tenors`|forward tenors used (at least four, labelled like `"3M"`)|
|`lambda`|Nelson–Siegel decay (per month)|
|`inr_rate`|`"ois"` or `"tbill"` for the rupee rate|
|`us_switch`|date from which FRED series replace the Treasury par yield file|
|`turn_walk`, `use_fbd`|year-end charge as a random walk; use the FBD April contract|
|`two_routes`, `maxiter`|optimiser settings|
|`comparisons`|also fit the comparison models (needed for the turn validation)|
|`risk`, `stress`, `local_projections`, `simulation`, `null_simulation`, `turn_test`|switch each analysis on or off|
|`risk_tenors`, `risk_models`, `risk_levels`, `risk_window`|risk backtest options|
|`save`|save the run to `results/`|

### Typical workflow

* **First run:** `data_mode="raw"`, `param_mode="fit"`, `comparisons=True`.
* Then copy `params.csv` and `basis_curve_long.csv` from that run folder into `raw/`.
* **Later runs:** `data_mode="csv"`, `param_mode="saved"`, with risk, stress or simulations switched on. Nothing is re-estimated.
* Simulations and the turn test are slow (several hours). Set `two_routes=False` for them.

## Data

|Key in `FILES`|Source|Format expected|
|-|-|-|
|`forwards`|FBIL USD/INR forward premia|FBIL download as is: two title rows, then `Date, Time, Tenor, Settlement Date, USD/INR Premia(%), USD/INR Premia(Rs.), Comments`; dates like `6-Oct-26`. Must include the `Spot` and `FBD April` rows|
|`ois`|FBIL MIBOR-OIS|FBIL download as is: two title rows, then `Date, Time, Period, Rate(%), Comments`|
|`tbill`|FBIL T-bill curve|FBIL download as is: two title rows, then `Date, Time, Tenor, Rate(%), Comments`|
|`us_par`|US Treasury par yield curve|columns `date` (mm/dd/yyyy), `1 mo`, `3 mo`, `6 mo`, `1 yr`, …|
|`us_1m`, `us_3m`, `us_6m`, `us_1y`|FRED DGS1MO, DGS3MO, DGS6MO, DGS1|FRED CSV as downloaded: two columns, date and value|
|(optional)|US Treasury daily bill rates|files named `us_treasury_bill_rates_*.csv`; used only for rate checks|
|`monthly`|RBI and FRED (hand-assembled)|one row per month, column `date`, and `net_purchase_usd_mn`, `chg_net_forward_usd_mn`, `net_liq_inj_lakh_cr`, `broad_usd_index`; optional `n_days`. Only needed for local projections|

### Do

* Use the files exactly as downloaded, and set their names in `FILES`.
* Make sure the sample covers every year end you include: the code needs FBIL's FBD April date for each year (or set `fbd_fallback=True` to assume 1 April).
* Use `param_mode="fit"` whenever you change `start`, `train_end`, `tenors`, `lambda`, `inr_rate` or `us_switch`. The notebook stops if saved parameters don't match these settings.

### Don't

* Don't open and re-save the CSVs in Excel. Excel changes date formats (`6-Oct-26`) and the loaders will fail or misread dates.
* Don't rename columns or delete the title rows in FBIL files.
* Don't move `raw/` away from the code; paths are set relative to the code folder.
* Don't edit files inside `results/`; copy what you need into `raw/` instead.

## Code files and functions

### `data_prep.py`: loading raw files

|Function|What it does|
|-|-|
|`standardize_curve`|Checks a rate table (dates × days to maturity): no duplicates, positive maturities, numeric rates|
|`load_rate_files`|Builds a curve from one file per maturity|
|`load_long_curve`|Reads FBIL OIS or T-bill files into a dates × days curve|
|`load_wide_curve`|Reads a curve saved in wide format|
|`load_fbil_forwards`|Reads FBIL forward premia; returns forwards with spot settlement date and spot rate|
|`read_fred`|Reads a FRED series|
|`zero_1y`|Converts the 1Y par yield to a zero rate|
|`load_us_treasury`|Combines the Treasury par file (before `us_switch`) and FRED series (after) into one US curve|

### `curve_build.py`: building the basis curve

|Function|What it does|
|-|-|
|`interpolate`|Linear interpolation of a rate curve to each contract's number of days|
|`implied_bp`|CIP-implied premium in annualised bp|
|`prepare_forwards`|Cleans forwards and computes contract days|
|`turn_calendar`|Finds the FBD April settlement date for each year|
|`add_turn_exposure`|Marks contracts that settle after the coming year end (exposure 365/n)|
|`match_us_dates`|Uses the previous US business day for US rates|
|`build`|Builds the long basis table for all tenors and the FBD April contract|
|`checks`|Basic checks by tenor (counts, missing values, US data lag)|
|`rate_checks`|OIS vs T-bill and Treasury CMT vs bill-rate comparisons|
|`load_us_bills`|Reads US daily bill-rate files|

### `curve_model.py`: the state-space model

|Function|What it does|
|-|-|
|`load_curve`|Turns the long basis table into model inputs (dates × tenors)|
|`sample`|Cuts the inputs to a date range|
|`ns_params`|Nelson–Siegel slope and curvature loadings|
|`parameter_names`|Names of the 30 parameters|
|`BasisCurveModel`|The model (statsmodels `MLEModel`): level, slope, curvature, year-end charge, AR(1) tenor deviations, FBD April observation|
|`cross_section`|Daily OLS fit of the curve (two-step Diebold–Li)|
|`start_values`|Starting values from the two-step fit|
|`held`|Parameters to hold fixed for restricted versions|
|`mle`|One maximum-likelihood run (L-BFGS)|
|`fit`|Fit with two starting routes; keeps the better one|
|`run_filter`|Kalman filter and smoother at given parameters|
|`states_table`|Filtered or smoothed states with 90% bands|
|`profile_phi`|Likelihood profile over factor persistence|
|`simulate`|Simulates data from the model|

### `curve_estimate.py`: estimation and checks

|Function|What it does|
|-|-|
|`fiscal_year`|Financial year for each date|
|`turn_days`|Season reset and rollover days|
|`observed_turn_price`|Year-end charge implied by the FBD April contract|
|`estimate`|Main fit plus comparison models|
|`innovation_check`|Checks on one-step forecast errors|
|`turn_by_year`|Year-end charge by financial year|
|`analyse`|Runs filter/smoother and all model checks, including turn validation|
|`plot_states`|Plot of smoothed states|
|`save_outputs`|Saves estimation outputs|

### `curve_simulation.py`: Monte Carlo check

|Function|What it does|
|-|-|
|`true_params`|Parameters used to simulate (`baseline` or `no_turn`)|
|`state_errors`|Errors in recovered states|
|`turn_by_month`|Errors in the year-end charge by month|
|`run`|Simulates and re-fits the model `reps` times|
|`save_results`|Saves results|

### `curve_turn_test.py`: LR test for the year-end factor

|Function|What it does|
|-|-|
|`restricted_fit`|Fits the model without the year-end factor|
|`compare`|LR statistic for one data set|
|`run`|LR test on real data, with simulated null LRs if given|

### `curve_drivers.py`: rates vs basis and local projections

|Function|What it does|
|-|-|
|`covariance_allocation`|Share of forward premium variance from rates and from the basis|
|`half_lives`|Half-lives of the curve factors|
|`prepare_drivers`|Builds monthly drivers from `panel_monthly.csv`|
|`projections`|Local projections of the factors on RBI drivers (lags chosen by AIC)|
|`save_results`|Saves results|

### `risk_backtest.py`: VaR/ES models and backtests

|Function|What it does|
|-|-|
|`ewma_var`|EWMA variance|
|`t_unit`, `fit_nu`|Unit-variance Student-t and its degrees of freedom|
|`build_forecasts`|Forecast mean and standard deviation for each model|
|`add_ss_t_ewma`|EWMA-rescaled state-space forecasts (used by SS-t and SS-FHS)|
|`choose_settings`|Picks the EWMA weight and degrees of freedom on the training sample|
|`var_es`, `upper_prob`|VaR, ES and exceedance probabilities|
|`empirical_forecast`|Empirical quantiles for HS and FHS|
|`kupiec`, `christoffersen`, `traffic_light`, `du_escanciano`|Backtests|
|`tick_loss`, `fz0_loss`, `diebold_mariano`|Loss functions and DM test|
|`forecasts_for`|VaR/ES series for one model|
|`evaluate`|All backtests and comparisons|
|`plot_var`|VaR against realised changes|
|`run`|Runs everything|
|`save_results`|Saves results|

### `stress_turn.py`: year-end charge, event study, stress

|Function|What it does|
|-|-|
|`paise`|Converts bp of notional to paise per USD|
|`turn_by_year`|Year-end charge each year, with bands|
|`turn_now`|Current year-end charge|
|`event_study`|Basis changes around each rollover|
|`stress`|Replays historical episodes on the latest curve, in bp and ₹|
|`plot_turn`|Year-end charge against the FBD April price|
|`run`|Runs everything|
|`save_results`|Saves results|

### `paths.py`: result folders

|Function|What it does|
|-|-|
|`new_run`|Creates a new run folder|
|`latest_run`|Finds the latest run folder|
|`step_folder`|Creates a subfolder inside a run|
|`save_settings`|Saves settings|
|`file_hash`|SHA-256 hash of a file|
|`save_run`|Saves settings, outputs, code copy, versions and hashes|

### `report.py`: tables and figures

|Function|What it does|
|-|-|
|`tables`|Shows and saves the main tables, including parameter standard errors|
|`figures`|Shows and saves the main figures|
|`keep`, `save`|Helpers to display and save a table or figure|
