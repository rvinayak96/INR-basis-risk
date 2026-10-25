# USD/INR basis risk

Code for building a daily USD/INR forward basis curve (1M–12M), modelling it with a dynamic Nelson–Siegel state-space model that includes a 31 March year-end factor, and backtesting VaR/ES models for basis risk. The research note is available on request.

## Setup

```
pip install -r requirements.txt
jupyter notebook USDINR\_Basis\_Research.ipynb
```

Tested with Python 3.12, numpy 2.3.5, pandas 2.2.3, scipy 1.17.0, statsmodels 0.15.0, matplotlib 3.11.1.

## Folder layout

```
project/
├── USDINR\_Basis\_Research.ipynb   runs everything
├── \*.py                          code (below)
├── raw/                          input files (not included)
└── results/                      one new folder per run (created automatically)
```

Keep the notebook, the `.py` files and `raw/` in the same folder.

## How to run

1. Put the data files in `raw/`.
2. Open the notebook. In the **Settings** cell, set the file names in `FILES` and the options in `S`.
3. Run all cells. Each run is saved to `results/basis\_run\_<date>\_<time>/` with its settings, outputs, a copy of the code, package versions and file hashes. Nothing is overwritten.
4. The last cell calls `report.py`, which shows the main tables and figures and saves them in the same run folder.

### Main settings (`S`)

|Setting|What it does|
|-|-|
|`data\_mode`|`"raw"` builds the basis from the raw files; `"csv"` loads a saved `basis\_curve\_long.csv`|
|`param\_mode`|`"fit"` estimates the model (about 25 minutes); `"saved"` loads `params.csv`|
|`start`, `end`, `train\_end`, `test\_start`|sample, estimation and test periods|
|`tenors`|forward tenors used (at least four, labelled like `"3M"`)|
|`lambda`|Nelson–Siegel decay (per month)|
|`inr\_rate`|`"ois"` or `"tbill"` for the rupee rate|
|`us\_switch`|date from which FRED series replace the Treasury par yield file|
|`turn\_walk`, `use\_fbd`|year-end charge as a random walk; use the FBD April contract|
|`two\_routes`, `maxiter`|optimiser settings|
|`comparisons`|also fit the comparison models (needed for the turn validation)|
|`risk`, `stress`, `local\_projections`, `simulation`, `null\_simulation`, `turn\_test`|switch each analysis on or off|
|`risk\_tenors`, `risk\_models`, `risk\_levels`, `risk\_window`|risk backtest options|
|`save`|save the run to `results/`|

### Typical workflow

* **First run:** `data\_mode="raw"`, `param\_mode="fit"`, `comparisons=True`.
* Then copy `params.csv` and `basis\_curve\_long.csv` from that run folder into `raw/`.
* **Later runs:** `data\_mode="csv"`, `param\_mode="saved"`, with risk, stress or simulations switched on. Nothing is re-estimated.
* Simulations and the turn test are slow (several hours). Set `two\_routes=False` for them.

## Data

|Key in `FILES`|Source|Format expected|
|-|-|-|
|`forwards`|FBIL USD/INR forward premia|FBIL download as is: two title rows, then `Date, Time, Tenor, Settlement Date, USD/INR Premia(%), USD/INR Premia(Rs.), Comments`; dates like `6-Oct-26`. Must include the `Spot` and `FBD April` rows|
|`ois`|FBIL MIBOR-OIS|FBIL download as is: two title rows, then `Date, Time, Period, Rate(%), Comments`|
|`tbill`|FBIL T-bill curve|FBIL download as is: two title rows, then `Date, Time, Tenor, Rate(%), Comments`|
|`us\_par`|US Treasury par yield curve|columns `date` (mm/dd/yyyy), `1 mo`, `3 mo`, `6 mo`, `1 yr`, …|
|`us\_1m`, `us\_3m`, `us\_6m`, `us\_1y`|FRED DGS1MO, DGS3MO, DGS6MO, DGS1|FRED CSV as downloaded: two columns, date and value|
|(optional)|US Treasury daily bill rates|files named `us\_treasury\_bill\_rates\_\*.csv`; used only for rate checks|
|`monthly`|RBI and FRED (hand-assembled)|one row per month, column `date`, and `net\_purchase\_usd\_mn`, `chg\_net\_forward\_usd\_mn`, `net\_liq\_inj\_lakh\_cr`, `broad\_usd\_index`; optional `n\_days`. Only needed for local projections|

### Do

* Use the files exactly as downloaded, and set their names in `FILES`.
* Make sure the sample covers every year end you include: the code needs FBIL's FBD April date for each year (or set `fbd\_fallback=True` to assume 1 April).
* Use `param\_mode="fit"` whenever you change `start`, `train\_end`, `tenors`, `lambda`, `inr\_rate` or `us\_switch`. The notebook stops if saved parameters don't match these settings.

### Don't

* Don't open and re-save the CSVs in Excel. Excel changes date formats (`6-Oct-26`) and the loaders will fail or misread dates.
* Don't rename columns or delete the title rows in FBIL files.
* Don't move `raw/` away from the code; paths are set relative to the code folder.
* Don't edit files inside `results/`; copy what you need into `raw/` instead.

## Code files and functions

### `data\_prep.py`: loading raw files

|Function|What it does|
|-|-|
|`standardize\_curve`|Checks a rate table (dates × days to maturity): no duplicates, positive maturities, numeric rates|
|`load\_rate\_files`|Builds a curve from one file per maturity|
|`load\_long\_curve`|Reads FBIL OIS or T-bill files into a dates × days curve|
|`load\_wide\_curve`|Reads a curve saved in wide format|
|`load\_fbil\_forwards`|Reads FBIL forward premia; returns forwards with spot settlement date and spot rate|
|`read\_fred`|Reads a FRED series|
|`zero\_1y`|Converts the 1Y par yield to a zero rate|
|`load\_us\_treasury`|Combines the Treasury par file (before `us\_switch`) and FRED series (after) into one US curve|

### `curve\_build.py`: building the basis curve

|Function|What it does|
|-|-|
|`interpolate`|Linear interpolation of a rate curve to each contract's number of days|
|`implied\_bp`|CIP-implied premium in annualised bp|
|`prepare\_forwards`|Cleans forwards and computes contract days|
|`turn\_calendar`|Finds the FBD April settlement date for each year|
|`add\_turn\_exposure`|Marks contracts that settle after the coming year end (exposure 365/n)|
|`match\_us\_dates`|Uses the previous US business day for US rates|
|`build`|Builds the long basis table for all tenors and the FBD April contract|
|`checks`|Basic checks by tenor (counts, missing values, US data lag)|
|`rate\_checks`|OIS vs T-bill and Treasury CMT vs bill-rate comparisons|
|`load\_us\_bills`|Reads US daily bill-rate files|

### `curve\_model.py`: the state-space model

|Function|What it does|
|-|-|
|`load\_curve`|Turns the long basis table into model inputs (dates × tenors)|
|`sample`|Cuts the inputs to a date range|
|`ns\_params`|Nelson–Siegel slope and curvature loadings|
|`parameter\_names`|Names of the 30 parameters|
|`BasisCurveModel`|The model (statsmodels `MLEModel`): level, slope, curvature, year-end charge, AR(1) tenor deviations, FBD April observation|
|`cross\_section`|Daily OLS fit of the curve (two-step Diebold–Li)|
|`start\_values`|Starting values from the two-step fit|
|`held`|Parameters to hold fixed for restricted versions|
|`mle`|One maximum-likelihood run (L-BFGS)|
|`fit`|Fit with two starting routes; keeps the better one|
|`run\_filter`|Kalman filter and smoother at given parameters|
|`states\_table`|Filtered or smoothed states with 90% bands|
|`profile\_phi`|Likelihood profile over factor persistence|
|`simulate`|Simulates data from the model|

### `curve\_estimate.py`: estimation and checks

|Function|What it does|
|-|-|
|`fiscal\_year`|Financial year for each date|
|`turn\_days`|Season reset and rollover days|
|`observed\_turn\_price`|Year-end charge implied by the FBD April contract|
|`estimate`|Main fit plus comparison models|
|`innovation\_check`|Checks on one-step forecast errors|
|`turn\_by\_year`|Year-end charge by financial year|
|`analyse`|Runs filter/smoother and all model checks, including turn validation|
|`plot\_states`|Plot of smoothed states|
|`save\_outputs`|Saves estimation outputs|

### `curve\_simulation.py`: Monte Carlo check

|Function|What it does|
|-|-|
|`true\_params`|Parameters used to simulate (`baseline` or `no\_turn`)|
|`state\_errors`|Errors in recovered states|
|`turn\_by\_month`|Errors in the year-end charge by month|
|`run`|Simulates and re-fits the model `reps` times|
|`save\_results`|Saves results|

### `curve\_turn\_test.py`: LR test for the year-end factor

|Function|What it does|
|-|-|
|`restricted\_fit`|Fits the model without the year-end factor|
|`compare`|LR statistic for one data set|
|`run`|LR test on real data, with simulated null LRs if given|

### `curve\_drivers.py`: rates vs basis and local projections

|Function|What it does|
|-|-|
|`covariance\_allocation`|Share of forward premium variance from rates and from the basis|
|`half\_lives`|Half-lives of the curve factors|
|`prepare\_drivers`|Builds monthly drivers from `panel\_monthly.csv`|
|`projections`|Local projections of the factors on RBI drivers (lags chosen by AIC)|
|`save\_results`|Saves results|

### `risk\_backtest.py`: VaR/ES models and backtests

|Function|What it does|
|-|-|
|`ewma\_var`|EWMA variance|
|`t\_unit`, `fit\_nu`|Unit-variance Student-t and its degrees of freedom|
|`build\_forecasts`|Forecast mean and standard deviation for each model|
|`add\_ss\_t\_ewma`|EWMA-rescaled state-space forecasts (used by SS-t and SS-FHS)|
|`choose\_settings`|Picks the EWMA weight and degrees of freedom on the training sample|
|`var\_es`, `upper\_prob`|VaR, ES and exceedance probabilities|
|`empirical\_forecast`|Empirical quantiles for HS and FHS|
|`kupiec`, `christoffersen`, `traffic\_light`, `du\_escanciano`|Backtests|
|`tick\_loss`, `fz0\_loss`, `diebold\_mariano`|Loss functions and DM test|
|`forecasts\_for`|VaR/ES series for one model|
|`evaluate`|All backtests and comparisons|
|`plot\_var`|VaR against realised changes|
|`run`|Runs everything|
|`save\_results`|Saves results|

### `stress\_turn.py`: year-end charge, event study, stress

|Function|What it does|
|-|-|
|`paise`|Converts bp of notional to paise per USD|
|`turn\_by\_year`|Year-end charge each year, with bands|
|`turn\_now`|Current year-end charge|
|`event\_study`|Basis changes around each rollover|
|`stress`|Replays historical episodes on the latest curve, in bp and ₹|
|`plot\_turn`|Year-end charge against the FBD April price|
|`run`|Runs everything|
|`save\_results`|Saves results|

### `paths.py`: result folders

|Function|What it does|
|-|-|
|`new\_run`|Creates a new run folder|
|`latest\_run`|Finds the latest run folder|
|`step\_folder`|Creates a subfolder inside a run|
|`save\_settings`|Saves settings|
|`file\_hash`|SHA-256 hash of a file|
|`save\_run`|Saves settings, outputs, code copy, versions and hashes|

### `report.py`: tables and figures

|Function|What it does|
|-|-|
|`tables`|Shows and saves the main tables, including parameter standard errors|
|`figures`|Shows and saves the main figures|
|`keep`, `save`|Helpers to display and save a table or figure|



