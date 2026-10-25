# DNS state-space model with the year-end turn

import numpy as np
import pandas as pd


from statsmodels.tsa.statespace.mlemodel import MLEModel


STATES = ["level", "slope", "curvature", "turn"]

LAM = 0.30


# model inputs
def load_curve(curve, tenors):

    tenors = list(tenors)

    if len(tenors) < 4 or len(set(tenors)) != len(tenors):
        raise ValueError("Supply at least four distinct forward tenors")

    long = curve.copy()

    long["date"] = pd.to_datetime(long["date"])

    monthly = long[long["tenor"].isin(tenors)]

    # pivot to dates x tenors
    def wide(column):
        return monthly.pivot(
            index="date",
            columns="tenor",
            values=column
        )[tenors]

    basis = wide("basis_bp")
    index = basis.index

    fbd = long[
        long["tenor"] == "FBD April"
    ].set_index("date").reindex(index)

    year = monthly.groupby("date")[
        "turn_year"
    ].first().reindex(index)

    if year.isna().any():
        raise ValueError("Missing turn year for a model date")

    reset = np.array(year.ne(year.shift()), dtype=bool, copy=True)

    reset[0] = False

    return {
        "tenors": tenors,
        "basis": basis,
        "basis_tbill": wide("basis_tbill_bp"),
        "tau": wide("tau_months"),
        "x": wide("x_fy"),
        "n": wide("n"),
        "prem": wide("prem_bp"),
        "implied": wide("implied_bp"),
        "spot": monthly.groupby("date")["spot"].first().reindex(index),
        "fbd": fbd["basis_notional_bp"],
        "fbd_n": fbd["n"].fillna(1.0),
        "turn_year": year,
        "reset": reset
    }


# cut sample
def sample(data, start=None, end=None):

    index = data["basis"].index

    keep = (
        (index >= (start or "1900"))
        & (index <= (end or "2100"))
    )

    return {
        key: (
            value[keep]
            if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray))
            else value
        )
        for key, value in data.items()
    }

# NS parameters
def ns_params(tau, lam=LAM):

    tau = np.asarray(tau, dtype=float)

    g = (1 - np.exp(-lam * tau)) / (lam * tau)

    return g, g - np.exp(-lam * tau)


# parameter names
def parameter_names(tenors):

    names = [
        "phi_L", "phi_S", "phi_C",
        "mu_L", "mu_S", "mu_C",
        "chol_00", "chol_10", "chol_11",
        "chol_20", "chol_21", "chol_22"
    ]

    names += [f"sigma_{t}" for t in tenors]

    names += [
        "sigma_fbd", "mu_D",
        "sigma_w", "sigma_xi",
        "rho", "kappa"
    ]

    return names


# the model
class BasisCurveModel(MLEModel):

    # set up matrices
    def __init__(self, data, lam=LAM, use_fbd=True):

        self.tenors = list(data["tenors"])
        self.m = len(self.tenors)

        k = self.m + 5

        self.names = parameter_names(self.tenors)
        self.pos = {
            name: i for i, name in enumerate(self.names)
        }

        self.positive = (
            [self.pos[p] for p in [
                "chol_00", "chol_11", "chol_22"
            ]]
            + [self.pos[f"sigma_{t}"] for t in self.tenors]
            + [self.pos[p] for p in [
                "sigma_fbd", "sigma_w", "sigma_xi"
            ]]
        )

        self.corr = [
            self.pos[p] for p in [
                "phi_L", "phi_S", "phi_C", "rho", "kappa"
            ]
        ]

        y = data["basis"].copy()

        y["FBD"] = (
            data["fbd"].values if use_fbd else np.nan
        )

        super().__init__(y, k_states=k, k_posdef=k)

        n = self.nobs

        g, h = ns_params(data["tau"].values, lam)

        nf = data["fbd_n"].values
        gf, hf = ns_params(nf / (365 / 12), lam)

        Z = np.zeros((self.m + 1, k, n))

        Z[:self.m, 0] = 1.0
        Z[:self.m, 1] = g.T
        Z[:self.m, 2] = h.T
        Z[:self.m, 3] = data["x"].values.T

        Z[:self.m, 4:4 + self.m] = np.eye(self.m)[:, :, None]

        Z[self.m, 0] = nf / 365
        Z[self.m, 1] = nf / 365 * gf
        Z[self.m, 2] = nf / 365 * hf
        Z[self.m, 3] = 1.0
        Z[self.m, -1] = 1.0

        self["design"] = Z
        self["selection"] = np.eye(k)
        # no separate noise, errors are states
        self["obs_cov"] = np.zeros((self.m + 1, self.m + 1))

        self.reset = np.asarray(data["reset"], dtype=bool)


        months = [int(t[:-1]) for t in self.tenors if t.endswith("M") and t[:-1].isdigit()]
        if len(months) != self.m:
            raise ValueError("The deviation covariance needs month-labelled forward tenors")
        self.lags = np.abs(np.subtract.outer(months, months))

        self.initialize_approximate_diffuse(1e4)
        self._start: np.ndarray | None = None

    # names
    @property
    def param_names(self):
        return self.names

    # starting values
    @property
    def start_params(self):
        if self._start is None:
            raise ValueError("Starting values have not been set")

        return self._start

    # keep params valid
    def transform_params(self, u):
        p = np.array(u)

        p[self.positive] = np.exp(u[self.positive])
        p[self.corr] = np.tanh(u[self.corr]) * 0.999

        return p

    # inverse
    def untransform_params(self, p):
        u = np.array(p)

        u[self.positive] = np.log(
            np.maximum(p[self.positive], 1e-8)
        )

        u[self.corr] = np.arctanh(
            np.clip(
                p[self.corr] / 0.999,
                -0.99999,
                0.99999
            )
        )

        return u

    # fill matrices
    def update(self, params, **kwargs):

        p = super().update(params, **kwargs)

        n = self.nobs
        k = self.k_states
        m = self.m

        phi = p[0:3]
        mu = p[3:6]
        rho = p[self.pos["rho"]]

        chol = np.zeros((3, 3), dtype=p.dtype)
        chol[np.tril_indices(3)] = p[6:12]

        sig = p[12:12 + m]

        T = np.zeros((k, k, n), dtype=p.dtype)
        c = np.zeros((k, n), dtype=p.dtype)
        V = np.zeros((k, k, n), dtype=p.dtype)

        T[[0, 1, 2], [0, 1, 2]] = phi[:, None]
        T[3, 3] = 1.0
        T[range(4, k), range(4, k)] = rho

        c[:3] = ((1 - phi) * mu)[:, None]

        V[:3, :3] = (chol @ chol.T)[:, :, None]

        V[3, 3] = p[self.pos["sigma_w"]] ** 2

        V[4:4 + m, 4:4 + m] = (
            np.outer(sig, sig)
            * p[self.pos["kappa"]] ** self.lags
            * (1 - rho ** 2)
        )[:, :, None]

        V[-1, -1] = (
            p[self.pos["sigma_fbd"]] ** 2
            * (1 - rho ** 2)
        )

        # new season: reset turn on the day before
        r = np.where(self.reset)[0] - 1
        r = r[r >= 0]

        T[3, 3, r] = 0.0
        c[3, r] = p[self.pos["mu_D"]]
        V[3, 3, r] = p[self.pos["sigma_xi"]] ** 2

        self["transition"] = T
        self["state_intercept"] = c
        self["state_cov"] = V


# daily OLS (two-step)
def cross_section(data, lam=LAM):

    m = len(data["tenors"])

    B = np.full((len(data["basis"]), 4), np.nan)
    R = np.full(data["basis"].shape, np.nan)

    for i in range(len(B)):

        y = data["basis"].values[i]
        x = data["x"].values[i]

        g, h = ns_params(data["tau"].values[i], lam)

        good = np.isfinite(y) & np.isfinite(x) & np.isfinite(g) & np.isfinite(h)
        if good.sum() < 4:
            continue

        X = np.column_stack([np.ones(m), g, h, x])[good]
        X = X[:, :4 if x[good].max() > 0 else 3]
        b = np.linalg.lstsq(X, y[good], rcond=None)[0]

        B[i, :len(b)] = b
        R[i, good] = y[good] - X @ b

    index = data["basis"].index

    factors = pd.DataFrame(
        B, index=index, columns=STATES
    )

    residuals = pd.DataFrame(
        R, index=index, columns=data["tenors"]
    )

    return factors, residuals


# starting values
def start_values(data, lam=LAM, phi_cap=0.95):

    F, R = cross_section(data, lam)

    f = F[STATES[:3]]

    phi = np.array([
        np.clip(f[s].autocorr(), 0.5, phi_cap)
        for s in STATES[:3]
    ])

    mu = f.mean().values

    eta = (
        f - f.shift() * phi - (1 - phi) * mu
    ).dropna()

    chol = np.linalg.cholesky(
        np.cov(eta.values.T) + 1e-6 * np.eye(3)
    )[np.tril_indices(3)]

    sig = np.sqrt(
        (R ** 2).mean()
    ).clip(0.5).values

    d = F["turn"].dropna()

    return np.r_[
        phi, mu, chol, sig,
        2.0, d.median(), 0.5,
        max(d.std(), 1.0),
        0.5, 0.3
    ]


# fixed params
def held(turn_walk=True, use_fbd=True, fixed=None):

    fixed = dict(fixed or {})

    if not turn_walk:
        fixed["sigma_w"] = 1e-4

    if not use_fbd:
        fixed["sigma_fbd"] = 1.0

    return fixed


# one ML run
def mle(data, start, fixed=None, lam=LAM,
        use_fbd=True, maxiter=1000):

    model = BasisCurveModel(data, lam, use_fbd)

    hold = dict(fixed or {})

    st = np.asarray(start, dtype=float).copy()

    if len(st) != len(model.param_names):
        raise ValueError("Wrong number of starting parameters")

    for name, value in hold.items():
        st[model.pos[name]] = value

    model._start = st

    if hold:
        free = [
            i for i, name in enumerate(model.param_names)
            if name not in hold
        ]

        res = model.fit_constrained(
            hold,
            start_params=st[free],
            method="lbfgs",
            maxiter=maxiter,
            disp=False
        )

    else:
        res = model.fit(
            start_params=st,
            method="lbfgs",
            maxiter=maxiter,
            disp=False
        )

    return model, res


# ML fit, two routes
def fit(data, lam=LAM, turn_walk=True, use_fbd=True,
        start=None, fixed=None, maxiter=1000,
        two_routes=True):

    hold = held(turn_walk, use_fbd, fixed)

    st = (
        start_values(data, lam)
        if start is None
        else np.asarray(start, dtype=float)
    )

    best = mle(data, st, hold, lam, use_fbd, maxiter)

    if not two_routes:
        return best

    phis = {
        p: 0.97
        for p in ["phi_L", "phi_S", "phi_C"]
        if p not in hold
    }

    _, first = mle(
        data, st, {**hold, **phis},
        lam, use_fbd, maxiter
    )

    second = mle(
        data,
        np.asarray(first.params),
        hold,
        lam,
        use_fbd,
        maxiter
    )

    return second if second[1].llf > best[1].llf else best


# filter + smoother
def run_filter(params, data, lam=LAM, use_fbd=True):

    model = BasisCurveModel(data, lam, use_fbd)

    model._start = np.asarray(params, dtype=float)

    return model.smooth(params)


# states with bands
def states_table(result, index, which="smoothed"):

    if which not in {"smoothed", "filtered"}:
        raise ValueError("Choose 'smoothed' or 'filtered'")

    values = (
        result.smoothed_state
        if which == "smoothed"
        else result.filtered_state
    )

    cov = (
        result.smoothed_state_cov
        if which == "smoothed"
        else result.filtered_state_cov
    )

    out = {}

    for i, state in enumerate(STATES):

        sd = np.sqrt(cov[i, i])

        out[state] = values[i]
        out[f"{state}_lo"] = values[i] - 1.645 * sd
        out[f"{state}_hi"] = values[i] + 1.645 * sd

    return pd.DataFrame(out, index=index)


# phi profile
def profile_phi(
    data,
    params,
    values=(0.95, 0.97, 0.99, 0.999),
    lam=LAM,
    turn_walk=True,
    use_fbd=True,
    maxiter=1000
):

    rows = []

    for phi in values:

        fixed = held(turn_walk, use_fbd)

        fixed.update({
            "phi_L": phi,
            "phi_S": phi,
            "phi_C": phi
        })

        _, res = mle(
            data,
            params,
            fixed=fixed,
            lam=lam,
            use_fbd=use_fbd,
            maxiter=maxiter
        )

        rows.append({
            "phi_all_three": phi,
            "loglik": res.llf,
            "converged": bool(
                res.mle_retvals["converged"]
            )
        })

    return pd.DataFrame(rows)


# simulate data
def simulate(params, data, rng, lam=LAM, mu_D=None):

    tenors = data["tenors"]
    m = len(tenors)

    names = parameter_names(tenors)
    P = {name: i for i, name in enumerate(names)}

    p = np.asarray(params, dtype=float).copy()

    if mu_D is not None:
        p[P["mu_D"]] = mu_D

        if mu_D == 0:
            p[P["sigma_xi"]] = 0.0
            p[P["sigma_w"]] = 0.0

    n = len(data["basis"])

    phi = p[:3]
    mu = p[3:6]
    rho = p[P["rho"]]
    kappa = p[P["kappa"]]

    chol = np.zeros((3, 3))
    chol[np.tril_indices(3)] = p[6:12]

    sig = p[12:12 + m]

    months = [int(t[:-1]) for t in tenors]
    lags = np.abs(np.subtract.outer(months, months))

    chol_dev = np.linalg.cholesky(
        np.outer(sig, sig) * kappa ** lags
    )

    g, h = ns_params(data["tau"].values, lam)

    nf = data["fbd_n"].values
    gf, hf = ns_params(nf / (365 / 12), lam)

    f = mu.copy()
    D = p[P["mu_D"]]

    e = chol_dev @ rng.standard_normal(m)
    ea = p[P["sigma_fbd"]] * rng.standard_normal()

    y = np.zeros((n, m + 1))
    states = np.zeros((n, 4))

    for t in range(n):

        if t > 0:

            f = (
                mu
                + phi * (f - mu)
                + chol @ rng.standard_normal(3)
            )

            e = (
                rho * e
                + np.sqrt(1 - rho ** 2)
                * chol_dev @ rng.standard_normal(m)
            )

            ea = (
                rho * ea
                + np.sqrt(1 - rho ** 2)
                * p[P["sigma_fbd"]]
                * rng.standard_normal()
            )

            shock = rng.standard_normal()

            if data["reset"][t]:
                D = (
                    p[P["mu_D"]]
                    + p[P["sigma_xi"]] * shock
                )
            else:
                D = D + p[P["sigma_w"]] * shock

        y[t, :m] = (
            f[0]
            + f[1] * g[t]
            + f[2] * h[t]
            + D * data["x"].values[t]
            + e
        )

        y[t, m] = (
            nf[t] / 365
            * (f[0] + f[1] * gf[t] + f[2] * hf[t])
            + D
            + ea
        )

        states[t] = [*f, D]

    out = dict(data)

    idx = data["basis"].index

    out["basis"] = pd.DataFrame(
        y[:, :m],
        index=idx,
        columns=tenors
    )

    out["fbd"] = pd.Series(
        np.where(data["fbd"].notna(), y[:, m], np.nan),
        index=idx
    )

    true_states = pd.DataFrame(
        states,
        index=idx,
        columns=STATES
    )

    return out, true_states
