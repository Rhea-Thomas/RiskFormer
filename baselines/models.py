"""
Baseline forecasters.

Every baseline exposes the same interface as RiskFormer's output:

    model = Baseline().fit(train_split)
    out = model.predict(split)  # {'mu': (N, A), 'sigma': (N, A), 'quantiles': (N, A, 3)}

Units match the targets: percent daily log returns and percent daily vol.
Quantiles are Gaussian (μ ± z·σ) unless a model produces its own.

Return baselines
  zero        random walk: μ = 0 (the honest benchmark for daily returns)
  persistence μ = yesterday's return
  hist_mean   μ = training-period mean return
  moving_avg  μ = mean of the last 20 returns
  ridge       μ = Ridge regression on the last day's features
  gbm         μ = gradient boosting on the last day's features

Volatility baselines (used as σ)
  rolling20   σ = std of last 20 returns
  ewma        σ = RiskMetrics EWMA, λ = 0.94
  garch       σ = GARCH(1,1) fitted by maximum likelihood on train
  har         σ = HAR regression on |r|, 5-day, 22-day realized vol (Corsi, 2009)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

QUANTILES = (0.05, 0.5, 0.95)


def gaussian_quantiles(mu, sigma, quantiles=QUANTILES):
    z = norm.ppf(np.asarray(quantiles))
    return mu[..., None] + sigma[..., None] * z


# --------------------------------------------------------------------- vol
def rolling_vol(hist, window=20):
    return hist[:, -window:, :].std(axis=1, ddof=1)


def ewma_vol(hist, lam=0.94):
    L = hist.shape[1]
    w = (1 - lam) * lam ** np.arange(L - 1, -1, -1)  # most recent gets most weight
    w = w / w.sum()
    return np.sqrt(np.einsum("l,nla->na", w, hist**2))


class GarchVol:
    """GARCH(1,1) with zero mean, one parameter set per asset, fit on train returns."""

    def fit(self, returns):  # returns: (T, A) contiguous training returns
        self.params = []
        for a in range(returns.shape[1]):
            r = returns[:, a]
            r = r[np.isfinite(r)]
            var0 = r.var()

            def nll(p):
                omega, alpha, beta = p
                if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
                    return 1e10
                h = np.empty_like(r)
                h[0] = var0
                for t in range(1, len(r)):
                    h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
                return 0.5 * np.sum(np.log(h) + r**2 / h)

            res = minimize(nll, x0=[var0 * 0.05, 0.08, 0.9], method="Nelder-Mead",
                           options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6})
            self.params.append(res.x)
        return self

    def predict(self, hist):  # hist: (N, L, A) -> one-step-ahead σ
        N, L, A = hist.shape
        out = np.empty((N, A))
        for a, (omega, alpha, beta) in enumerate(self.params):
            r = hist[:, :, a]
            h = np.full(N, omega / max(1e-6, 1 - alpha - beta))  # start at long-run var
            for t in range(L):
                h = omega + alpha * r[:, t] ** 2 + beta * h
            out[:, a] = np.sqrt(h)
        return out


def har_features(hist):
    """[|r_{t-1}|, 5-day vol, 22-day vol] per asset: (N, A, 3) in log space."""
    f = np.stack(
        [np.abs(hist[:, -1, :]), hist[:, -5:, :].std(1, ddof=1), hist[:, -22:, :].std(1, ddof=1)], -1
    )
    return np.log(f + 1e-3)


# ---------------------------------------------------------------- baselines
class Baseline:
    name = "baseline"
    vol = "ewma"

    def fit(self, train):
        if self.vol == "garch":
            # Contiguous training return series: first window, then each target day
            series = np.concatenate([train.hist_returns[0], train.Y[:, :, 0]], axis=0)
            self._garch = GarchVol().fit(series)
        if self.vol == "har":
            Xh = har_features(train.hist_returns).reshape(-1, 3)
            yh = np.log(train.Y[..., 1].reshape(-1) + 1e-3)
            self._har = Ridge(alpha=1.0).fit(Xh, yh)
        self._fit_mean(train)
        return self

    def _fit_mean(self, train):
        pass

    def _mu(self, split):
        raise NotImplementedError

    def _sigma(self, split):
        h = split.hist_returns
        if self.vol == "rolling20":
            return rolling_vol(h, 20)
        if self.vol == "ewma":
            return ewma_vol(h)
        if self.vol == "garch":
            return self._garch.predict(h)
        if self.vol == "har":
            N, A = h.shape[0], h.shape[2]
            return np.exp(self._har.predict(har_features(h).reshape(-1, 3))).reshape(N, A)
        raise ValueError(self.vol)

    def predict(self, split):
        mu = np.asarray(self._mu(split), dtype=np.float64)
        sigma = np.maximum(np.asarray(self._sigma(split), dtype=np.float64), 1e-3)
        return {"mu": mu, "sigma": sigma, "quantiles": gaussian_quantiles(mu, sigma)}


class ZeroMean(Baseline):
    name = "zero"

    def __init__(self, vol="ewma"):
        self.vol = vol
        self.name = f"zero+{vol}"

    def _mu(self, split):
        return np.zeros(split.Y.shape[:2])


class Persistence(Baseline):
    name = "persistence"

    def _mu(self, split):
        return split.hist_returns[:, -1, :]


class HistoricalMean(Baseline):
    name = "hist_mean"

    def _fit_mean(self, train):
        self.mean_ = train.Y[..., 0].mean(axis=0)

    def _mu(self, split):
        return np.broadcast_to(self.mean_, split.Y.shape[:2])


class MovingAverage(Baseline):
    name = "moving_avg"

    def __init__(self, window=20):
        self.window = window

    def _mu(self, split):
        return split.hist_returns[:, -self.window :, :].mean(axis=1)


class _FeatureRegressor(Baseline):
    """Pooled across assets: last-day normalized features + one-hot asset id."""

    def _design(self, split):
        N, _, A, F = split.X.shape
        last = split.X[:, -1, :, :].reshape(N * A, F).astype(np.float64)
        onehot = np.tile(np.eye(A), (N, 1))
        return np.hstack([last, onehot])

    def _fit_mean(self, train):
        self.reg_ = self._make().fit(self._design(train), train.Y[..., 0].reshape(-1))

    def _mu(self, split):
        N, A = split.Y.shape[:2]
        return self.reg_.predict(self._design(split)).reshape(N, A)


class RidgeReturn(_FeatureRegressor):
    name = "ridge"

    def _make(self):
        return Ridge(alpha=10.0)


class GradientBoostingReturn(_FeatureRegressor):
    name = "gbm"

    def _make(self):
        return HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.03, max_depth=3, min_samples_leaf=50,
            l2_regularization=1.0, random_state=0,
        )


def default_baselines():
    """The comparison set used in evaluate.py."""
    return [
        ZeroMean("rolling20"),
        ZeroMean("ewma"),
        ZeroMean("garch"),
        ZeroMean("har"),
        Persistence(),
        HistoricalMean(),
        MovingAverage(20),
        RidgeReturn(),
        GradientBoostingReturn(),
    ]
