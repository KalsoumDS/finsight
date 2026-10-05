"""
core/garch.py — GARCH(1,1) Conditional Volatility Model

Implements a pure-NumPy GARCH(1,1) estimator for daily return series.
Used by FinSight for dynamic VaR (GARCH-VaR) and volatility forecasting.

The GARCH(1,1) model is defined as:
    sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}

where:
    epsilon_t = r_t - mu  (zero-mean residuals)
    sigma^2_t             (conditional variance at time t)
    omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1 (stationarity)

References:
    Bollerslev (1986) "Generalized Autoregressive Conditional Heteroskedasticity"
    Journal of Econometrics, 31(3), 307-327.
"""
from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats


class GARCHModel:
    """
    GARCH(1,1) conditional heteroskedasticity model.

    Fits omega, alpha, beta via maximum likelihood (Gaussian innovations).
    Provides one-step-ahead volatility forecast and GARCH-VaR.

    Parameters
    ----------
    returns : array-like of daily log-returns or percentage returns.

    Attributes
    ----------
    omega_, alpha_, beta_ : fitted GARCH parameters (available after fit).
    sigma2_               : fitted conditional variance series.
    loglik_               : maximised log-likelihood.
    aic_, bic_            : information criteria.
    """

    def __init__(self) -> None:
        self.omega_: float | None = None
        self.alpha_: float | None = None
        self.beta_:  float | None = None
        self.sigma2_: np.ndarray | None = None
        self.loglik_: float | None = None
        self.aic_:   float | None = None
        self.bic_:   float | None = None
        self._returns: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sigma2(
        params: np.ndarray, residuals: np.ndarray
    ) -> np.ndarray:
        """Recursively compute the conditional variance series."""
        omega, alpha, beta = params
        n = len(residuals)
        sigma2 = np.empty(n)
        sigma2[0] = np.var(residuals)  # initialise with unconditional variance
        for t in range(1, n):
            sigma2[t] = omega + alpha * residuals[t - 1] ** 2 + beta * sigma2[t - 1]
        return sigma2

    @staticmethod
    def _neg_loglik(params: np.ndarray, residuals: np.ndarray) -> float:
        """Gaussian GARCH(1,1) negative log-likelihood."""
        omega, alpha, beta = params
        # Parameter constraints
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        sigma2 = GARCHModel._compute_sigma2(params, residuals)
        if np.any(sigma2 <= 0):
            return 1e10
        log_lik = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + residuals ** 2 / sigma2)
        return -log_lik

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, returns: "np.ndarray | pd.Series") -> "GARCHModel":
        """
        Estimate GARCH(1,1) parameters via maximum likelihood.

        Parameters
        ----------
        returns : daily return series (fractional, e.g. 0.01 = 1%).

        Returns
        -------
        self (fitted model).
        """
        if isinstance(returns, pd.Series):
            returns = returns.dropna().values
        self._returns = np.asarray(returns, dtype=np.float64)
        residuals = self._returns - self._returns.mean()

        # Initial parameter guess: unconditional variance decomposition
        var_unc = np.var(residuals)
        x0 = np.array([var_unc * 0.05, 0.1, 0.85])
        bounds = [(1e-8, None), (0.0, 1.0), (0.0, 1.0)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                self._neg_loglik,
                x0,
                args=(residuals,),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-12},
            )

        if not result.success:
            # Fallback: use initial values
            self.omega_ = float(x0[0])
            self.alpha_ = float(x0[1])
            self.beta_  = float(x0[2])
        else:
            self.omega_ = float(result.x[0])
            self.alpha_ = float(result.x[1])
            self.beta_  = float(result.x[2])

        self.sigma2_ = self._compute_sigma2(
            np.array([self.omega_, self.alpha_, self.beta_]), residuals
        )
        self.loglik_ = float(-result.fun)

        n = len(residuals)
        k = 3  # omega, alpha, beta
        self.aic_ = float(2 * k - 2 * self.loglik_)
        self.bic_ = float(k * np.log(n) - 2 * self.loglik_)

        return self

    # ------------------------------------------------------------------

    def forecast_variance(self, h: int = 1) -> np.ndarray:
        """
        Multi-step conditional variance forecasts.

        Parameters
        ----------
        h : forecast horizon in days.

        Returns
        -------
        Array of length h with h-step-ahead variance forecasts.
        """
        if self.sigma2_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        omega, alpha, beta = self.omega_, self.alpha_, self.beta_
        forecasts = np.empty(h)
        last_sigma2 = self.sigma2_[-1]
        last_eps2   = (self._returns[-1] - self._returns.mean()) ** 2
        sigma2_h    = omega + alpha * last_eps2 + beta * last_sigma2
        forecasts[0] = sigma2_h
        for i in range(1, h):
            sigma2_h = omega + (alpha + beta) * sigma2_h
            forecasts[i] = sigma2_h
        return forecasts

    def forecast_volatility(self, h: int = 1) -> np.ndarray:
        """Annualised volatility forecasts (sqrt of daily variance * sqrt(252))."""
        return np.sqrt(self.forecast_variance(h) * 252)

    # ------------------------------------------------------------------

    def garch_var(self, confidence: float = 0.95, horizon: int = 1) -> float:
        """
        GARCH-VaR: Value-at-Risk using the one-step-ahead conditional volatility.

        More conservative than historical VaR during volatile regimes.

        Parameters
        ----------
        confidence : confidence level (e.g. 0.95).
        horizon    : VaR horizon in days (sqrt-of-time scaling).

        Returns
        -------
        VaR as a positive loss magnitude (fraction of portfolio value).
        """
        if self.sigma2_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        sigma_forecast = float(np.sqrt(self.forecast_variance(1)[0]))
        z_score = float(stats.norm.ppf(1 - confidence))
        return abs(z_score) * sigma_forecast * np.sqrt(horizon)

    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """Return a dict of model diagnostics suitable for display."""
        if self.omega_ is None:
            raise RuntimeError("Model not fitted.")
        persistence = self.alpha_ + self.beta_
        half_life   = float(np.log(0.5) / np.log(persistence)) if persistence < 1 else float("inf")
        ann_vol_forecast = float(self.forecast_volatility(1)[0])
        return {
            "omega":             round(self.omega_, 8),
            "alpha":             round(self.alpha_, 4),
            "beta":              round(self.beta_,  4),
            "persistence":       round(persistence, 4),
            "half_life_days":    round(half_life,   1),
            "loglikelihood":     round(self.loglik_, 2),
            "AIC":               round(self.aic_,    2),
            "BIC":               round(self.bic_,    2),
            "forecast_vol_ann":  f"{ann_vol_forecast * 100:.2f}%",
            "garch_var_95":      f"{self.garch_var(0.95) * 100:.3f}%",
            "garch_var_99":      f"{self.garch_var(0.99) * 100:.3f}%",
        }
