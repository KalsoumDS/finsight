"""
Unit tests for FinSight GARCH(1,1) model.

All tests use synthetic return series — no live data required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from core.garch import GARCHModel


@pytest.fixture(scope="module")
def returns_series() -> pd.Series:
    """500-day synthetic AR-GARCH-like returns with volatility clustering."""
    rng = np.random.default_rng(42)
    n = 500
    r = rng.normal(0.0005, 0.015, n)
    # Inject a volatility spike in the middle
    r[200:230] = rng.normal(0.0, 0.04, 30)
    return pd.Series(r)


@pytest.fixture(scope="module")
def fitted_model(returns_series) -> GARCHModel:
    return GARCHModel().fit(returns_series)


class TestGARCHParameters:

    def test_omega_strictly_positive(self, fitted_model):
        """omega must be strictly positive (unconditional variance base)."""
        assert fitted_model.omega_ > 0

    def test_alpha_non_negative(self, fitted_model):
        """alpha (ARCH effect) must be in [0, 1)."""
        assert 0 <= fitted_model.alpha_ < 1

    def test_beta_non_negative(self, fitted_model):
        """beta (GARCH smoothing) must be in [0, 1)."""
        assert 0 <= fitted_model.beta_ < 1

    def test_stationarity_constraint(self, fitted_model):
        """alpha + beta < 1 is required for covariance stationarity."""
        persistence = fitted_model.alpha_ + fitted_model.beta_
        assert persistence < 1.0, f"alpha+beta = {persistence:.4f} violates stationarity"

    def test_sigma2_strictly_positive(self, fitted_model):
        """All conditional variances must be strictly positive."""
        assert (fitted_model.sigma2_ > 0).all()

    def test_loglikelihood_finite(self, fitted_model):
        """Log-likelihood must be a finite real number."""
        assert np.isfinite(fitted_model.loglik_)


class TestGARCHForecasts:

    def test_forecast_variance_positive(self, fitted_model):
        """One-step-ahead variance forecast must be positive."""
        fcast = fitted_model.forecast_variance(h=1)
        assert fcast[0] > 0

    def test_forecast_horizon_length(self, fitted_model):
        """forecast_variance(h) must return exactly h values."""
        for h in [1, 5, 10, 22]:
            assert len(fitted_model.forecast_variance(h)) == h

    def test_long_run_mean_reversion(self, fitted_model):
        """Long-horizon forecasts must converge toward unconditional variance."""
        short_fcast = fitted_model.forecast_variance(h=1)[0]
        long_fcast  = fitted_model.forecast_variance(h=252)[-1]
        omega, alpha, beta = fitted_model.omega_, fitted_model.alpha_, fitted_model.beta_
        unconditional = omega / (1 - alpha - beta)
        # Long-run forecast should be closer to unconditional than short-run
        assert abs(long_fcast - unconditional) <= abs(short_fcast - unconditional) + 1e-8

    def test_annualised_vol_reasonable(self, fitted_model):
        """Annualised volatility forecast must be in a plausible range (1-200%)."""
        ann_vol = fitted_model.forecast_volatility(1)[0]
        assert 0.01 <= ann_vol <= 2.0


class TestGARCHVaR:

    def test_garch_var_positive(self, fitted_model):
        """GARCH-VaR must be expressed as a positive loss magnitude."""
        assert fitted_model.garch_var(confidence=0.95) > 0

    def test_var99_exceeds_var95(self, fitted_model):
        """VaR at 99% must exceed VaR at 95%."""
        var95 = fitted_model.garch_var(confidence=0.95)
        var99 = fitted_model.garch_var(confidence=0.99)
        assert var99 > var95

    def test_garch_var_horizon_scaling(self, fitted_model):
        """10-day VaR must exceed 1-day VaR (sqrt-of-time scaling)."""
        var1  = fitted_model.garch_var(confidence=0.95, horizon=1)
        var10 = fitted_model.garch_var(confidence=0.95, horizon=10)
        assert var10 > var1


class TestGARCHSummary:

    def test_summary_has_required_keys(self, fitted_model):
        s = fitted_model.summary()
        required = {"omega", "alpha", "beta", "persistence", "half_life_days",
                    "loglikelihood", "AIC", "BIC", "forecast_vol_ann",
                    "garch_var_95", "garch_var_99"}
        assert required <= s.keys()

    def test_unfitted_model_raises(self):
        with pytest.raises(RuntimeError):
            GARCHModel().summary()
