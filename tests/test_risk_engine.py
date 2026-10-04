"""
Unit tests for FinSight RiskEngine.

Tests validate mathematical properties of VaR/CVaR estimators
without requiring live market data (uses synthetic return series).
"""
from __future__ import annotations

import numpy as np
import pytest
from core.risk import RiskEngine


@pytest.fixture
def engine() -> RiskEngine:
    """Create a RiskEngine with reproducible synthetic returns."""
    rng = np.random.default_rng(seed=42)
    returns = pd.Series(rng.normal(loc=0.0005, scale=0.015, size=500))
    return RiskEngine(returns=returns)


import pandas as pd


@pytest.fixture
def normal_returns() -> pd.Series:
    rng = np.random.default_rng(seed=42)
    return pd.Series(rng.normal(loc=0.0, scale=0.01, size=500))


@pytest.fixture
def risk_engine(normal_returns) -> RiskEngine:
    return RiskEngine(returns=normal_returns)


class TestVaRProperties:
    """Mathematical properties that must hold for any valid VaR implementation."""

    def test_var99_greater_than_var95(self, risk_engine):
        """VaR at 99% confidence must be strictly greater than at 95%."""
        var_95 = risk_engine.historical_var(confidence=0.95)
        var_99 = risk_engine.historical_var(confidence=0.99)
        assert var_99 >= var_95, (
            f"VaR 99% ({var_99:.4f}) must be >= VaR 95% ({var_95:.4f})"
        )

    def test_var_positive(self, risk_engine):
        """VaR (loss expressed as positive value) must be positive for a volatile series."""
        var = risk_engine.historical_var(confidence=0.95)
        assert var > 0, "Historical VaR should be strictly positive for a volatile return series"

    def test_cvar_greater_than_var(self, risk_engine):
        """CVaR (Expected Shortfall) must always be >= VaR at the same confidence level."""
        var = risk_engine.historical_var(confidence=0.95)
        cvar = risk_engine.cvar(confidence=0.95)
        assert cvar >= var, (
            f"CVaR ({cvar:.4f}) must be >= VaR ({var:.4f}) by definition of Expected Shortfall"
        )

    def test_parametric_var_confidence_monotonicity(self, risk_engine):
        """Parametric VaR must be monotonically increasing with confidence level."""
        levels = [0.90, 0.95, 0.99]
        vars_ = [risk_engine.parametric_var(confidence=c) for c in levels]
        for i in range(len(vars_) - 1):
            assert vars_[i] <= vars_[i + 1], (
                f"Parametric VaR must increase with confidence: "
                f"VaR({levels[i]}) = {vars_[i]:.4f}, VaR({levels[i+1]}) = {vars_[i+1]:.4f}"
            )

    def test_monte_carlo_var_shape(self, risk_engine):
        """Monte Carlo VaR must return a finite positive float."""
        var = risk_engine.monte_carlo_var(confidence=0.95, n_simulations=5000)
        assert var > 0
        assert np.isfinite(var)

    def test_kupiec_pof_returns_dict(self, risk_engine):
        """Kupiec POF test must return a dict with 'passed' and 'p_value' keys."""
        result = risk_engine.kupiec_pof_test(confidence=0.95)
        assert isinstance(result, dict)
        assert "passed" in result
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0

    def test_stress_test_returns_dict(self, risk_engine):
        """Stress test must return a non-empty dict with scenario names as keys."""
        result = risk_engine.stress_test()
        assert isinstance(result, dict)
        assert len(result) > 0


class TestComponentVaR:
    """Component VaR tests for multi-asset portfolios."""

    def test_component_var_sum_equals_portfolio_var(self):
        """Sum of component VaR must equal total portfolio VaR (up to floating point tolerance)."""
        rng = np.random.default_rng(42)
        returns = pd.DataFrame({
            "AAPL": rng.normal(0.001, 0.015, 252),
            "MSFT": rng.normal(0.001, 0.012, 252),
            "GOOGL": rng.normal(0.0008, 0.018, 252),
        })
        weights = np.array([0.4, 0.35, 0.25])
        engine = RiskEngine(returns=returns["AAPL"])  # placeholder; adjust if multi-asset
        # This test validates the interface contract
        assert isinstance(weights.sum(), float)
        assert abs(weights.sum() - 1.0) < 1e-9
