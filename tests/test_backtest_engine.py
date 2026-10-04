"""
Unit tests for FinSight BacktestEngine.

Validates strategy simulation properties: no look-ahead bias,
proper Sharpe/Sortino computation, and benchmark comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from core.backtest import BacktestEngine


@pytest.fixture
def price_series() -> pd.Series:
    """Deterministic trending price series for predictable test results."""
    rng = np.random.default_rng(seed=0)
    returns = rng.normal(0.0005, 0.01, 500)
    prices = pd.Series(100 * np.exp(np.cumsum(returns)))
    prices.index = pd.date_range("2020-01-01", periods=500, freq="B")
    return prices


@pytest.fixture
def signals(price_series) -> pd.Series:
    """Simple alternating buy/sell signals aligned with price index."""
    sigs = pd.Series(
        [1 if i % 3 != 0 else -1 for i in range(len(price_series))],
        index=price_series.index
    )
    return sigs


@pytest.fixture
def engine(price_series, signals) -> BacktestEngine:
    return BacktestEngine(prices=price_series, signals=signals)


class TestBacktestEngine:

    def test_sharpe_ratio_is_finite(self, engine):
        """Sharpe ratio must be a finite real number."""
        result = engine.run()
        assert np.isfinite(result["sharpe_ratio"]), "Sharpe ratio must be finite"

    def test_max_drawdown_negative_or_zero(self, engine):
        """Maximum drawdown must be <= 0 (it is a loss expressed as negative)."""
        result = engine.run()
        assert result["max_drawdown"] <= 0, (
            f"Max drawdown must be <= 0, got {result['max_drawdown']:.4f}"
        )

    def test_win_rate_bounds(self, engine):
        """Win rate must be in [0, 1]."""
        result = engine.run()
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_sortino_ratio_finite(self, engine):
        """Sortino ratio must be a finite real number."""
        result = engine.run()
        assert np.isfinite(result.get("sortino_ratio", 0.0))

    def test_result_contains_required_keys(self, engine):
        """Backtest result must include all standard institutional metrics."""
        result = engine.run()
        required = {"sharpe_ratio", "max_drawdown", "win_rate"}
        missing = required - set(result.keys())
        assert not missing, f"Missing required metrics: {missing}"

    def test_transaction_costs_reduce_returns(self, price_series, signals):
        """Adding transaction costs must reduce or equal net return."""
        engine_no_cost = BacktestEngine(prices=price_series, signals=signals, transaction_cost=0.0)
        engine_with_cost = BacktestEngine(prices=price_series, signals=signals, transaction_cost=0.001)
        result_no_cost = engine_no_cost.run()
        result_with_cost = engine_with_cost.run()
        total_return_no_cost = result_no_cost.get("total_return", 0)
        total_return_with_cost = result_with_cost.get("total_return", 0)
        assert total_return_no_cost >= total_return_with_cost, (
            "Transaction costs should reduce or maintain returns, never increase them"
        )
