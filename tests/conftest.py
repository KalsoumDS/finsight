"""
Shared pytest fixtures and configuration for FinSight test suite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_returns_500d() -> pd.Series:
    """500-day synthetic daily return series, normally distributed."""
    rng = np.random.default_rng(seed=42)
    returns = rng.normal(loc=0.0005, scale=0.015, size=500)
    index = pd.date_range("2022-01-01", periods=500, freq="B")
    return pd.Series(returns, index=index, name="synthetic")


@pytest.fixture(scope="session")
def synthetic_price_series(synthetic_returns_500d) -> pd.Series:
    """Price series derived from synthetic returns starting at 100."""
    return (1 + synthetic_returns_500d).cumprod() * 100


@pytest.fixture(scope="session")
def multi_asset_returns() -> pd.DataFrame:
    """3-asset correlated return matrix for portfolio-level tests."""
    rng = np.random.default_rng(seed=99)
    cov = np.array([
        [0.0002, 0.00008, 0.00005],
        [0.00008, 0.00015, 0.00004],
        [0.00005, 0.00004, 0.00025],
    ])
    data = rng.multivariate_normal(
        mean=[0.0005, 0.0004, 0.0006],
        cov=cov,
        size=500
    )
    index = pd.date_range("2022-01-01", periods=500, freq="B")
    return pd.DataFrame(data, index=index, columns=["AAPL", "MSFT", "GOOGL"])
