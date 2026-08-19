# FinSight — Quantitative Risk & Alpha Platform

> Professional quantitative finance platform combining risk management, ML-driven alpha signals and rigorous backtesting on real market data.

---

## What it does

FinSight is a full quantitative research platform built around four modules:

**Risk Engine** — Computes Value at Risk using three industry-standard methods (Historical, Parametric with Jarque-Bera normality testing, Monte Carlo GBM) plus CVaR, Component VaR, and historical stress tests on real crisis scenarios (2008, COVID-19, SVB 2023). VaR models are validated using the Kupiec POF statistical test.

**ML Alpha Signals** — Generates directional trading signals (Up/Down/Neutral) from 20+ engineered technical features. Two model options: XGBoost with 5-fold TimeSeriesSplit walk-forward validation (no data leakage), and LSTM PyTorch with class weighting and gradient clipping. Evaluated with accuracy, F1-macro, confusion matrix and per-class precision/recall.

**Backtesting Engine** — Simulates strategy performance on historical data with realistic transaction costs (10 bps) and slippage (5 bps). Reports Sharpe ratio, Sortino ratio, Calmar ratio, Maximum Drawdown, Win Rate and Profit Factor vs. Buy & Hold benchmark.

**Market Data** — Live data via yfinance (equities, ETFs, crypto, FX). Portfolio-level statistics with correlation matrix and equal-weight performance attribution.

---

## Architecture

```
finsight/
├── core/
│   ├── data.py          ← yfinance pipeline + 20+ technical features
│   ├── risk.py          ← VaR (3 methods) + CVaR + Kupiec test + stress testing
│   ├── ml_signals.py    ← XGBoost + LSTM PyTorch + walk-forward validation
│   └── backtest.py      ← BacktestEngine with transaction costs + metrics
├── app.py               ← Streamlit dashboard (4 tabs)
└── requirements.txt
```

---

## Technical details

**Risk calculations:**
- Historical VaR: non-parametric, captures fat tails from real data
- Parametric VaR: Normal + Student-t distributions, analytical CVaR formula
- Monte Carlo: GBM simulation with bootstrap confidence intervals
- Kupiec POF test: chi-squared likelihood ratio test to validate VaR models

**ML pipeline:**
- Features: log-returns (1/5/10/21d), realized volatility, RSI-14, MACD, Bollinger Bands (%B, width), ROC, distance to MAs (20/50/200), ATR
- Labels: forward returns over configurable horizon, threshold-based ternary classification
- Walk-forward: TimeSeriesSplit with minimum 252-day training window — no look-ahead bias
- Class imbalance: CrossEntropyLoss with inverse-frequency weights

**Backtesting:**
- Signal lag: positions shifted by 1 day to eliminate look-ahead bias
- Costs: configurable transaction cost + slippage applied on position changes
- Metrics: Sharpe, Sortino (downside deviation), Calmar, Max Drawdown, Profit Factor

---

## Stack

Python · PyTorch · XGBoost · Scikit-learn · yfinance · Streamlit · Plotly · SciPy

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

**Oumou Kaltoum Sall** — Data Scientist & ML Engineer  
[GitHub](https://github.com/KalsoumDS) · [s.sall@mundiapolis.ma](mailto:s.sall@mundiapolis.ma)
