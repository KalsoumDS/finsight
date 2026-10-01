# FinSight — Quantitative Risk & Alpha Engine

> Professional quantitative finance platform combining risk management, ML-driven alpha signals and rigorous backtesting on real market data.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)](https://streamlit.io)
[![Live Demo](https://img.shields.io/badge/Demo-Live-brightgreen)](https://finsight-signals.streamlit.app/)

**Live application:** https://finsight-signals.streamlit.app/

---

## Overview

FinSight is a full quantitative research platform built around four modules.

**Risk Engine** — Computes Value at Risk using three industry-standard methods (Historical, Parametric with Jarque-Bera normality testing, Monte Carlo GBM) plus CVaR, Component VaR, and historical stress tests on real crisis scenarios (2008, COVID-19, SVB 2023). VaR models are validated using the Kupiec POF statistical test.

**ML Alpha Signals** — Generates directional trading signals (Up/Down/Neutral) from 20+ engineered technical features. Two model options: XGBoost with 5-fold TimeSeriesSplit walk-forward validation (no data leakage), and LSTM PyTorch with class weighting and gradient clipping.

**Backtesting Engine** — Simulates strategy performance on historical data with realistic transaction costs (10 bps) and slippage (5 bps). Reports Sharpe ratio, Sortino ratio, Calmar ratio, Maximum Drawdown, Win Rate and Profit Factor vs. Buy & Hold benchmark.

**Market Data** — Live data via yfinance (equities, ETFs, crypto, FX). Portfolio-level statistics with correlation matrix and equal-weight performance attribution.

---

## Architecture

```
finsight/
├── core/
│   ├── data.py          # yfinance pipeline + 20+ technical features
│   ├── risk.py          # VaR (3 methods) + CVaR + Kupiec test + stress testing
│   ├── ml_signals.py    # XGBoost + LSTM training, evaluation, walk-forward CV
│   └── backtest.py      # Strategy simulation + performance metrics
├── app.py               # Streamlit multi-page application
└── requirements.txt
```

---

## Performance Summary

| Module | Method | Key Output |
|--------|--------|------------|
| Risk Engine | Historical, Parametric, Monte Carlo | VaR 95%/99%, CVaR, Component VaR |
| Backtesting | Walk-forward simulation | Sharpe, Sortino, Calmar, Max Drawdown |
| ML Signals | XGBoost / LSTM | Direction accuracy, F1-macro per class |
| Stress Testing | 2008, COVID-19, SVB 2023 | Portfolio loss % per scenario |

---

## Installation

```bash
git clone https://github.com/KalsoumDS/finsight.git
cd finsight
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Technologies

- Python 3.10+, PyTorch, XGBoost, yfinance
- Streamlit, Plotly, SciPy, NumPy, pandas

---

## Author

Oumou Kaltoum Sall — Data Scientist & ML Engineer  
[Portfolio](https://luxury-sunshine-073627.netlify.app) · [LinkedIn](https://linkedin.com/in/oumou-kaltoum-sall)
