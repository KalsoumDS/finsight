"""
FinSight — Data Pipeline
Récupère et prépare les données de marché réelles.
Sources : yfinance (prix) + calculs de features techniques.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta


# ── Univers d'actifs disponibles ─────────────────────────────────────────────
ASSET_UNIVERSE = {
    "Actions US": {
        "AAPL":  "Apple",
        "MSFT":  "Microsoft",
        "GOOGL": "Alphabet",
        "AMZN":  "Amazon",
        "NVDA":  "NVIDIA",
        "JPM":   "JPMorgan",
        "GS":    "Goldman Sachs",
        "BAC":   "Bank of America",
    },
    "ETFs & Indices": {
        "SPY":  "S&P 500 ETF",
        "QQQ":  "NASDAQ ETF",
        "VTI":  "Total Market ETF",
        "GLD":  "Gold ETF",
        "TLT":  "20Y Treasury ETF",
        "VIX":  "Volatility Index",
    },
    "Crypto": {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
    },
    "FX": {
        "EURUSD=X": "EUR/USD",
        "GBPUSD=X": "GBP/USD",
    },
}

# Scénarios de stress historiques réels
STRESS_SCENARIOS = {
    "Crise 2008 (Lehman)":      ("2008-09-01", "2009-03-31"),
    "Flash Crash 2010":         ("2010-05-01", "2010-06-30"),
    "COVID Crash 2020":         ("2020-02-01", "2020-04-30"),
    "Correction Tech 2022":     ("2022-01-01", "2022-12-31"),
    "SVB Crisis 2023":          ("2023-03-01", "2023-04-30"),
}


class MarketDataLoader:
    """
    Télécharge et met en cache les données de marché via yfinance.
    Calcule les features techniques nécessaires au Risk Engine et au ML.
    """

    def __init__(self, cache: bool = True):
        self._cache: Dict[str, pd.DataFrame] = {}
        self.use_cache = cache

    def get_prices(
        self,
        tickers: List[str],
        start: str,
        end: str = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Télécharge les prix ajustés (Close) pour une liste de tickers.
        Retourne un DataFrame wide : index=date, colonnes=tickers.
        """
        end = end or datetime.today().strftime("%Y-%m-%d")
        key = f"{'_'.join(sorted(tickers))}_{start}_{end}_{interval}"

        if self.use_cache and key in self._cache:
            return self._cache[key]

        raw = yf.download(
            tickers, start=start, end=end,
            interval=interval, auto_adjust=True,
            progress=False, threads=True
        )

        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].dropna(how="all")
        else:
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

        prices = prices.ffill().dropna()

        if self.use_cache:
            self._cache[key] = prices

        return prices

    def get_returns(
        self,
        prices: pd.DataFrame,
        method: str = "log",
    ) -> pd.DataFrame:
        """
        Calcule les rendements journaliers.
        method : 'log' (log-returns) ou 'simple' (rendements simples)
        """
        if method == "log":
            return np.log(prices / prices.shift(1)).dropna()
        else:
            return prices.pct_change().dropna()

    def compute_features(self, prices: pd.Series) -> pd.DataFrame:
        """
        Feature engineering complet sur une série de prix.
        Utilisé pour l'entraînement des modèles ML.

        Features calculées :
        - Rendements (1j, 5j, 10j, 21j)
        - Volatilité réalisée (5j, 21j)
        - RSI (14 périodes)
        - MACD + Signal
        - Bollinger Bands (20j, 2σ)
        - Volume ratio (si disponible)
        - Momentum (ROC)
        """
        df = pd.DataFrame(index=prices.index)
        df["price"] = prices

        # Rendements multi-horizons
        for lag in [1, 5, 10, 21]:
            df[f"ret_{lag}d"] = np.log(prices / prices.shift(lag))

        # Volatilité réalisée
        log_ret = np.log(prices / prices.shift(1))
        for window in [5, 21, 63]:
            df[f"vol_{window}d"] = log_ret.rolling(window).std() * np.sqrt(252)

        # RSI (14)
        df["rsi_14"] = self._rsi(prices, 14)

        # MACD
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        sma20 = prices.rolling(20).mean()
        std20 = prices.rolling(20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_lower"] = sma20 - 2 * std20
        df["bb_pct"] = (prices - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (sma20 + 1e-9)

        # Momentum (Rate of Change)
        for period in [5, 10, 21]:
            df[f"roc_{period}"] = (prices - prices.shift(period)) / (prices.shift(period) + 1e-9)

        # Distance aux moyennes mobiles
        for ma in [20, 50, 200]:
            df[f"dist_ma{ma}"] = (prices - prices.rolling(ma).mean()) / (prices.rolling(ma).mean() + 1e-9)

        # ATR proxy (sans high/low, on utilise |ret|)
        df["atr_14"] = log_ret.abs().rolling(14).mean()

        return df.dropna()

    @staticmethod
    def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def get_portfolio_stats(
        self,
        prices: pd.DataFrame,
        weights: np.ndarray = None,
    ) -> Dict:
        """
        Calcule les statistiques d'un portefeuille.
        Si weights=None, portefeuille équipondéré.
        """
        returns = self.get_returns(prices)
        n = len(prices.columns)
        weights = weights if weights is not None else np.ones(n) / n

        port_returns = returns @ weights
        annual_return = port_returns.mean() * 252
        annual_vol = port_returns.std() * np.sqrt(252)
        sharpe = annual_return / (annual_vol + 1e-9)

        # Drawdown
        cumulative = (1 + port_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()

        # Corrélation
        corr_matrix = returns.corr()

        return {
            "annual_return": float(annual_return),
            "annual_vol": float(annual_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "port_returns": port_returns,
            "cumulative": cumulative,
            "drawdown": drawdown,
            "corr_matrix": corr_matrix,
            "weights": dict(zip(prices.columns, weights)),
        }

    def get_stress_data(
        self,
        tickers: List[str],
        scenario_name: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Récupère les données sur une période de stress historique.
        Retourne (prix, rendements) sur la période du scénario.
        """
        if scenario_name not in STRESS_SCENARIOS:
            raise ValueError(f"Scénario inconnu : {scenario_name}")

        start, end = STRESS_SCENARIOS[scenario_name]
        prices = self.get_prices(tickers, start=start, end=end)
        returns = self.get_returns(prices)
        return prices, returns
