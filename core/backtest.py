"""
FinSight — Backtesting Engine
Évaluation des stratégies de trading avec métriques institutionnelles.
Pas de look-ahead bias, walk-forward propre.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class BacktestEngine:
    """
    Moteur de backtesting rigoureux.
    Applique des signaux ML sur des données historiques
    et calcule les métriques de performance.
    """

    def __init__(
        self,
        transaction_cost: float = 0.001,   # 10 bps par trade
        slippage: float = 0.0005,          # 5 bps de slippage
        risk_free_rate: float = 0.04,      # Taux sans risque annualisé
    ):
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate

    def run(
        self,
        returns: pd.Series,
        signals: pd.Series,
        initial_capital: float = 100_000,
    ) -> Dict:
        """
        Exécute un backtest sur des signaux discrets.
        signals : -1 (short), 0 (flat), +1 (long)
        """
        common_idx = returns.index.intersection(signals.index)
        ret = returns.loc[common_idx]
        sig = signals.loc[common_idx].shift(1).fillna(0)  # décalage pour éviter look-ahead

        # Coûts de transaction sur les changements de position
        position_changes = sig.diff().abs()
        total_cost = self.transaction_cost + self.slippage
        costs = position_changes * total_cost

        # Rendements de la stratégie
        strat_returns = sig * ret - costs
        bench_returns = ret  # Buy & Hold

        # Courbes de capital
        strat_equity = initial_capital * (1 + strat_returns).cumprod()
        bench_equity = initial_capital * (1 + bench_returns).cumprod()

        # Métriques
        strat_metrics = self._compute_metrics(strat_returns, initial_capital)
        bench_metrics = self._compute_metrics(bench_returns, initial_capital)

        # Nombre de trades
        n_trades = int((position_changes > 0).sum())

        return {
            "strategy_returns": strat_returns,
            "benchmark_returns": bench_returns,
            "strategy_equity": strat_equity,
            "benchmark_equity": bench_equity,
            "strategy_metrics": strat_metrics,
            "benchmark_metrics": bench_metrics,
            "n_trades": n_trades,
            "total_costs": float(costs.sum()),
            "signals": sig,
        }

    def _compute_metrics(
        self,
        returns: pd.Series,
        initial_capital: float = 100_000,
    ) -> Dict:
        """
        Calcule les métriques de performance standard.
        """
        rf_daily = self.risk_free_rate / 252

        # Rendement annualisé
        n_days = len(returns)
        total_return = float((1 + returns).prod() - 1)
        annual_return = float((1 + total_return) ** (252 / max(n_days, 1)) - 1)

        # Volatilité annualisée
        annual_vol = float(returns.std() * np.sqrt(252))

        # Sharpe Ratio
        excess_returns = returns - rf_daily
        sharpe = float(excess_returns.mean() / (excess_returns.std() + 1e-9) * np.sqrt(252))

        # Sortino Ratio (downside deviation)
        downside = returns[returns < rf_daily]
        downside_std = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else 1e-9
        sortino = float((annual_return - self.risk_free_rate) / (downside_std + 1e-9))

        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = float(drawdown.min())

        # Calmar Ratio
        calmar = float(annual_return / (abs(max_dd) + 1e-9))

        # Win Rate
        positive_days = (returns > 0).sum()
        win_rate = float(positive_days / max(len(returns), 1))

        # Profit Factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = float(gains / (losses + 1e-9))

        # VaR 95%
        var_95 = float(np.percentile(returns, 5))

        return {
            "total_return_pct": total_return * 100,
            "annual_return_pct": annual_return * 100,
            "annual_vol_pct": annual_vol * 100,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown_pct": max_dd * 100,
            "win_rate_pct": win_rate * 100,
            "profit_factor": profit_factor,
            "var_95_pct": var_95 * 100,
            "final_capital": float(initial_capital * (1 + total_return)),
            "cumulative": cumulative,
            "drawdown": drawdown,
        }

    def compare_strategies(
        self,
        returns: pd.Series,
        strategies: Dict[str, pd.Series],
        initial_capital: float = 100_000,
    ) -> pd.DataFrame:
        """
        Compare plusieurs stratégies en un tableau.
        strategies : dict {nom: pd.Series de signaux}
        """
        rows = []
        for name, signals in strategies.items():
            result = self.run(returns, signals, initial_capital)
            m = result["strategy_metrics"]
            rows.append({
                "Stratégie": name,
                "Rendement annualisé": f"{m['annual_return_pct']:.2f}%",
                "Volatilité": f"{m['annual_vol_pct']:.2f}%",
                "Sharpe": f"{m['sharpe_ratio']:.3f}",
                "Sortino": f"{m['sortino_ratio']:.3f}",
                "Max Drawdown": f"{m['max_drawdown_pct']:.2f}%",
                "Win Rate": f"{m['win_rate_pct']:.1f}%",
                "Profit Factor": f"{m['profit_factor']:.3f}",
            })

        # Ajouter Buy & Hold
        bh_result = self._compute_metrics(returns, initial_capital)
        rows.append({
            "Stratégie": "Buy & Hold (benchmark)",
            "Rendement annualisé": f"{bh_result['annual_return_pct']:.2f}%",
            "Volatilité": f"{bh_result['annual_vol_pct']:.2f}%",
            "Sharpe": f"{bh_result['sharpe_ratio']:.3f}",
            "Sortino": f"{bh_result['sortino_ratio']:.3f}",
            "Max Drawdown": f"{bh_result['max_drawdown_pct']:.2f}%",
            "Win Rate": f"{bh_result['win_rate_pct']:.1f}%",
            "Profit Factor": f"{bh_result['profit_factor']:.3f}",
        })

        return pd.DataFrame(rows).set_index("Stratégie")
