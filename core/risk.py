"""
FinSight — Risk Engine
Calculs de risque quantitatif :
  - VaR (historique, paramétrique, Monte Carlo)
  - CVaR / Expected Shortfall
  - Stress Testing sur scénarios historiques réels
  - Greeks approximatifs
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional


class RiskEngine:
    """
    Moteur de calcul de risque quantitatif.
    Implémente les 3 méthodes standard de VaR utilisées
    en finance institutionnelle.
    """

    def __init__(self, confidence: float = 0.95, horizon: int = 1):
        """
        confidence : niveau de confiance (ex: 0.95 = VaR 95%)
        horizon    : horizon temporel en jours
        """
        assert 0 < confidence < 1, "Confiance doit être entre 0 et 1"
        self.confidence = confidence
        self.horizon = horizon
        self.alpha = 1 - confidence  # quantile bas

    # ── 1. VaR Historique ────────────────────────────────────────────────────
    def var_historical(
        self,
        returns: pd.Series,
        portfolio_value: float = 100_000,
    ) -> Dict:
        """
        VaR historique (non-paramétrique) :
        Trie les rendements passés et prend le quantile alpha.
        Avantage : capture les queues épaisses réelles.
        """
        sorted_ret = returns.sort_values()
        var_pct = float(np.percentile(returns, self.alpha * 100))
        var_abs = abs(var_pct) * portfolio_value * np.sqrt(self.horizon)
        cvar_pct = float(sorted_ret[sorted_ret <= var_pct].mean())
        cvar_abs = abs(cvar_pct) * portfolio_value * np.sqrt(self.horizon)

        return {
            "method": "Historique",
            "confidence": self.confidence,
            "horizon_days": self.horizon,
            "var_pct": var_pct,
            "var_abs": var_abs,
            "cvar_pct": cvar_pct,
            "cvar_abs": cvar_abs,
            "n_observations": len(returns),
        }

    # ── 2. VaR Paramétrique (Normal) ─────────────────────────────────────────
    def var_parametric(
        self,
        returns: pd.Series,
        portfolio_value: float = 100_000,
        distribution: str = "normal",
    ) -> Dict:
        """
        VaR paramétrique :
        Suppose une distribution normale (ou t de Student) des rendements.
        Méthode analytique — rapide mais hypothèse forte.
        """
        mu = returns.mean()
        sigma = returns.std()

        if distribution == "normal":
            z = stats.norm.ppf(self.alpha)
            var_pct = mu + z * sigma
        elif distribution == "t":
            # Ajustement t de Student (meilleures queues)
            df_t, loc_t, scale_t = stats.t.fit(returns)
            z = stats.t.ppf(self.alpha, df=df_t)
            var_pct = loc_t + z * scale_t
        else:
            raise ValueError(f"Distribution inconnue : {distribution}")

        var_abs = abs(var_pct) * portfolio_value * np.sqrt(self.horizon)

        # CVaR analytique (formule fermée pour normale)
        if distribution == "normal":
            cvar_pct = mu - sigma * stats.norm.pdf(z) / self.alpha
        else:
            cvar_pct = var_pct * 1.2  # approximation pour t

        cvar_abs = abs(cvar_pct) * portfolio_value * np.sqrt(self.horizon)

        # Tests de normalité
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        skewness = float(stats.skew(returns))
        kurtosis = float(stats.kurtosis(returns))

        return {
            "method": f"Paramétrique ({distribution})",
            "confidence": self.confidence,
            "horizon_days": self.horizon,
            "var_pct": float(var_pct),
            "var_abs": var_abs,
            "cvar_pct": float(cvar_pct),
            "cvar_abs": cvar_abs,
            "mu_daily": float(mu),
            "sigma_daily": float(sigma),
            "skewness": skewness,
            "kurtosis": kurtosis,
            "jarque_bera_pvalue": float(jb_pvalue),
            "normality_ok": jb_pvalue > 0.05,
        }

    # ── 3. VaR Monte Carlo ───────────────────────────────────────────────────
    def var_monte_carlo(
        self,
        returns: pd.Series,
        portfolio_value: float = 100_000,
        n_simulations: int = 10_000,
        seed: int = 42,
    ) -> Dict:
        """
        VaR Monte Carlo :
        Simule n_simulations scénarios de rendements futurs
        à partir des paramètres historiques (GBM avec drift et vol).
        """
        np.random.seed(seed)
        mu = returns.mean()
        sigma = returns.std()

        # Simulation GBM (Geometric Brownian Motion)
        simulated = np.random.normal(
            mu * self.horizon,
            sigma * np.sqrt(self.horizon),
            n_simulations
        )

        var_pct = float(np.percentile(simulated, self.alpha * 100))
        var_abs = abs(var_pct) * portfolio_value
        cvar_pct = float(simulated[simulated <= var_pct].mean())
        cvar_abs = abs(cvar_pct) * portfolio_value

        # Intervalle de confiance sur la VaR (bootstrap)
        bootstrap_vars = []
        for _ in range(200):
            sample = np.random.choice(simulated, size=len(simulated), replace=True)
            bootstrap_vars.append(np.percentile(sample, self.alpha * 100))
        ci_low = float(np.percentile(bootstrap_vars, 2.5))
        ci_high = float(np.percentile(bootstrap_vars, 97.5))

        return {
            "method": "Monte Carlo (GBM)",
            "confidence": self.confidence,
            "horizon_days": self.horizon,
            "n_simulations": n_simulations,
            "var_pct": var_pct,
            "var_abs": var_abs,
            "cvar_pct": cvar_pct,
            "cvar_abs": cvar_abs,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "simulated_returns": simulated,
        }

    # ── 4. Comparaison des 3 méthodes ────────────────────────────────────────
    def compare_methods(
        self,
        returns: pd.Series,
        portfolio_value: float = 100_000,
        n_mc: int = 10_000,
    ) -> pd.DataFrame:
        """
        Compare les 3 méthodes de VaR en un seul DataFrame.
        """
        hist = self.var_historical(returns, portfolio_value)
        param = self.var_parametric(returns, portfolio_value)
        mc = self.var_monte_carlo(returns, portfolio_value, n_mc)

        rows = []
        for result in [hist, param, mc]:
            rows.append({
                "Méthode": result["method"],
                "VaR (%)": f"{result['var_pct']*100:.3f}%",
                "VaR ($)": f"${result['var_abs']:,.0f}",
                "CVaR (%)": f"{result['cvar_pct']*100:.3f}%",
                "CVaR ($)": f"${result['cvar_abs']:,.0f}",
            })

        return pd.DataFrame(rows).set_index("Méthode")

    # ── 5. Stress Testing ────────────────────────────────────────────────────
    def stress_test(
        self,
        stress_returns: pd.DataFrame,
        weights: np.ndarray,
        portfolio_value: float = 100_000,
        scenario_name: str = "Scénario",
    ) -> Dict:
        """
        Applique un scénario de stress sur un portefeuille.
        stress_returns : rendements pendant la période de stress
        weights        : poids du portefeuille
        """
        port_returns = stress_returns @ weights
        total_return = float((1 + port_returns).prod() - 1)
        max_drawdown = float(
            ((1 + port_returns).cumprod() /
             (1 + port_returns).cumprod().cummax() - 1).min()
        )
        worst_day = float(port_returns.min())
        best_day = float(port_returns.max())
        volatility = float(port_returns.std() * np.sqrt(252))
        loss_abs = total_return * portfolio_value

        return {
            "scenario": scenario_name,
            "total_return_pct": total_return * 100,
            "total_loss_abs": loss_abs,
            "max_drawdown_pct": max_drawdown * 100,
            "worst_day_pct": worst_day * 100,
            "best_day_pct": best_day * 100,
            "annualized_vol_pct": volatility * 100,
            "n_days": len(port_returns),
            "port_returns": port_returns,
        }

    # ── 6. Backtesting VaR (Kupiec Test) ─────────────────────────────────────
    def backtest_var(
        self,
        returns: pd.Series,
        portfolio_value: float = 100_000,
        window: int = 252,
    ) -> Dict:
        """
        Backtesting de la VaR avec rolling window.
        Kupiec POF test pour valider le modèle.
        Violations = jours où la perte réelle > VaR estimée.
        """
        violations = []
        var_series = []

        for i in range(window, len(returns)):
            train = returns.iloc[i - window:i]
            var_pct = np.percentile(train, self.alpha * 100)
            var_series.append(var_pct)
            actual = returns.iloc[i]
            violations.append(1 if actual < var_pct else 0)

        violations = np.array(violations)
        n_violations = int(violations.sum())
        n_total = len(violations)
        violation_rate = n_violations / n_total

        # Kupiec POF Test
        expected_rate = self.alpha
        if n_violations > 0 and n_violations < n_total:
            lr_stat = -2 * (
                n_violations * np.log(expected_rate / violation_rate) +
                (n_total - n_violations) * np.log((1 - expected_rate) / (1 - violation_rate))
            )
            kupiec_pvalue = float(1 - stats.chi2.cdf(lr_stat, df=1))
        else:
            kupiec_pvalue = 0.0

        return {
            "n_violations": n_violations,
            "n_total": n_total,
            "violation_rate": violation_rate,
            "expected_rate": expected_rate,
            "kupiec_pvalue": kupiec_pvalue,
            "model_valid": kupiec_pvalue > 0.05,
            "var_series": pd.Series(var_series, index=returns.index[window:]),
            "violation_series": pd.Series(violations, index=returns.index[window:]),
        }

    # ── 7. Portfolio VaR (multi-actifs) ──────────────────────────────────────
    def portfolio_var(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        portfolio_value: float = 100_000,
        method: str = "historical",
    ) -> Dict:
        """
        VaR d'un portefeuille multi-actifs.
        Agrège les rendements avec les poids puis applique la méthode choisie.
        """
        port_returns = returns @ weights

        if method == "historical":
            return self.var_historical(port_returns, portfolio_value)
        elif method == "parametric":
            return self.var_parametric(port_returns, portfolio_value)
        elif method == "monte_carlo":
            # Monte Carlo multivarié (corrélations)
            mu = returns.mean().values
            cov = returns.cov().values
            np.random.seed(42)
            simulated_assets = np.random.multivariate_normal(
                mu * self.horizon,
                cov * self.horizon,
                10_000
            )
            simulated_port = simulated_assets @ weights
            var_pct = float(np.percentile(simulated_port, self.alpha * 100))
            cvar_pct = float(simulated_port[simulated_port <= var_pct].mean())
            return {
                "method": "Monte Carlo Multivarié",
                "confidence": self.confidence,
                "var_pct": var_pct,
                "var_abs": abs(var_pct) * portfolio_value,
                "cvar_pct": cvar_pct,
                "cvar_abs": abs(cvar_pct) * portfolio_value,
                "simulated_returns": simulated_port,
            }
        else:
            raise ValueError(f"Méthode inconnue : {method}")

    # ── 8. Marginal & Component VaR ──────────────────────────────────────────
    def component_var(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        portfolio_value: float = 100_000,
    ) -> pd.DataFrame:
        """
        Décompose la VaR du portefeuille par actif.
        Component VaR = contribution marginale de chaque actif au risque total.
        """
        port_returns = returns @ weights
        var_port = np.percentile(port_returns, self.alpha * 100)

        component_vars = {}
        cov = returns.cov().values
        port_var_cov = float(weights @ cov @ weights)

        for i, col in enumerate(returns.columns):
            # Covariance de l'actif avec le portefeuille
            cov_with_port = float(cov[i] @ weights)
            # Beta de l'actif
            beta_i = cov_with_port / (port_var_cov + 1e-9)
            # Component VaR
            comp_var_pct = beta_i * var_port * weights[i]
            component_vars[col] = {
                "weight": float(weights[i] * 100),
                "component_var_pct": float(comp_var_pct * 100),
                "component_var_abs": float(abs(comp_var_pct) * portfolio_value),
                "beta": float(beta_i),
            }

        df = pd.DataFrame(component_vars).T
        df.index.name = "Actif"
        return df
