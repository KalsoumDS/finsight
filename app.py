"""
FinSight — Quantitative Risk & Alpha Platform
Dashboard Streamlit : Risk Engine + ML Signals + Backtesting
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="FinSight — Quant Risk & Alpha",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size:1.8rem; font-weight:900; }
.kpi-positive { color: #00c864; }
.kpi-negative { color: #ff4444; }
.section-header {
    font-size:1.1rem; font-weight:700; letter-spacing:.05em;
    text-transform:uppercase; color:#888; margin:24px 0 12px 0;
    border-bottom:1px solid #222; padding-bottom:8px;
}
</style>
""", unsafe_allow_html=True)

from core.data import MarketDataLoader, ASSET_UNIVERSE, STRESS_SCENARIOS
from core.risk import RiskEngine
from core.backtest import BacktestEngine
from core.ml_signals import XGBoostSignalGenerator, LSTMSignalGenerator

# ── SESSION STATE ─────────────────────────────────────────────────────────────
def init():
    defaults = {
        "prices": None, "returns": None, "tickers": [],
        "risk_results": {}, "ml_xgb": None, "ml_lstm": None,
        "backtest_result": None, "features": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init()

loader = MarketDataLoader()

# ── HELPERS PLOTS ─────────────────────────────────────────────────────────────
def plot_prices(prices: pd.DataFrame, title="Prix ajustés") -> go.Figure:
    fig = go.Figure()
    for col in prices.columns:
        norm = prices[col] / prices[col].iloc[0] * 100
        fig.add_trace(go.Scatter(x=prices.index, y=norm, mode='lines',
                                  name=col, line=dict(width=1.5)))
    fig.update_layout(template='plotly_dark', height=380, title=title,
                      yaxis_title="Performance base 100",
                      margin=dict(l=0,r=0,t=40,b=0),
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_returns_dist(returns: pd.Series, var_hist: float,
                       var_param: float, title="Distribution des rendements") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns*100, nbinsx=80,
                                marker_color='#7c6af7', opacity=0.75, name='Rendements'))
    for val, name, color in [(var_hist*100, 'VaR Historique', '#ff6b6b'),
                              (var_param*100, 'VaR Paramétrique', '#ffa500')]:
        fig.add_vline(x=val, line_dash='dash', line_color=color,
                      annotation_text=name, annotation_position='top left')
    fig.update_layout(template='plotly_dark', height=320, title=title,
                      xaxis_title='Rendement journalier (%)', yaxis_title='Fréquence',
                      margin=dict(l=0,r=0,t=40,b=0),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_mc_simulation(simulated: np.ndarray, var_pct: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=simulated*100, nbinsx=80,
                                marker_color='#4ecdc4', opacity=0.75, name='Simulations'))
    fig.add_vline(x=var_pct*100, line_dash='dash', line_color='red',
                  annotation_text=f'VaR = {var_pct*100:.3f}%')
    fig.update_layout(template='plotly_dark', height=300,
                      title=f'Monte Carlo — {len(simulated):,} simulations',
                      xaxis_title='Rendement simulé (%)',
                      margin=dict(l=0,r=0,t=40,b=0),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_equity_curve(result: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result['strategy_equity'].index,
        y=result['strategy_equity'].values,
        mode='lines', name='Stratégie ML',
        line=dict(color='#c8ff00', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=result['benchmark_equity'].index,
        y=result['benchmark_equity'].values,
        mode='lines', name='Buy & Hold',
        line=dict(color='#888', width=1.5, dash='dash')
    ))
    fig.update_layout(template='plotly_dark', height=380,
                      title='Courbe de capital — Stratégie vs Buy & Hold',
                      yaxis_title='Capital ($)',
                      margin=dict(l=0,r=0,t=40,b=0),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_drawdown(result: dict) -> go.Figure:
    fig = go.Figure()
    dd_strat = result['strategy_metrics']['drawdown']
    dd_bench = result['benchmark_metrics']['drawdown']
    fig.add_trace(go.Scatter(x=dd_strat.index, y=dd_strat.values*100,
                              fill='tozeroy', name='Stratégie',
                              line=dict(color='#ff6b6b', width=1.5),
                              fillcolor='rgba(255,107,107,0.15)'))
    fig.add_trace(go.Scatter(x=dd_bench.index, y=dd_bench.values*100,
                              fill='tozeroy', name='Buy & Hold',
                              line=dict(color='#888', width=1, dash='dash'),
                              fillcolor='rgba(136,136,136,0.08)'))
    fig.update_layout(template='plotly_dark', height=280,
                      title='Drawdown (%)',
                      yaxis_title='Drawdown (%)',
                      margin=dict(l=0,r=0,t=40,b=0),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_confusion_matrix(cm: np.ndarray, labels=["Down","Neutral","Up"]) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale='Viridis', text=cm,
        texttemplate='%{text}', textfont={"size":14}
    ))
    fig.update_layout(template='plotly_dark', height=320,
                      title='Matrice de confusion (validation)',
                      xaxis_title='Prédit', yaxis_title='Réel',
                      margin=dict(l=0,r=0,t=40,b=0),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_feature_importance(fi: dict, top_n=15) -> go.Figure:
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [x[0] for x in sorted_fi]
    values = [x[1] for x in sorted_fi]
    fig = go.Figure(go.Bar(x=values, y=names, orientation='h',
                            marker_color='#a78bfa'))
    fig.update_layout(template='plotly_dark', height=380,
                      title=f'Top {top_n} features importantes (XGBoost)',
                      xaxis_title='Importance',
                      margin=dict(l=0,r=0,t=40,b=0),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_corr_matrix(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
        text=corr.round(2).values, texttemplate='%{text}',
        textfont={"size":10}
    ))
    fig.update_layout(template='plotly_dark', height=400,
                      title='Matrice de corrélation des rendements',
                      margin=dict(l=0,r=0,t=40,b=0),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <h1 style='font-size:2rem;font-weight:900;margin-bottom:0'>
        📈 FinSight — Quantitative Risk & Alpha Platform
    </h1>
    <p style='color:#888;margin-top:4px'>
        Risk Engine · ML Signals · Backtesting · Données réelles (yfinance)
    </p>
    """, unsafe_allow_html=True)
    st.divider()

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        st.markdown("### 📊 Univers d'actifs")
        category = st.selectbox("Catégorie", list(ASSET_UNIVERSE.keys()))
        available = ASSET_UNIVERSE[category]
        selected = st.multiselect(
            "Tickers", list(available.keys()),
            default=list(available.keys())[:3],
            format_func=lambda x: f"{x} — {available.get(x, x)}"
        )

        st.markdown("### 📅 Période")
        period = st.selectbox("Historique", ["1 an", "2 ans", "3 ans", "5 ans"], index=1)
        period_map = {"1 an": "365", "2 ans": "730", "3 ans": "1095", "5 ans": "1825"}
        from datetime import datetime, timedelta
        n_days = int(period_map[period])
        start_date = (datetime.today() - timedelta(days=n_days)).strftime("%Y-%m-%d")

        st.markdown("### 💼 Portefeuille")
        portfolio_value = st.number_input("Valeur ($)", 10_000, 10_000_000, 100_000, 10_000)
        equal_weight = st.checkbox("Pondération égale", value=True)

        st.markdown("### ⚠️ Risk Engine")
        confidence = st.slider("Niveau de confiance VaR", 0.90, 0.99, 0.95, 0.01)
        horizon = st.selectbox("Horizon (jours)", [1, 5, 10, 21], index=0)
        n_mc = st.select_slider("Simulations MC", [1000, 5000, 10000, 50000], value=10000)

        load_btn = st.button("🔄 Charger les données", type="primary", use_container_width=True)

        st.divider()
        st.markdown("""
        **Stack :** yfinance · PyTorch · XGBoost · Streamlit · Plotly

        **Auteur :** [Oumou Kaltoum Sall](https://github.com/KalsoumDS)
        """)

    # ── CHARGEMENT DONNÉES ────────────────────────────────────────────────────
    if load_btn and selected:
        with st.spinner(f"📥 Téléchargement de {selected} depuis yfinance..."):
            try:
                prices = loader.get_prices(selected, start=start_date)
                returns = loader.get_returns(prices)
                st.session_state.prices = prices
                st.session_state.returns = returns
                st.session_state.tickers = selected
                st.session_state.risk_results = {}
                st.session_state.ml_xgb = None
                st.session_state.backtest_result = None
                st.success(f"✅ {len(prices)} jours de données chargées pour {selected}")
            except Exception as e:
                st.error(f"❌ Erreur : {e}")

    if st.session_state.prices is None:
        st.info("👈 Sélectionner des actifs et cliquer sur **Charger les données**.")
        return

    prices = st.session_state.prices
    returns = st.session_state.returns
    tickers = st.session_state.tickers
    weights = np.ones(len(tickers)) / len(tickers)

    # ── ONGLETS ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Marché & Portefeuille",
        "⚠️ Risk Engine",
        "🤖 ML Alpha Signals",
        "📋 Backtesting"
    ])

    # ── TAB 1 : MARCHÉ ────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Performance des actifs</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_prices(prices), use_container_width=True)

        port_stats = loader.get_portfolio_stats(prices, weights)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📈 Rendement annualisé",
                      f"{port_stats['annual_return']*100:.2f}%",
                      delta=f"{port_stats['annual_return']*100:.2f}%")
        with col2:
            st.metric("📉 Volatilité annualisée", f"{port_stats['annual_vol']*100:.2f}%")
        with col3:
            st.metric("⚡ Sharpe Ratio", f"{port_stats['sharpe_ratio']:.3f}")
        with col4:
            st.metric("🔻 Max Drawdown", f"{port_stats['max_drawdown']*100:.2f}%")

        st.divider()
        if len(tickers) > 1:
            st.markdown('<div class="section-header">Corrélations</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_corr_matrix(port_stats['corr_matrix']), use_container_width=True)

    # ── TAB 2 : RISK ENGINE ───────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Value at Risk — 3 méthodes</div>', unsafe_allow_html=True)

        # Actif principal pour VaR univariée
        primary = st.selectbox("Actif principal", tickers, key="risk_ticker")
        ret_primary = returns[primary] if primary in returns.columns else returns.iloc[:, 0]

        risk = RiskEngine(confidence=confidence, horizon=horizon)

        compute_risk = st.button("⚡ Calculer le Risk", type="primary")
        if compute_risk:
            with st.spinner("Calcul en cours..."):
                hist = risk.var_historical(ret_primary, portfolio_value)
                param = risk.var_parametric(ret_primary, portfolio_value)
                mc = risk.var_monte_carlo(ret_primary, portfolio_value, n_mc)
                port_var = risk.portfolio_var(returns, weights, portfolio_value, "historical")
                comp_var = risk.component_var(returns, weights, portfolio_value)
                backtest_var = risk.backtest_var(ret_primary, portfolio_value)
                st.session_state.risk_results = {
                    "hist": hist, "param": param, "mc": mc,
                    "port_var": port_var, "comp_var": comp_var,
                    "backtest_var": backtest_var,
                    "comparison": risk.compare_methods(ret_primary, portfolio_value, n_mc)
                }

        if st.session_state.risk_results:
            r = st.session_state.risk_results
            hist, param, mc = r["hist"], r["param"], r["mc"]

            # Tableau comparatif
            st.markdown("#### Comparaison des 3 méthodes")
            st.dataframe(r["comparison"], use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(
                    plot_returns_dist(ret_primary, hist["var_pct"], param["var_pct"]),
                    use_container_width=True
                )
            with col_b:
                st.plotly_chart(
                    plot_mc_simulation(mc["simulated_returns"], mc["var_pct"]),
                    use_container_width=True
                )

            # Métriques clés
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("VaR Historique", f"${hist['var_abs']:,.0f}",
                          f"{hist['var_pct']*100:.3f}%")
            with col2:
                st.metric("CVaR Historique", f"${hist['cvar_abs']:,.0f}",
                          f"{hist['cvar_pct']*100:.3f}%")
            with col3:
                st.metric("VaR Monte Carlo", f"${mc['var_abs']:,.0f}",
                          f"{mc['var_pct']*100:.3f}%")
            with col4:
                bv = r["backtest_var"]
                status = "✅ Valide" if bv["model_valid"] else "⚠️ Invalide"
                st.metric("Kupiec Test", status,
                          f"{bv['n_violations']} violations ({bv['violation_rate']*100:.1f}%)")

            # Normalité
            st.info(f"📊 **Test de normalité (Jarque-Bera)** — p-value: "
                    f"`{param['jarque_bera_pvalue']:.4f}` — "
                    f"Skewness: `{param['skewness']:.3f}` — "
                    f"Kurtosis: `{param['kurtosis']:.3f}` — "
                    f"{'Distribution normale ✅' if param['normality_ok'] else 'Non-normale ⚠️ (queues épaisses)'}")

            # Component VaR
            if len(tickers) > 1:
                st.markdown("#### Component VaR — Contribution par actif")
                st.dataframe(r["comp_var"].round(4), use_container_width=True)

            # Stress Testing
            st.divider()
            st.markdown('<div class="section-header">Stress Testing — Scénarios historiques réels</div>',
                        unsafe_allow_html=True)
            scenario = st.selectbox("Scénario", list(STRESS_SCENARIOS.keys()))
            if st.button("🔥 Lancer le stress test"):
                with st.spinner(f"Simulation : {scenario}..."):
                    try:
                        _, stress_ret = loader.get_stress_data(tickers, scenario)
                        stress_ret = stress_ret.reindex(columns=tickers).fillna(0)
                        stress_result = risk.stress_test(
                            stress_ret, weights, portfolio_value, scenario
                        )
                        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                        with col_s1:
                            st.metric("Perte totale",
                                      f"${stress_result['total_loss_abs']:,.0f}",
                                      f"{stress_result['total_return_pct']:.2f}%")
                        with col_s2:
                            st.metric("Max Drawdown",
                                      f"{stress_result['max_drawdown_pct']:.2f}%")
                        with col_s3:
                            st.metric("Pire journée",
                                      f"{stress_result['worst_day_pct']:.2f}%")
                        with col_s4:
                            st.metric("Volatilité annualisée",
                                      f"{stress_result['annualized_vol_pct']:.2f}%")
                    except Exception as e:
                        st.error(f"❌ {e}")

    # ── TAB 3 : ML SIGNALS ───────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">ML Alpha Signal Generation</div>',
                    unsafe_allow_html=True)

        ml_ticker = st.selectbox("Actif cible", tickers, key="ml_ticker")
        ret_ml = returns[ml_ticker] if ml_ticker in returns.columns else returns.iloc[:, 0]
        price_ml = prices[ml_ticker] if ml_ticker in prices.columns else prices.iloc[:, 0]

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            fwd_horizon = st.selectbox("Horizon de prédiction (jours)", [1, 3, 5, 10], index=2)
        with col_p2:
            threshold = st.slider("Seuil signal (%)", 0.1, 2.0, 0.5, 0.1) / 100
        with col_p3:
            model_choice = st.selectbox("Modèle", ["XGBoost", "LSTM (PyTorch)"])

        train_btn = st.button(f"🚀 Entraîner {model_choice}", type="primary")

        if train_btn:
            with st.spinner(f"🔧 Feature engineering + entraînement {model_choice}..."):
                features_df = loader.compute_features(price_ml)

                if model_choice == "XGBoost":
                    model = XGBoostSignalGenerator(
                        forward_horizon=fwd_horizon,
                        threshold=threshold,
                        n_splits=5,
                    )
                    try:
                        results = model.fit(features_df, ret_ml)
                        st.session_state.ml_xgb = model
                        st.session_state.features = features_df
                        st.success(f"✅ XGBoost entraîné — Accuracy: {results['accuracy']*100:.1f}%"
                                   f" | F1 macro: {results['f1_macro']:.3f}")
                    except Exception as e:
                        st.error(f"❌ {e}")

                else:  # LSTM
                    epochs = st.session_state.get("lstm_epochs", 30)
                    model = LSTMSignalGenerator(
                        forward_horizon=fwd_horizon,
                        threshold=threshold,
                        n_epochs=30,
                        sequence_length=20,
                    )
                    progress_bar = st.progress(0)
                    loss_ph = st.empty()

                    def cb(epoch, total, loss, val_acc):
                        progress_bar.progress(epoch / total)
                        loss_ph.markdown(
                            f"Epoch {epoch}/{total} — Loss: `{loss:.4f}` — Val Acc: `{val_acc*100:.1f}%`"
                        )

                    try:
                        results = model.fit(features_df, ret_ml, progress_callback=cb)
                        st.session_state.ml_xgb = model
                        st.session_state.features = features_df
                        st.success(f"✅ LSTM entraîné — Accuracy: {results['accuracy']*100:.1f}%"
                                   f" | F1 macro: {results['f1_macro']:.3f}")
                    except Exception as e:
                        st.error(f"❌ {e}")

        # Afficher résultats
        ml_model = st.session_state.ml_xgb
        if ml_model and hasattr(ml_model, 'results_') and ml_model.results_:
            res = ml_model.results_
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{res['accuracy']*100:.2f}%")
            with col2:
                st.metric("F1 Macro", f"{res['f1_macro']:.4f}")
            with col3:
                st.metric("F1 Weighted", f"{res['f1_weighted']:.4f}")
            with col4:
                n = res.get('n_samples') or res.get('n_val', '?')
                st.metric("Échantillons test", str(n))

            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(
                    plot_confusion_matrix(res['confusion_matrix']),
                    use_container_width=True
                )
            with col_b:
                if "feature_importance" in res:
                    st.plotly_chart(
                        plot_feature_importance(res['feature_importance']),
                        use_container_width=True
                    )
                elif "train_losses" in res:
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(
                        y=res['train_losses'], mode='lines+markers',
                        line=dict(color='#c8ff00', width=2), name='Loss'
                    ))
                    fig_loss.update_layout(
                        template='plotly_dark', height=320,
                        title='Courbe de loss LSTM',
                        xaxis_title='Epoch', yaxis_title='Cross-Entropy Loss',
                        margin=dict(l=0,r=0,t=40,b=0),
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)

            # Rapport de classification
            if "classification_report" in res:
                cr = res["classification_report"]
                cr_df = pd.DataFrame({
                    "Classe": ["Down ↓", "Neutral →", "Up ↑"],
                    "Precision": [cr.get("Down",{}).get("precision",0),
                                  cr.get("Neutral",{}).get("precision",0),
                                  cr.get("Up",{}).get("precision",0)],
                    "Recall": [cr.get("Down",{}).get("recall",0),
                               cr.get("Neutral",{}).get("recall",0),
                               cr.get("Up",{}).get("recall",0)],
                    "F1": [cr.get("Down",{}).get("f1-score",0),
                           cr.get("Neutral",{}).get("f1-score",0),
                           cr.get("Up",{}).get("f1-score",0)],
                }).set_index("Classe")
                st.dataframe(cr_df.round(4), use_container_width=True)

if __name__ == "__main__":
    main()
