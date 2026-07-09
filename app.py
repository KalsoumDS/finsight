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

# Page config
st.set_page_config(
    page_title="FinSight | Quant Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/KalsoumDS/finsight",
        "Report a bug": "https://github.com/KalsoumDS/finsight/issues",
        "About": "FinSight\nProfessional Quantitative Risk & Alpha Platform.",
    }
)

# Custom CSS
st.markdown("""
<style>
/* Main container */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header styling */
h1 {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

h2 {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.5rem !important;
}

h3 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
}

/* Metric styling */
[data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
    font-weight: 800 !important;
}

[data-testid="stMetricDelta"] {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* Section header */
.section-header {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: #6b7280 !important;
    margin: 2rem 0 0.75rem 0 !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 2px solid #e5e7eb !important;
}

@media (prefers-color-scheme: dark) {
    .section-header {
        color: #9ca3af !important;
        border-bottom-color: #374151 !important;
    }
}

/* Info box */
.info-box {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    color: #1f2937;
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
}

.info-box h3, .info-box h4 {
    color: #1f2937;
}

.info-box p, .info-box li {
    color: inherit;
}

/* Aperçu des fonctionnalités boxes */
[data-testid="column"] > div > div {
    background-color: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    color: #1f2937 !important;
}

/* Dark mode styles */
@media (prefers-color-scheme: dark) {
    .info-box {
        background-color: #1e293b;
        border-color: #334155;
        color: #f8fafc;
    }

    .info-box h3, .info-box h4 {
        color: #e0e7ff;
    }

    [data-testid="column"] > div > div {
        background-color: #1e293b !important;
        border-color: #334155 !important;
        color: #f8fafc !important;
    }
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #f8fafc;
}

[data-testid="stSidebar"] h2 {
    color: #e0e7ff !important;
}

[data-testid="stSidebar"] h3 {
    color: #c7d2fe !important;
}

/* Success/Error messages */
div[data-testid="stSuccess"] {
    background-color: #d1fae5;
    border-left: 4px solid #10b981;
}

div[data-testid="stError"] {
    background-color: #fee2e2;
    border-left: 4px solid #ef4444;
}

div[data-testid="stWarning"] {
    background-color: #fef3c7;
    border-left: 4px solid #f59e0b;
}

div[data-testid="stInfo"] {
    background-color: #dbeafe;
    border-left: 4px solid #3b82f6;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding: 0 1.5rem;
    border-radius: 8px 8px 0 0;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# Import core modules
from core.data import MarketDataLoader, ASSET_UNIVERSE, STRESS_SCENARIOS
from core.risk import RiskEngine
from core.backtest import BacktestEngine

# Try to import ML modules with graceful fallback
try:
    from core.ml_signals import XGBoostSignalGenerator, LSTMSignalGenerator
    ML_AVAILABLE = True
except Exception as e:
    print(f"ML modules not available: {e}")
    ML_AVAILABLE = False

# ── Session State Initialization ──────────────────────────────────────────────
def init_session_state():
    defaults = {
        "prices": None,
        "returns": None,
        "tickers": [],
        "risk_results": {},
        "ml_xgb": None,
        "ml_lstm": None,
        "backtest_result": None,
        "features": None,
        "data_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()
loader = MarketDataLoader()

# ── Helper Functions ──────────────────────────────────────────────────────────
def plot_prices(prices: pd.DataFrame, title="Performance des Actifs (Base 100)") -> go.Figure:
    fig = go.Figure()
    if prices is not None and not prices.empty and len(prices.columns) > 0:
        colors = px.colors.qualitative.D3
        for i, col in enumerate(prices.columns):
            if len(prices[col].dropna()) > 0:
                norm = prices[col] / prices[col].dropna().iloc[0] * 100
                fig.add_trace(go.Scatter(
                    x=prices.index, y=norm, mode='lines',
                    name=col, line=dict(width=2.5, color=colors[i % len(colors)]),
                    hovertemplate="%{y:.1f}<extra></extra>"
                ))
    fig.update_layout(
        template='plotly_white', height=380, title=title,
        yaxis_title="Performance (Base 100)",
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig

def plot_returns_dist(returns: pd.Series, var_hist: float, var_param: float, title="Distribution des Rendements Journaliers") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns*100, nbinsx=80,
        marker_color='#667eea', opacity=0.7, name='Rendements',
        hovertemplate="%{x:.2f}%<br>%{y}<extra></extra>"
    ))
    for val, name, color in [(var_hist*100, 'VaR Historique', '#ef4444'), (var_param*100, 'VaR Paramétrique', '#f59e0b')]:
        fig.add_vline(
            x=val, line_dash='dash', line_color=color, line_width=2,
            annotation_text=name, annotation_position='top left',
            annotation_font=dict(size=12, color=color)
        )
    fig.update_layout(
        template='plotly_white', height=320, title=title,
        xaxis_title='Rendement journalier (%)', yaxis_title='Fréquence',
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

def plot_mc_simulation(simulated: np.ndarray, var_pct: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=simulated*100, nbinsx=80,
        marker_color='#10b981', opacity=0.7, name='Simulations',
        hovertemplate="%{x:.2f}%<br>%{y}<extra></extra>"
    ))
    fig.add_vline(
        x=var_pct*100, line_dash='dash', line_color='#ef4444', line_width=2,
        annotation_text=f'VaR = {var_pct*100:.3f}%', annotation_position='top left',
        annotation_font=dict(size=12, color='#ef4444')
    )
    fig.update_layout(
        template='plotly_white', height=300,
        title=f'Simulation Monte Carlo ({len(simulated):,} scénarios)',
        xaxis_title='Rendement simulé (%)',
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

def plot_equity_curve(result: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result['strategy_equity'].index,
        y=result['strategy_equity'].values,
        mode='lines', name='Stratégie ML',
        line=dict(color='#10b981', width=3),
        hovertemplate="$%{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=result['benchmark_equity'].index,
        y=result['benchmark_equity'].values,
        mode='lines', name='Buy & Hold',
        line=dict(color='#64748b', width=2, dash='dash'),
        hovertemplate="$%{y:,.0f}<extra></extra>"
    ))
    fig.update_layout(
        template='plotly_white', height=380,
        title='Courbe de Capital — Stratégie vs Buy & Hold',
        yaxis_title='Capital ($)',
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig

def plot_drawdown(result: dict) -> go.Figure:
    fig = go.Figure()
    dd_strat = result['strategy_metrics']['drawdown']
    dd_bench = result['benchmark_metrics']['drawdown']
    fig.add_trace(go.Scatter(
        x=dd_strat.index, y=dd_strat.values*100,
        fill='tozeroy', name='Stratégie',
        line=dict(color='#ef4444', width=2),
        fillcolor='rgba(239, 68, 68, 0.15)',
        hovertemplate="%{y:.1f}%<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=dd_bench.index, y=dd_bench.values*100,
        fill='tozeroy', name='Buy & Hold',
        line=dict(color='#64748b', width=1.5, dash='dash'),
        fillcolor='rgba(100, 116, 139, 0.1)',
        hovertemplate="%{y:.1f}%<extra></extra>"
    ))
    fig.update_layout(
        template='plotly_white', height=280,
        title='Drawdown (%)',
        yaxis_title='Drawdown (%)',
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

def plot_confusion_matrix(cm: np.ndarray, labels=["Down","Neutral","Up"]) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale='Viridis', text=cm,
        texttemplate='%{text}', textfont={"size": 16},
        hovertemplate="%{y} → %{x}<br>%{z}<extra></extra>"
    ))
    fig.update_layout(
        template='plotly_white', height=320,
        title='Matrice de Confusion (Validation)',
        xaxis_title='Classe Prédite', yaxis_title='Classe Réelle',
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

def plot_feature_importance(fi: dict, top_n=15) -> go.Figure:
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [x[0] for x in sorted_fi]
    values = [x[1] for x in sorted_fi]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation='h',
        marker_color='#667eea',
        hovertemplate="%{x:.4f}<extra></extra>"
    ))
    fig.update_layout(
        template='plotly_white', height=380,
        title=f'Top {top_n} Features Importantes (XGBoost)',
        xaxis_title='Importance',
        yaxis=dict(autorange='reversed'),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

def plot_corr_matrix(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
        text=corr.round(2).values, texttemplate='%{text}',
        textfont={"size": 11},
        hovertemplate="%{y} vs %{x}<br>%{z:.2f}<extra></extra>"
    ))
    fig.update_layout(
        template='plotly_white', height=400,
        title='Matrice de Corrélation des Rendements',
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

# ── Main Application ──────────────────────────────────────────────────────────
def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        # FinSight
        ## Analyse et gestion de risque de portefeuille financier
        """)
        st.markdown("""
        <div class="info-box">
            <h3>Que fait cette application ?</h3>
            <p style="margin: 0.5rem 0;">FinSight vous aide à analyser vos investissements en 4 étapes :</p>
            <ol style="margin: 0.5rem 0; padding-left: 1.5rem;">
                <li><strong>Analyse du marché</strong> : Visualiser la performance des actifs</li>
                <li><strong>Évaluation du risque</strong> : Mesurer les pertes potentielles (VaR) et faire des stress tests</li>
                <li><strong>Signaux IA</strong> : Utiliser le machine learning pour prédire les mouvements futurs</li>
                <li><strong>Backtesting</strong> : Tester une stratégie sur des données passées</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='text-align: right; padding-top: 1rem;'>
            <a href='https://github.com/KalsoumDS/finsight' target='_blank' style='text-decoration: none;'>
                <span style='background: #f1f5f9; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 600; color: #475569;'>
                    GitHub
                </span>
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()

    # ── Sidebar Configuration ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Configuration")
        
        st.markdown("### Univers d'actifs")
        category = st.selectbox("Catégorie", list(ASSET_UNIVERSE.keys()), index=0)
        available = ASSET_UNIVERSE[category]
        default_tickers = list(available.keys())[:3] if category == "Actions US" else list(available.keys())[:2]
        selected = st.multiselect(
            "Tickers", list(available.keys()),
            default=default_tickers,
            format_func=lambda x: f"{x} — {available.get(x, x)}"
        )

        st.markdown("### Période")
        period = st.selectbox("Historique", ["1 an", "2 ans", "3 ans", "5 ans"], index=1)
        period_map = {"1 an": 365, "2 ans": 730, "3 ans": 1095, "5 ans": 1825}
        from datetime import datetime, timedelta
        n_days = period_map[period]
        start_date = (datetime.today() - timedelta(days=n_days)).strftime("%Y-%m-%d")

        st.markdown("### Portefeuille")
        portfolio_value = st.number_input("Valeur initiale ($)", 10_000, 50_000_000, 100_000, 10_000)
        equal_weight = st.checkbox("Pondération égale", value=True)

        st.markdown("### Paramètres Risque")
        confidence = st.slider("Niveau de confiance VaR", 0.90, 0.99, 0.95, 0.01)
        horizon = st.selectbox("Horizon (jours)", [1, 5, 10, 21], index=0)
        n_mc = st.select_slider("Simulations Monte Carlo", [1_000, 5_000, 10_000, 50_000], value=10_000)

        load_btn = st.button("Charger les données", type="primary", use_container_width=True)

        st.divider()
        st.markdown("""
        <div style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>
            <p><strong>Stack Technique</strong></p>
            <p>Python · Streamlit · Plotly</p>
            <p>XGBoost · PyTorch · yfinance</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Load Data ─────────────────────────────────────────────────────────────
    if load_btn and selected:
        with st.spinner(f"Téléchargement des données pour {', '.join(selected)} depuis Yahoo Finance..."):
            try:
                prices = loader.get_prices(selected, start=start_date)
                returns = loader.get_returns(prices)
                st.session_state.prices = prices
                st.session_state.returns = returns
                st.session_state.tickers = selected
                st.session_state.risk_results = {}
                st.session_state.ml_xgb = None
                st.session_state.backtest_result = None
                st.session_state.data_loaded = True
                st.success(f"Données chargées ! {len(prices)} jours historiques pour {len(selected)} actifs.")
            except Exception as e:
                st.error(f"Erreur lors du chargement : {str(e)}")

    if not st.session_state.data_loaded or st.session_state.prices is None:
        st.info("Commencez par sélectionner des actifs et cliquez sur Charger les données dans la barre latérale.")
        
        # Demo preview
        st.markdown("---")
        st.markdown("### Aperçu des Fonctionnalités")
        demo_cols = st.columns(4)
        with demo_cols[0]:
            st.markdown("""
            <div style='background: #f8fafc; padding: 1.5rem; border-radius: 12px; text-align: center;'>
                <h4 style='margin-top: 0.75rem; margin-bottom: 0.25rem;'>Marché</h4>
                <p style='color: #64748b; font-size: 0.9rem;'>Prix & rendements</p>
            </div>
            """, unsafe_allow_html=True)
        with demo_cols[1]:
            st.markdown("""
            <div style='background: #f8fafc; padding: 1.5rem; border-radius: 12px; text-align: center;'>
                <h4 style='margin-top: 0.75rem; margin-bottom: 0.25rem;'>Risk Engine</h4>
                <p style='color: #64748b; font-size: 0.9rem;'>VaR & Stress Tests</p>
            </div>
            """, unsafe_allow_html=True)
        with demo_cols[2]:
            st.markdown("""
            <div style='background: #f8fafc; padding: 1.5rem; border-radius: 12px; text-align: center;'>
                <h4 style='margin-top: 0.75rem; margin-bottom: 0.25rem;'>ML Signals</h4>
                <p style='color: #64748b; font-size: 0.9rem;'>XGBoost & LSTM</p>
            </div>
            """, unsafe_allow_html=True)
        with demo_cols[3]:
            st.markdown("""
            <div style='background: #f8fafc; padding: 1.5rem; border-radius: 12px; text-align: center;'>
                <h4 style='margin-top: 0.75rem; margin-bottom: 0.25rem;'>Backtesting</h4>
                <p style='color: #64748b; font-size: 0.9rem;'>Performance metrics</p>
            </div>
            """, unsafe_allow_html=True)
        return

    prices = st.session_state.prices
    returns = st.session_state.returns
    tickers = st.session_state.tickers
    weights = np.ones(len(tickers)) / len(tickers)

    # Check empty data
    if prices.empty or len(prices.columns) == 0:
        st.warning("Aucune donnée valide n'a été chargée. Essayez avec d'autres tickers ou une période plus longue.")
        return

    # ── Tabs Navigation ────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Marché & Portefeuille",
        "Risk Engine",
        "ML Alpha Signals",
        "Backtesting"
    ])

    # ── Tab 1: Market & Portfolio ─────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Performance des Actifs</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <p style="margin: 0;">Ce graphique montre la performance de chaque actif depuis le début de la période, normalisée à une base 100. Cela permet de comparer facilement comment les différents actifs ont évolué par rapport les uns aux autres.</p>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(plot_prices(prices), use_container_width=True)

        # Portfolio stats
        port_stats = loader.get_portfolio_stats(prices, weights)
        
        st.markdown('<div class="section-header">Métriques du Portefeuille</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <h4 style="margin: 0.5rem 0;">Comprendre ces métriques :</h4>
            <ul style="margin: 0;">
                <li><strong>Rendement Annualisé</strong> : Si vous aviez investi il y a un an, c'est le pourcentage que vous auriez gagné ou perdu par an</li>
                <li><strong>Volatilité Annualisée</strong> : Mesure du risque : plus c'est élevé, plus les prix ont fluctué</li>
                <li><strong>Sharpe Ratio</strong> : Rapport entre le rendement et le risque. Plus c'est élevé, mieux c'est (généralement > 1 c'est bon)</li>
                <li><strong>Max Drawdown</strong> : La pire perte que vous auriez eu si vous aviez acheté au plus haut et vendu au plus bas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        metric_cols = st.columns(4)
        with metric_cols[0]:
            val = port_stats['annual_return'] * 100
            delta = f"{val:.2f}%"
            st.metric("Rendement Annualisé", f"{val:.2f}%", delta=delta)
        with metric_cols[1]:
            val = port_stats['annual_vol'] * 100
            st.metric("Volatilité Annualisée", f"{val:.2f}%")
        with metric_cols[2]:
            st.metric("Sharpe Ratio", f"{port_stats['sharpe_ratio']:.3f}")
        with metric_cols[3]:
            val = port_stats['max_drawdown'] * 100
            st.metric("Max Drawdown", f"{val:.2f}%", delta=f"{val:.2f}%", delta_color="inverse")

        # Correlation matrix
        if len(tickers) > 1:
            st.divider()
            st.markdown('<div class="section-header">Matrice de Corrélation</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
                <p style="margin: 0;">La matrice de corrélation montre comment les actifs évoluent les uns par rapport aux autres :</p>
                <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                    <li><strong>Proche de 1</strong> : Les actifs montent et descendent en même temps</li>
                    <li><strong>Proche de -1</strong> : Quand l'un monte, l'autre descend</li>
                    <li><strong>Proche de 0</strong> : Aucune relation</li>
                </ul>
                <p style="margin: 0.5rem 0 0; font-size: 0.9rem;">Astuce : pour diversifier votre portefeuille, il vaut mieux choisir des actifs peu corrélés.</p>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(plot_corr_matrix(port_stats['corr_matrix']), use_container_width=True)

    # ── Tab 2: Risk Engine ────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Value at Risk (VaR) — 3 Méthodes</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <h4 style="margin: 0.5rem 0;">Qu'est-ce que la VaR ?</h4>
            <p style="margin: 0.5rem 0;">La Value at Risk (VaR) est une mesure du risque : c'est la <strong>perte maximale que vous pouvez attendre</strong> avec un certain niveau de confiance et sur une certaine période.</p>
            <h5 style="margin: 0.5rem 0;">Exemple :</h5>
            <p style="margin: 0;">Si vous avez une VaR de 5% au niveau de confiance de 95% sur 1 jour, cela signifie qu'il y a 95% de chances que vous ne perdiez pas plus de 5% de votre portefeuille en un jour.</p>
        </div>
        """, unsafe_allow_html=True)

        # Asset selection
        primary = st.selectbox("Actif pour l'analyse univariée", tickers, key="risk_ticker")
        ret_primary = returns[primary] if primary in returns.columns else returns.iloc[:, 0]

        risk = RiskEngine(confidence=confidence, horizon=horizon)

        compute_risk = st.button("Calculer les métriques de risque", type="primary")
        
        if compute_risk or st.session_state.risk_results:
            if compute_risk:
                with st.spinner("Calcul des métriques de risque en cours..."):
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
            
            r = st.session_state.risk_results
            hist, param, mc = r["hist"], r["param"], r["mc"]

            # Comparison table
            st.markdown("#### Comparatif des Méthodes VaR")
            st.dataframe(r["comparison"], use_container_width=True)

            # Charts
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.markdown("**Distribution des rendements**")
                st.markdown("Ce graphique montre la distribution des rendements journaliers de l'actif avec les seuils de VaR.")
                st.plotly_chart(
                    plot_returns_dist(ret_primary, hist["var_pct"], param["var_pct"]),
                    use_container_width=True
                )
            with chart_cols[1]:
                st.markdown("**Simulation Monte Carlo**")
                st.markdown(f"Nous avons simulé {n_mc} scénarios futurs pour voir comment votre portefeuille pourrait évoluer.")
                st.plotly_chart(
                    plot_mc_simulation(mc["simulated_returns"], mc["var_pct"]),
                    use_container_width=True
                )

            # Key metrics
            st.markdown('<div class="section-header">Métriques Clés</div>', unsafe_allow_html=True)
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric(
                    "VaR Historique",
                    f"${hist['var_abs']:,.0f}",
                    delta=f"{hist['var_pct']*100:.3f}%",
                    delta_color="inverse"
                )
            with metric_cols[1]:
                st.metric(
                    "CVaR Historique",
                    f"${hist['cvar_abs']:,.0f}",
                    delta=f"{hist['cvar_pct']*100:.3f}%",
                    delta_color="inverse"
                )
            with metric_cols[2]:
                st.metric(
                    "VaR Monte Carlo",
                    f"${mc['var_abs']:,.0f}",
                    delta=f"{mc['var_pct']*100:.3f}%",
                    delta_color="inverse"
                )
            with metric_cols[3]:
                bv = r["backtest_var"]
                if bv.get("insufficient_data", False):
                    st.metric(
                        "Test Kupiec",
                        "Données insuffisantes",
                        delta="Pas assez de données historiques"
                    )
                else:
                    status = "Valide" if bv["model_valid"] else "Invalide"
                    st.metric(
                        "Test Kupiec",
                        status,
                        delta=f"{bv['n_violations']} violations ({bv['violation_rate']*100:.1f}%)"
                    )

            # Normality test
            norm_ok = param["normality_ok"]
            st.info(
                f"Test de Normalité (Jarque-Bera) — "
                f"p-value: `{param['jarque_bera_pvalue']:.4f}` | "
                f"Skewness: `{param['skewness']:.3f}` | "
                f"Kurtosis: `{param['kurtosis']:.3f}` | "
                f"{'Distribution normale' if norm_ok else 'Non-normale (queues épaisses)'}"
            )
            st.markdown("""
            <div class="info-box">
                <p style="margin: 0;">Le test de Jarque-Bera vérifie si les rendements suivent une distribution normale :</p>
                <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                    <li><strong>Skewness</strong> : Mesure l'asymétrie (0 = symétrique)</li>
                    <li><strong>Kurtosis</strong> : Mesure la "fatness" des queues (3 = normale, >3 = plus de risques extrêmes)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Component VaR
            if len(tickers) > 1:
                st.markdown('<div class="section-header">Component VaR — Contribution par Actif</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="info-box">
                    <p style="margin: 0;">Le Component VaR vous montre combien chaque actif contribue au risque total du portefeuille. Cela vous permet d'identifier les actifs qui font monter le risque le plus.</p>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(r["comp_var"].round(4), use_container_width=True)

            # Stress Testing
            st.divider()
            st.markdown('<div class="section-header">Stress Testing — Scénarios Historiques</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
                <h4 style="margin: 0.5rem 0;">Qu'est-ce que le Stress Testing ?</h4>
                <p style="margin: 0;">Le stress testing simule comment votre portefeuille aurait performé lors de crises financières passées (crise de 2008, COVID-19, etc.). C'est une façon de voir : <em>"Qu'est-ce qui pourrait arriver si l'histoire se répète ?"</em></p>
            </div>
            """, unsafe_allow_html=True)
            scenario = st.selectbox("Sélectionnez un scénario", list(STRESS_SCENARIOS.keys()))
            if st.button("Lancer le Stress Test"):
                with st.spinner(f"Simulation du scénario : {scenario}..."):
                    try:
                        _, stress_ret = loader.get_stress_data(tickers, scenario)
                        stress_ret = stress_ret.reindex(columns=tickers).fillna(0)
                        stress_result = risk.stress_test(
                            stress_ret, weights, portfolio_value, scenario
                        )
                        st.markdown("#### Résultats du stress test")
                        stress_cols = st.columns(4)
                        with stress_cols[0]:
                            st.metric(
                                "Perte Totale",
                                f"${stress_result['total_loss_abs']:,.0f}",
                                delta=f"{stress_result['total_return_pct']:.2f}%",
                                delta_color="inverse"
                            )
                        with stress_cols[1]:
                            st.metric(
                                "Max Drawdown",
                                f"{stress_result['max_drawdown_pct']:.2f}%",
                                delta_color="inverse"
                            )
                        with stress_cols[2]:
                            st.metric(
                                "Pire Journée",
                                f"{stress_result['worst_day_pct']:.2f}%",
                                delta_color="inverse"
                            )
                        with stress_cols[3]:
                            st.metric(
                                "Volatilité Annualisée",
                                f"{stress_result['annualized_vol_pct']:.2f}%"
                            )
                            
                    except Exception as e:
                        st.error(f"Erreur lors du stress test : {str(e)}")

    # ── Tab 3: ML Signals ─────────────────────────────────────────────────────
    with tab3:
        if not ML_AVAILABLE:
            st.warning("Les modules ML (XGBoost / PyTorch) ne sont pas disponibles sur cette instance.")
            st.info("Pour utiliser cette fonctionnalité, installez les dépendances : `pip install xgboost torch scikit-learn`")
        else:
            st.markdown('<div class="section-header">Génération de Signaux ML</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
                <h4 style="margin: 0.5rem 0;">Comment ça marche ?</h4>
                <ol style="margin: 0.5rem 0; padding-left: 1.5rem;">
                    <li>Nous calculons des indicateurs techniques (moyennes mobiles, RSI, MACD, etc.)</li>
                    <li>Nous entraînons un modèle de machine learning pour prédire si le prix va monter, descendre ou rester stable</li>
                    <li>Le modèle génère des signaux que vous pouvez utiliser pour vos décisions d'investissement</li>
                </ol>
                <p style="margin: 0.5rem 0; font-size: 0.9rem;">Ceci est à des fins éducatives et ne constitue pas des conseils financiers.</p>
            </div>
            """, unsafe_allow_html=True)

            ml_ticker = st.selectbox("Actif cible pour les signaux", tickers, key="ml_ticker")
            ret_ml = returns[ml_ticker] if ml_ticker in returns.columns else returns.iloc[:, 0]
            price_ml = prices[ml_ticker] if ml_ticker in prices.columns else prices.iloc[:, 0]

            param_cols = st.columns(3)
            with param_cols[0]:
                fwd_horizon = st.selectbox("Horizon de prédiction (jours)", [1, 3, 5, 10], index=2)
            with param_cols[1]:
                threshold = st.slider("Seuil de signal (%)", 0.1, 2.0, 0.5, 0.1) / 100
            with param_cols[2]:
                model_choice = st.selectbox("Modèle", ["XGBoost", "LSTM (PyTorch)"])

            train_btn = st.button(f"Entraîner le modèle {model_choice}", type="primary")

            if train_btn:
                with st.spinner(f"Feature engineering + entraînement {model_choice} en cours..."):
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
                            st.success(
                                f"XGBoost entraîné avec succès ! "
                                f"Accuracy: {results['accuracy']*100:.1f}% | "
                                f"F1 Macro: {results['f1_macro']:.3f}"
                            )
                        except Exception as e:
                            st.error(f"Erreur lors de l'entraînement : {str(e)}")
                    else:  # LSTM
                        model = LSTMSignalGenerator(
                            forward_horizon=fwd_horizon,
                            threshold=threshold,
                            n_epochs=30,
                            sequence_length=20,
                        )
                        progress_bar = st.progress(0)
                        loss_ph = st.empty()

                        def callback(epoch, total, loss, val_acc):
                            progress_bar.progress(epoch / total)
                            loss_ph.markdown(
                                f"Epoch {epoch}/{total} - Loss: `{loss:.4f}` - Val Acc: `{val_acc*100:.1f}%`"
                            )

                        try:
                            results = model.fit(features_df, ret_ml, progress_callback=callback)
                            st.session_state.ml_xgb = model
                            st.session_state.features = features_df
                            st.success(
                                f"LSTM entraîné avec succès ! "
                                f"Accuracy: {results['accuracy']*100:.1f}% | "
                                f"F1 Macro: {results['f1_macro']:.3f}"
                            )
                        except Exception as e:
                            st.error(f"Erreur lors de l'entraînement : {str(e)}")

            # Display results
            ml_model = st.session_state.ml_xgb
            if ml_model and hasattr(ml_model, 'results_') and ml_model.results_:
                res = ml_model.results_
                st.divider()
                
                st.markdown('<div class="section-header">Résultats du Modèle</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="info-box">
                    <h4 style="margin: 0.5rem 0;">Comment évaluer la performance ?</h4>
                    <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                        <li><strong>Accuracy</strong> : Pourcentage de prédictions correctes</li>
                        <li><strong>F1 Macro</strong> : Moyenne des performances par classe (Down, Neutral, Up)</li>
                        <li><strong>Matrice de Confusion</strong> : Où le modèle se trompe le plus souvent</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Accuracy", f"{res['accuracy']*100:.2f}%")
                with metric_cols[1]:
                    st.metric("F1 Macro", f"{res['f1_macro']:.4f}")
                with metric_cols[2]:
                    st.metric("F1 Weighted", f"{res['f1_weighted']:.4f}")
                with metric_cols[3]:
                    n_samples = res.get('n_samples') or res.get('n_val', '?')
                    st.metric("Échantillons de Test", str(n_samples))

                result_cols = st.columns(2)
                with result_cols[0]:
                    st.markdown("**Matrice de Confusion**")
                    st.markdown("Cette table montre où le modèle a raison et où il se trompe.")
                    st.plotly_chart(
                        plot_confusion_matrix(res['confusion_matrix']),
                        use_container_width=True
                    )
                with result_cols[1]:
                    if "feature_importance" in res:
                        st.markdown("**Feature Importance (XGBoost)**")
                        st.markdown("Ces indicateurs ont le plus influencé les prédictions du modèle.")
                        st.plotly_chart(
                            plot_feature_importance(res['feature_importance']),
                            use_container_width=True
                        )
                    elif "train_losses" in res:
                        st.markdown("**Courbe de Loss (LSTM)**")
                        st.markdown("Cette courbe montre comment l'erreur du modèle diminue au fil de l'entraînement.")
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(
                            y=res['train_losses'], mode='lines+markers',
                            line=dict(color='#10b981', width=2), name='Loss'
                        ))
                        fig_loss.update_layout(
                            template='plotly_white', height=320,
                            title='Courbe de Loss (LSTM)',
                            xaxis_title='Epoch', yaxis_title='Cross-Entropy Loss',
                            margin=dict(l=0, r=0, t=50, b=0)
                        )
                        st.plotly_chart(fig_loss, use_container_width=True)

                # Classification report
                if "classification_report" in res:
                    st.markdown("**Rapport de Classification**")
                    st.markdown("""
                    <div class="info-box">
                        <ul style="margin: 0; padding-left: 1.5rem;">
                            <li><strong>Precision</strong> : Si le modèle prédit une hausse, combien de fois a-t-il raison ?</li>
                            <li><strong>Recall</strong> : Combien des vraies hausses le modèle a-t-il détectées ?</li>
                            <li><strong>F1-Score</strong> : Moyenne harmonique des deux (mieux) </li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    cr = res["classification_report"]
                    cr_df = pd.DataFrame({
                        "Classe": ["Down", "Neutral", "Up"],
                        "Precision": [cr.get("Down",{}).get("precision",0), cr.get("Neutral",{}).get("precision",0), cr.get("Up",{}).get("precision",0)],
                        "Recall": [cr.get("Down",{}).get("recall",0), cr.get("Neutral",{}).get("recall",0), cr.get("Up",{}).get("recall",0)],
                        "F1-Score": [cr.get("Down",{}).get("f1-score",0), cr.get("Neutral",{}).get("f1-score",0), cr.get("Up",{}).get("f1-score",0)],
                    }).set_index("Classe")
                    st.dataframe(cr_df.round(4), use_container_width=True)

    # ── Tab 4: Backtesting ────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">Backtesting de Stratégie</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <h4 style="margin: 0.5rem 0;">Qu'est-ce que le Backtesting ?</h4>
            <p style="margin: 0.5rem 0;">Le backtesting simule comment une stratégie aurait fonctionné si vous l'aviez utilisée sur des données passées.</p>
            <h5 style="margin: 0.5rem 0;">Comment ça fonctionne :</h5>
            <ol style="margin: 0.5rem 0; padding-left: 1.5rem;">
                <li>Le modèle génère des signaux ("acheter", "vendre", "ne rien faire")</li>
                <li>Nous simulons ces trades sur les données historiques</li>
                <li>Nous incluons les coûts de transaction et le slippage pour plus de réalisme</li>
                <li>Nous comparons avec une stratégie simple : "Buy & Hold" (acheter et conserver)</li>
            </ol>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">Important : Les performances passées ne garantissent pas les performances futures.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if we have ML signals
        if not st.session_state.ml_xgb or not st.session_state.features:
            st.info("Pour backtester une stratégie ML, allez d'abord sur l'onglet **ML Alpha Signals** et entraînez un modèle.")
        else:
            ml_model = st.session_state.ml_xgb
            features_df = st.session_state.features
            ret_ml = returns[list(returns.columns)[0]]  # Use first ticker
            
            # Get signals
            try:
                signals_df = ml_model.get_signals(features_df)
                signals = signals_df['signal']
                
                # Run backtest
                backtest_engine = BacktestEngine(transaction_cost=0.001, slippage=0.0005)
                backtest_result = backtest_engine.run(
                    ret_ml, signals, initial_capital=portfolio_value
                )
                
                st.markdown('<div class="section-header">Résultats du Backtest</div>', unsafe_allow_html=True)
                
                # Charts
                st.markdown("**Courbe de Capital**")
                st.markdown("Ce graphique montre comment votre capital aurait évolué avec la stratégie ML par rapport à acheter et conserver.")
                st.plotly_chart(plot_equity_curve(backtest_result), use_container_width=True)
                
                dd_col, _ = st.columns([2, 1])
                with dd_col:
                    st.markdown("**Drawdown**")
                    st.markdown("Le drawdown mesure la perte depuis le sommet le plus haut. Moins il y en a, mieux c'est.")
                    st.plotly_chart(plot_drawdown(backtest_result), use_container_width=True)
                
                # Metrics table
                st.markdown('<div class="section-header">Métriques de Performance</div>', unsafe_allow_html=True)
                
                strat_metrics = backtest_result['strategy_metrics']
                bench_metrics = backtest_result['benchmark_metrics']
                
                comp_data = {
                    "Métrique": [
                        "Rendement Total", "Rendement Annualisé", "Volatilité",
                        "Sharpe Ratio", "Sortino Ratio", "Max Drawdown",
                        "Win Rate", "Profit Factor", "VaR 95%",
                    ],
                    "Stratégie ML": [
                        f"{strat_metrics['total_return_pct']:.2f}%",
                        f"{strat_metrics['annual_return_pct']:.2f}%",
                        f"{strat_metrics['annual_vol_pct']:.2f}%",
                        f"{strat_metrics['sharpe_ratio']:.3f}",
                        f"{strat_metrics['sortino_ratio']:.3f}",
                        f"{strat_metrics['max_drawdown_pct']:.2f}%",
                        f"{strat_metrics['win_rate_pct']:.1f}%",
                        f"{strat_metrics['profit_factor']:.3f}",
                        f"{strat_metrics['var_95_pct']:.2f}%",
                    ],
                    "Buy & Hold": [
                        f"{bench_metrics['total_return_pct']:.2f}%",
                        f"{bench_metrics['annual_return_pct']:.2f}%",
                        f"{bench_metrics['annual_vol_pct']:.2f}%",
                        f"{bench_metrics['sharpe_ratio']:.3f}",
                        f"{bench_metrics['sortino_ratio']:.3f}",
                        f"{bench_metrics['max_drawdown_pct']:.2f}%",
                        f"{bench_metrics['win_rate_pct']:.1f}%",
                        f"{bench_metrics['profit_factor']:.3f}",
                        f"{bench_metrics['var_95_pct']:.2f}%",
                    ],
                }
                comp_df = pd.DataFrame(comp_data).set_index("Métrique")
                st.dataframe(comp_df, use_container_width=True)
                
                # Additional info
                info_cols = st.columns(2)
                with info_cols[0]:
                    st.metric("Nombre de Trades", f"{backtest_result['n_trades']}")
                with info_cols[1]:
                    final_cap = strat_metrics['final_capital']
                    st.metric("Capital Final", f"${final_cap:,.0f}")
                
            except Exception as e:
                st.error(f"Erreur lors du backtest : {str(e)}")


if __name__ == "__main__":
    main()
