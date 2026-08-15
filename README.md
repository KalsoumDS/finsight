# 📈 Finsight — Stress-Testing & Analyse de Risque Financier (Résilience Économique)

> Moteur quantitatif d'évaluation de la résilience financière des portefeuilles et PME face aux chocs de marché (VaR/CVaR, XGBoost & Stress-Testing).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)
![Plotly](https://img.shields.io/badge/Plotly-5.15+-cyan)

---

## 🎯 Problématique Métier & Impact Économique

En période de volatilité et d'inflation, les PME et gestionnaires d'actifs manquent souvent d'outils institutionnels pour modéliser des chocs de liquidité et évaluer leur exposition au risque de baisse catastrophique.

### 💡 Solution & Valeur Ajoutée
**Finsight** démocratise l'ingénierie financière quantique en fournissant une plateforme intégrant :
- **Mesure de Risque Extrême (VaR / CVaR)** par méthodes Paramétrique, Historique et Monte Carlo.
- **Scénarios de Stress-Testing Historiques** (Crise 2008, Choc COVID 2020, Choc Inflationniste).
- **Génération de Signaux ML (XGBoost & PyTorch LSTM)** avec backtesting intégrant les coûts de transaction et le slippage.

---

## 🚀 Fonctionnalités Clés

- 📊 **Risk Engine Institutionnel** : Calcul de la Value-at-Risk (VaR) et Conditional VaR (Expected Shortfall).
- 🌪️ **Stress-Testing Multi-Scénarios** : Simulation de l'impact de crises financières majeures sur les portefeuilles.
- 🤖 **Signaux Alpha ML** : Modélisation prédictive des directions de marché sans data leakage (Walk-forward validation).
- 📈 **Moteur de Backtesting** : Simulation réaliste des stratégies avec calcul des ratios de Sharpe, Sortino et Calmar.

---

## 🛠️ Installation & Lancement

```bash
# Cloner le dépôt
git clone https://github.com/KalsoumDS/finsight.git
cd finsight

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application Streamlit
streamlit run app.py
```

---

## 🔬 Stack Technique

- **Interface** : Streamlit, Plotly
- **Data & Quantitative Finance** : NumPy, Pandas, SciPy, YFinance
- **Machine Learning** : XGBoost, PyTorch (LSTM)

---

## ✍️ Auteur

**Oumou Kaltoum Sall** — Data Scientist & ML Engineer  
[LinkedIn](https://linkedin.com/in/oumou-kaltoum-sall) · [GitHub](https://github.com/KalsoumDS) · [Portfolio](http://localhost:3001)
