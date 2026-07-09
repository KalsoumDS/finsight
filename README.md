# FinSight - Plateforme Quantitative de Gestion de Risque & Alpha

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![Plotly](https://img.shields.io/badge/Plotly-5.15+-cyan)
![License](https://img.shields.io/badge/license-MIT-green)

## Table des Matières

- [À propos](#a-propos)
- [Fonctionnalités](#fonctionnalites)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du Projet](#structure-du-projet)
- [Stack Technique](#stack-technique)
- [À propos de l'Auteur](#a-propos-de-lauteur)

## À propos

**FinSight** est une plateforme quantitative professionnelle pour l'analyse de données financières, la gestion de risque et la génération de signaux alpha via machine learning. Développée avec Streamlit pour une expérience utilisateur fluide et intuitive.

## Fonctionnalités

### 1. Marché & Portefeuille
- Téléchargement de données historiques via Yahoo Finance
- Visualisation des performances (base 100)
- Calcul des métriques clés :
  - Rendement annualisé
  - Volatilité annualisée
  - Sharpe Ratio
  - Max Drawdown
- Matrice de corrélation des rendements

### 2. Risk Engine
- **Value at Risk (VaR)** via 3 méthodes :
  - Historique (non paramétrique)
  - Paramétrique (distribution normale)
  - Monte Carlo (GBM)
- **Conditional VaR (CVaR) / Expected Shortfall**
- **Component VaR** pour la décomposition par actif
- **Stress Testing** sur scénarios historiques (Crise 2008, COVID 2020, etc.)
- **Backtesting VaR** avec test de Kupiec

### 3. ML Alpha Signals
- **XGBoost** : Modèle gradient boosting
- **LSTM (PyTorch)** : Réseau de neurones récurrent
- Feature engineering complet (indicateurs techniques)
- Walk-forward validation sans data leakage
- Métriques : Accuracy, F1-score, matrice de confusion
- Feature importance (pour XGBoost)

### 4. Backtesting
- Évaluation rigoureuse des stratégies ML
- Métriques de performance standard :
  - Sharpe Ratio
  - Sortino Ratio
  - Calmar Ratio
  - Profit Factor
  - Win Rate
- Coûts de transaction et slippage
- Courbes de capital et drawdown

## Installation

### Prérequis
- Python 3.10+
- pip

### Étapes

1. **Cloner le dépôt**
```bash
git clone https://github.com/KalsoumDS/finsight.git
cd finsight
```

2. **Créer un environnement virtuel (recommandé)**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

## Utilisation

1. **Lancer l'application Streamlit**
```bash
streamlit run app.py
```

2. **Naviguer dans l'application**
- Sélectionnez les actifs dans la barre latérale
- Choisissez la période historique
- Explorez les différentes fonctionnalités via les onglets

## Structure du Projet

```
finsight/
├── app.py                      # Application principale Streamlit
├── requirements.txt            # Dépendances Python
├── README.md                  # Ce fichier
└── core/                      # Modules métier
    ├── __init__.py
    ├── data.py               # Chargement et préparation des données
    ├── risk.py               # Moteur de calcul de risque (VaR, Stress Test)
    ├── backtest.py           # Moteur de backtesting
    └── ml_signals.py         # Modèles ML pour la génération de signaux
```

## Stack Technique

| Catégorie | Technologies |
|-----------|--------------|
| **Interface** | Streamlit, Plotly |
| **Data Science** | NumPy, Pandas, Scikit-learn, SciPy |
| **Machine Learning** | XGBoost, PyTorch |
| **Données** | Yahoo Finance (yfinance) |

## À propos de l'auteur

**Oumou Kaltoum Sall** - Data Scientist & ML Engineer

Passionnée par l'IA appliquée à la finance, avec expertise en :
- Computer Vision pour la reconnaissance faciale industrielle
- AutoML et pipelines MLOps
- Déploiement de modèles en production

- GitHub : [@KalsoumDS](https://github.com/KalsoumDS)
- Portfolio : [portfolio-oumou-kaltoum.vercel.app](https://portfolio-oumou-kaltoum.vercel.app)

## Licence

MIT License - voir le fichier LICENSE pour plus de détails.
