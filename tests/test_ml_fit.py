
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from core.data import MarketDataLoader
from core.ml_signals import make_labels


def walk_forward_split(n, n_splits=3, min_train_size=252):
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=None, test_size=None)
    indices = np.arange(n)
    splits = []
    for train_idx, test_idx in tscv.split(indices):
        if len(train_idx) >= min_train_size:
            splits.append((train_idx, test_idx))
    return splits


print("Testing XGB fit step by step...")
loader = MarketDataLoader()
prices = loader.get_prices(["AAPL"], start="2020-01-01")["AAPL"]
returns = loader.get_returns(prices.to_frame())["AAPL"]
features = loader.compute_features(prices)

print("Making labels...")
labels = make_labels(returns, 5, 0.005)
common_idx = features.index.intersection(labels.dropna().index)
X = features.loc[common_idx].values
y = labels.loc[common_idx].values.astype(int)
print("X shape", X.shape, "y shape", y.shape)

splits = walk_forward_split(len(X), n_splits=2)
print("Got splits:", splits)

all_preds, all_true, all_proba = [], [], []
params = {
    "n_estimators": 20,
    "max_depth": 2,
    "learning_rate": 0.3,
    "objective": "multi:softprob",
    "num_class": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": 1,
    "verbosity": 2
}
for i, (train_idx, test_idx) in enumerate(splits):
    print(f"Split {i+1}/{len(splits)}")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Train size: {len(X_train)}, test size {len(X_test)}")

    print("Scaling...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("Training model...")
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_s, y_train, verbose=True)

    print("Predicting...")
    preds = model.predict(X_test_s)
    proba = model.predict_proba(X_test_s)
    print(f"Predictions first 5: {preds[:5]}")

    all_preds.extend(preds)
    all_true.extend(y_test)
    all_proba.extend(proba)

print("All done!")
