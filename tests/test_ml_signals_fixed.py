
"""Test the exact flow of XGBoostSignalGenerator.fit()"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score

from core.data import MarketDataLoader
from core.ml_signals import make_labels


def walk_forward_split(n, n_splits=3, min_train_size=252):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    indices = np.arange(n)
    splits = []
    for train_idx, test_idx in tscv.split(indices):
        if len(train_idx) >= min_train_size:
            splits.append((train_idx, test_idx))
    return splits


def test_fit():
    print("=== Testing XGBoostSignalGenerator.fit() flow ===")
    loader = MarketDataLoader()
    prices = loader.get_prices(["AAPL"], start="2020-01-01")["AAPL"]
    returns = loader.get_returns(prices.to_frame())["AAPL"]
    features = loader.compute_features(prices)
    print(" Data loaded")

    labels = make_labels(returns, forward_horizon=5, threshold=0.005)
    common_idx = features.index.intersection(labels.dropna().index)

    # Extract numpy arrays, explicitly cast dtype to avoid any issues!
    X = features.loc[common_idx].to_numpy(dtype=np.float64)
    y = labels.loc[common_idx].to_numpy(dtype=np.int64)
    feature_names = list(features.columns)
    print(f" Labels created, X shape {X.shape}, y shape {y.shape}")

    splits = walk_forward_split(len(X), n_splits=2)
    print(f" {len(splits)} splits created")

    all_preds, all_true, all_proba = [], [], []

    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"  Processing split {i+1}/{len(splits)}...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = xgb.XGBClassifier(
            n_estimators=30,
            max_depth=2,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        print("    Training model...")
        model.fit(X_train_s, y_train, verbose=False)

        preds = model.predict(X_test_s)
        proba = model.predict_proba(X_test_s)
        print(f"    Predictions done, accuracy: {accuracy_score(y_test, preds):.3f}")

        all_preds.extend(preds)
        all_true.extend(y_test)
        all_proba.extend(proba)

    # Final model training on all data
    final_scaler = StandardScaler()
    X_s = final_scaler.fit_transform(X)
    final_model = xgb.XGBClassifier(
        n_estimators=30,
        max_depth=2,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    print("Training final model on all data...")
    final_model.fit(X_s, y, verbose=False)
    print(" Final model trained")

    # Aggregate metrics
    all_preds_arr = np.array(all_preds)
    all_true_arr = np.array(all_true)
    print("\n=== Metrics ===")
    print(f"Accuracy: {accuracy_score(all_true_arr, all_preds_arr):.3f}")
    print(f"F1 macro: {f1_score(all_true_arr, all_preds_arr, average='macro'):.3f}")
    print(" Test passed!")


if __name__ == "__main__":
    test_fit()
