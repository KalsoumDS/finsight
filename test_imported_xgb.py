
import sys
sys.path.insert(0, '/Users/macbook/Desktop/CV_Portfolio/finsight')

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from core.data import MarketDataLoader
from core.ml_signals import make_labels, XGBoostSignalGenerator

print("=== Testing imported XGBoostSignalGenerator ====")

loader = MarketDataLoader()
print("Downloading AAPL data")
prices = loader.get_prices(["AAPL"], start="2020-01-01")
print("Prices type", type(prices))
print("Prices columns", prices.columns)
prices_series = prices["AAPL"]
print("Calling get returns")
returns_df = loader.get_returns(prices)
print("returns type", type(returns_df))
print("Calling compute features")
features = loader.compute_features(prices_series)

# Check labels = make_labels(returns_df, 5, 0.005)
labels = make_labels(returns_df["AAPL"], 5, 0.005)
print("Labels type", type(labels))
print("Labels head", labels.head())

common_idx = features.index.intersection(labels.dropna().index)
X = features.loc[common_idx].to_numpy(dtype=np.float64)
y = labels.loc[common_idx].to_numpy(dtype=np.int64)
print("X shape", X.shape, "y shape", y.shape, "y dtype", y.dtype, "y unique", np.unique(y))

print("Creating model")
model = XGBoostSignalGenerator(
    n_estimators=20, max_depth=2, n_splits=1
)

print("Calling model.fit with features and returns_df['AAPL']")
results = model.fit(features, returns_df["AAPL"])
print("Fit complete, Accuracy:", results['accuracy'])
print("Done!")

