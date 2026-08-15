
from core.data import MarketDataLoader
import numpy as np
import pandas as pd

loader = MarketDataLoader()
prices = loader.get_prices(["AAPL"], start="2020-01-01")["AAPL"]
returns = loader.get_returns(prices.to_frame())["AAPL"]

features = loader.compute_features(prices)
print("Features shape:", features.shape)

print("\nChecking for NaNs in features:")
nans_per_col = features.isna().sum()
print(nans_per_col[nans_per_col > 0])

print("\nChecking for infs in features:")
inf_per_col = (features == np.inf).sum() + (features == -np.inf).sum()
print(inf_per_col[inf_per_col > 0])

print("\nFeatures min/max per column:")
print(features.describe().loc[["min", "max"]])

print("\nAny NaN or inf in all features?", features.isna().any().any() or (features == np.inf).any().any() or (features == -np.inf).any().any())

# Let's also get the common_idx data!
from core.ml_signals import make_labels
labels = make_labels(returns, 5, 0.005)
common_idx = features.index.intersection(labels.dropna().index)
X = features.loc[common_idx].values
print("\nX shape after common index", X.shape)
print("X has NaNs?", np.isnan(X).any())
print("X has infs?", np.isinf(X).any())
