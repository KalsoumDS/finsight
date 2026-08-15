
import numpy as np
import pandas as pd
from core.data import MarketDataLoader
from core.ml_signals import make_labels

print("Testing MarketDataLoader...")
loader = MarketDataLoader()
prices = loader.get_prices(["AAPL"], start="2020-01-01")["AAPL"]
returns = loader.get_returns(prices.to_frame())["AAPL"]
print("Prices loaded, len:", len(prices))
print("Returns loaded, len:", len(returns))
print("Computing features...")
features = loader.compute_features(prices)
print("Features shape:", features.shape)
print("Calling make_labels...")
labels = make_labels(returns, forward_horizon=5, threshold=0.005)
print("Labels shape:", labels.shape)
print("Labels value counts (dropna):")
print(labels.dropna().value_counts())
common_idx = features.index.intersection(labels.dropna().index)
print("Common idx len:", len(common_idx))
X = features.loc[common_idx].values
y = labels.loc[common_idx].values.astype(int)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("y classes counts:", np.bincount(y))
