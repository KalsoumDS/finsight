
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data import MarketDataLoader
from core.ml_signals import XGBoostSignalGenerator
import numpy as np

print("Testing XGBoost pipeline...")

# Load sample data
loader = MarketDataLoader()
prices = loader.get_prices(["AAPL"], start="2023-01-01")
print("Prices loaded:", prices.shape)
returns = loader.get_returns(prices)
print("Returns loaded:", returns.shape)

# Compute features
features = loader.compute_features(prices.iloc[:, 0])
print("Features computed:", features.shape)
print("Features columns:", list(features.columns))

# Test make_labels
from core.ml_signals import make_labels
labels = make_labels(returns.iloc[:,0], 5, 0.005)
print("Labels created:", labels.shape)
print("Label value counts:\n", labels.value_counts())

# Fit XGBoost model
model = XGBoostSignalGenerator()
try:
    results = model.fit(features, returns.iloc[:,0])
    print("Model fit successfully!")
    print("Results keys:", list(results.keys()))
    print("Accuracy:", results['accuracy'])
except Exception as e:
    print("ERROR fitting model:", type(e), str(e))
    import traceback
    traceback.print_exc()
