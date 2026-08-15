
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from core.data import MarketDataLoader
from core.ml_signals import make_labels

print("=== Testing xgboost with our real data ===")

# Step 1: Get data
loader = MarketDataLoader()
print("Downloading AAPL data...")
prices = loader.get_prices(["AAPL"], start="2020-01-01")
print("Prices type:", type(prices))
print("Prices columns:", prices.columns)
prices_series = prices["AAPL"]
returns = loader.get_returns(prices)["AAPL"]

# Step 2: Compute features and labels
print("Computing features...")
features = loader.compute_features(prices_series)
labels = make_labels(returns, 5, 0.005)

# Step 3: Get common index, extract numpy arrays
common_idx = features.index.intersection(labels.dropna().index)
X_np = features.loc[common_idx].to_numpy(dtype=np.float32)  # Use float32 to be safe!
y_np = labels.loc[common_idx].to_numpy(dtype=np.int32)

print("X_np shape:", X_np.shape, "dtype:", X_np.dtype)
print("y_np shape:", y_np.shape, "dtype:", y_np.dtype)
print("y_np unique:", np.unique(y_np))

# Step 4: Scale
print("Scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_np)

# Step 5: Train XGBoost!
print("Training XGBoost classifier...")
model = xgb.XGBClassifier(
    n_estimators=20,
    max_depth=2,
    learning_rate=0.3,
    objective="multi:softprob",
    num_class=3,
    random_state=42,
    n_jobs=-1,
    verbosity=3
)

print("Calling model.fit()...")
model.fit(X_scaled, y_np, verbose=True)

print("\n✅ Model trained successfully!")
print("Feature importances:", model.feature_importances_)
print("Accuracy on train:", model.score(X_scaled, y_np))
