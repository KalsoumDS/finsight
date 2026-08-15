
from core.data import MarketDataLoader
from core.ml_signals import make_labels
import pandas as pd
import numpy as np

loader = MarketDataLoader()
prices = loader.get_prices(["AAPL"], start="2020-01-01")["AAPL"]
returns = loader.get_returns(prices.to_frame())["AAPL"]
features = loader.compute_features(prices)
print("Features shape", features.shape)
labels = make_labels(returns, 5, 0.005)
print("Labels head:\n", labels.head(20))
print("Labels dtype:", labels.dtype)
print("Labels value counts:\n", labels.value_counts())

common_idx = features.index.intersection(labels.dropna().index)
X = features.loc[common_idx].values
y = labels.loc[common_idx].values

print("y dtype", y.dtype)
print("y unique values", np.unique(y))
y_int = y.astype(int)
print("y_int dtype", y_int.dtype)
print("y_int unique", np.unique(y_int))

# Now let's try to train a minimal XGBoost directly here!
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = xgb.XGBClassifier(
    n_estimators=10,
    max_depth=2,
    objective="multi:softprob",
    num_class=3,
    random_state=42,
    verbosity=2
)
print("Training XGBoost in test_labels_type.py...")
model.fit(X_scaled, y_int)
print("Model trained in test_labels_type.py!")
