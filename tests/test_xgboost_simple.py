
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

print("Testing simple xgboost classification...")
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data created!")
print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)

model = xgb.XGBClassifier(n_estimators=10, max_depth=2, learning_rate=0.3, objective="multi:softprob", num_class=3, random_state=42, verbosity=2)
print("Fitting model...")
model.fit(X_train, y_train, verbose=True)
print("Model fitted!")

preds = model.predict(X_test)
print("Predictions made! First 10 preds:", preds[:10])
print("XGBoost is working!")
