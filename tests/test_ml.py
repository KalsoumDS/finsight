
"""Test FinSight ML models locally"""
from core.data import MarketDataLoader
from core.ml_signals import XGBoostSignalGenerator, LSTMSignalGenerator
import sys

print("Testing FinSight ML models...")

try:
    # Load test data
    loader = MarketDataLoader()
    print("Downloading AAPL data...")
    prices_df = loader.get_prices(["AAPL"], start="2020-01-01")
    prices = prices_df["AAPL"]
    returns_df = loader.get_returns(prices_df)
    returns = returns_df["AAPL"]
    print(f"Loaded {len(prices)} days of data")

    # Compute features
    print("Computing features...")
    features = loader.compute_features(prices)
    print(f"Features shape: {features.shape}")

    # Test XGBoost
    print("\n--- Testing XGBoost ---")
    xgb_model = XGBoostSignalGenerator(
        n_estimators=20, max_depth=2, n_splits=1,
        forward_horizon=5, threshold=0.005
    )
    print("Calling xgb_model.fit...")
    results = xgb_model.fit(features, returns)
    print(" XGBoost trained successfully!")
    print(f"  Accuracy: {results['accuracy']:.3f}")
    print(f"  F1 macro: {results['f1_macro']:.3f}")

    # Test LSTM
    print("\n--- Testing LSTM ---")
    lstm_model = LSTMSignalGenerator(
        forward_horizon=5,
        threshold=0.005,
        n_epochs=5,
        sequence_length=20,
        batch_size=32
    )
    print("Calling lstm_model.fit...")
    lstm_results = lstm_model.fit(features, returns)
    print(" LSTM trained successfully!")
    print(f"  Accuracy: {lstm_results['accuracy']:.3f}")
    print(f"  F1 macro: {lstm_results['f1_macro']:.3f}")

    print("\n All tests passed!")
except Exception as e:
    print(f"\n Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

