
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

n = 1432  # Size of our X
print("Testing TimeSeriesSplit with n_splits=2, n_samples=", n)

tscv = TimeSeriesSplit(n_splits=2, max_train_size=None, test_size=None)
indices = np.arange(n)
splits = []
print("Looping over tscv.split(indices)...")
for i, (train_idx, test_idx) in enumerate(tscv.split(indices)):
    print(f"  Split {i}")
    print(f"    Train size: {len(train_idx)}, first 5 indices: {train_idx[:5]}")
    print(f"    Test size: {len(test_idx)}, first 5 indices: {test_idx[:5]}")
    splits.append((train_idx, test_idx))
print("Number of splits:", len(splits))

print("\nTesting with n_splits=3...")
tscv2 = TimeSeriesSplit(n_splits=3)
splits2 = []
for i, (train_idx, test_idx) in enumerate(tscv2.split(indices)):
    print(f"  Split {i}")
    print(f"    Train size: {len(train_idx)}, test size: {len(test_idx)}")
    splits2.append((train_idx, test_idx))
print("Number of splits with n_splits=3:", len(splits2))

print("Done!")
