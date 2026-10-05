import numpy as np
import random
import os
import openml

seed = 10

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)


def load_data():
    """Load the air dataset from OpenML, split into train/test, and preprocess."""

    # Load dataset
    air = openml.datasets.get_dataset(42493)
    X_air, y_air, _, _ = air.get_data(target=air.default_target_attribute)

    # Convert to NumPy arrays
    X_air = np.array(X_air)
    y_air = np.array(y_air)

    # Fixed train/test split (80/20)
    n = X_air.shape[0]
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X_air[train_idx], X_air[test_idx]
    y_train, y_test = y_air[train_idx], y_air[test_idx]

    # Preprocessing — encode categorical columns
    # Flight and DayOfWeek are also categorical but are integers, so no encoding needed
    categorical_cols = [0, 2, 3, 4]
    for col in categorical_cols:
        unique_vals = np.unique(X_train[:, col])
        mapping_vals = {val: idx for idx, val in enumerate(unique_vals)}
        X_train[:, col] = np.array([mapping_vals[val] for val in X_train[:, col]], dtype='float')
        X_test[:, col]  = np.array([mapping_vals.get(val, -1) for val in X_test[:, col]], dtype='float')

    X_train = X_train.astype(float)
    X_test  = X_test.astype(float)
    y_train = y_train.astype(int)
    y_test  = y_test.astype(int)

    return X_train, X_test, y_train, y_test