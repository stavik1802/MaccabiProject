"""Data loading helper to build (X, y) tensors from CSV given feature/target columns."""
import pandas as pd
import numpy as np
import torch

def load_data(
    csv_path,
    feature_cols,
    target_cols,
    horizon=10,
    sequence_length=10
):
    """
    Loads sequential data for LSTM-style models from a single-player CSV.

    Args:
        csv_path: Path to processed CSV.
        feature_cols: List of feature column names.
        target_cols: List of target column names.
        horizon: Predict how many minutes ahead.
        sequence_length: How many minutes of history to use.

    Returns:
        X: [num_samples, sequence_length, num_features]
        y: [num_samples, num_targets]
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values('minute')  # Ensure time order

    X, y = [], []
    for i in range(len(df) - sequence_length - horizon + 1):
        seq_x = df.iloc[i:i+sequence_length][feature_cols].values
        seq_y = df.iloc[i+sequence_length+horizon-1][target_cols].values
        X.append(seq_x)
        y.append(seq_y)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

def get_dataloaders(X, y, batch_size=32, val_split=0.2, test_split=0.1):
    """
    Splits data and creates PyTorch DataLoaders.
    """
    from torch.utils.data import TensorDataset, DataLoader, random_split

    total = len(X)
    test_len = int(total * test_split)
    val_len = int(total * val_split)
    train_len = total - val_len - test_len

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, val_loader, test_loader