import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.config import get_config
from utils.feature_columns import FEATURE_COLUMNS
from utils.metrics import compute_metrics
from utils.visualization import plot_learning_curves
from utils.data_loading import load_data, get_dataloaders
from models.lstm import LSTMRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def main():
    args = get_config()
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define features and targets
    feature_cols = FEATURE_COLUMNS
    target_cols = [t.strip() for t in args.target.split(',')]
    if args.mode == 'single' and len(target_cols) != 1:
        raise ValueError("In single mode, only one target should be specified.")
    output_dim = len(target_cols)

    # Load data
    X, y = load_data(args.data, feature_cols, target_cols, horizon=args.horizon, sequence_length=args.sequence_length)
    train_loader, val_loader, test_loader = get_dataloaders(X, y, batch_size=args.batch_size)

    # Model
    model = LSTMRegressor(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=output_dim,
        bidirectional=args.bidirectional,
        use_attention=args.attention
    ).to(device)
    print(model)

    # Training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                val_preds.append(out.cpu().numpy())
                val_trues.append(yb.cpu().numpy())
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: Train loss={train_loss:.4f}, Val loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), args.save_path)
            best_val_loss = val_loss

    # Plot learning curve
    plot_learning_curves(train_losses, val_losses, save_path='learning_curve.png')
    print("Best model saved to:", args.save_path)

    # Evaluate on test set
    model.load_state_dict(torch.load(args.save_path))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            y_true.append(yb.numpy())
            y_pred.append(out.cpu().numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    metrics = compute_metrics(y_true, y_pred)
    print("Test set metrics:", metrics)

if __name__ == '__main__':
    main()