import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.config import get_config
from utils.feature_columns import FEATURE_COLUMNS
from utils.metrics import compute_metrics, metrics_explanation
from utils.visualization import plot_pred_vs_true
from utils.data_loading import load_data, get_dataloaders
from models.lstm import LSTMRegressor
import torch
import numpy as np

def main():
    args = get_config()
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    feature_cols = FEATURE_COLUMNS
    target_cols = [t.strip() for t in args.target.split(',')]
    output_dim = len(target_cols)

    X, y = load_data(args.data, feature_cols, target_cols, horizon=args.horizon, sequence_length=args.sequence_length)
    _, _, test_loader = get_dataloaders(X, y, batch_size=args.batch_size, val_split=0.0, test_split=1.0)

    model = LSTMRegressor(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=output_dim,
        bidirectional=args.bidirectional,
        use_attention=args.attention
    ).to(device)
    model.load_state_dict(torch.load(args.model_path))
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
    print("Test set metrics:")
    for t, vals in metrics.items():
        print(f"{t}: MAE={vals['MAE']:.3f}, RMSE={vals['RMSE']:.3f}, R2={vals['R2']:.3f}")
    print(metrics_explanation())
    plot_pred_vs_true(y_true, y_pred, target_names=target_cols, save_path="pred_vs_true.png")
    print("Predicted vs. true plot saved to pred_vs_true.png")

if __name__ == '__main__':
    main()