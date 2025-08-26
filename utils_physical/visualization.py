"""Visualization utilities for physical modeling: learning curves and prediction vs. truth plots."""
import matplotlib.pyplot as plt

def plot_learning_curves(train_losses, val_losses, save_path=None):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Learning Curves')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_pred_vs_true(y_true, y_pred, target_names=None, save_path=None):
    import numpy as np
    n_targets = y_true.shape[1] if len(y_true.shape) > 1 else 1
    plt.figure(figsize=(6 * n_targets, 5))
    for t in range(n_targets):
        yt = y_true[:, t] if n_targets > 1 else y_true
        yp = y_pred[:, t] if n_targets > 1 else y_pred
        plt.subplot(1, n_targets, t+1)
        plt.scatter(yt, yp, alpha=0.5)
        plt.xlabel('True')
        plt.ylabel('Predicted')
        name = target_names[t] if target_names else f'Target {t}'
        plt.title(f'Predicted vs. True ({name})')
        minv, maxv = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        plt.plot([minv, maxv], [minv, maxv], 'r--')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()