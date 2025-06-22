import argparse

def get_config():
    parser = argparse.ArgumentParser(description="Football Player Physical Stats Prediction")
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--target', type=str, required=True, help='Target column(s), comma-separated for multi')
    parser.add_argument('--mode', type=str, choices=['single', 'multi'], default='single', help='Regression mode')
    parser.add_argument('--horizon', type=int, default=10, help='Prediction horizon in minutes ahead')
    parser.add_argument('--sequence_length', type=int, default=10, help='Sequence length (minutes back to use)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='LSTM hidden dim')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM layers')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional LSTM')
    parser.add_argument('--attention', action='store_true', help='Use attention')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--save_path', type=str, default='model.pt', help='Where to save the model')
    parser.add_argument('--model_path', type=str, help='Path to load a trained model')
    args = parser.parse_args()
    return args