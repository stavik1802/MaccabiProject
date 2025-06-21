import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ─────────────────────── Model Definition ───────────────────────
class RNNPredictor(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_targets=3):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, input_size * num_targets)
        self.num_targets = num_targets
        self.input_size = input_size

    def forward(self, x):
        _, (h_n, _) = self.rnn(x)
        out = self.fc(h_n.squeeze(0))
        return out.view(-1, self.num_targets, self.input_size)

# ─────────────────────── Configuration ───────────────────────
MODEL_PATH = "rnn_forecaster1.pt"
SCALER_PATH = "scaler1.pkl"
CSV_PATH = "first_half_features.csv"
INPUT_LENGTH = 10

# ─────────────────────── Load Data ───────────────────────
df = pd.read_csv(CSV_PATH)
features = df.drop(columns=["minute","accel_decel_balance_last"])
recent_data = features.values[-INPUT_LENGTH:]  # shape: (10, 14)

# ─────────────────────── Load Scaler & Normalize ───────────────────────
scaler: StandardScaler = joblib.load(SCALER_PATH)
recent_norm = scaler.transform(recent_data)

# ─────────────────────── Load Model ───────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = recent_norm.shape[1]

model = RNNPredictor(input_size=input_size).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ─────────────────────── Run Prediction ───────────────────────
input_tensor = torch.tensor(recent_norm, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(input_tensor)  # shape: (1, 3, num_features)
    pred_np = pred.cpu().numpy()[0]  # shape: (3, num_features)

# ─────────────────────── Inverse Transform ───────────────────────
pred_real = scaler.inverse_transform(pred_np)

# ─────────────────────── Output ───────────────────────
time_offsets = [1, 5, 10]
for i, offset in enumerate(time_offsets):
    print(f"Prediction for +{offset} minutes:")
    for col, value in zip(features.columns, pred_real[i]):
        print(f"  {col}: {value:.3f}")
    print()
