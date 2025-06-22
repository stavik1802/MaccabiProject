import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load your real CSV file
df = pd.read_csv("first_half_features.csv")

# Normalize features (excluding 'minute')
features = df.drop(columns=["minute","accel_decel_balance_last"])
scaler = StandardScaler()
normalized = scaler.fit_transform(features)

# Save scaler for future inverse_transform use
import joblib
joblib.dump(scaler, "scaler1.pkl")

# Custom PyTorch Dataset
class PlayerRNNForecastDataset(Dataset):
    def __init__(self, data, input_len=10, target_offsets=[1, 5, 10]):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.input_len = input_len
        self.target_offsets = target_offsets
        self.samples = []
        for i in range(len(data) - input_len - max(target_offsets)):
            X = self.data[i : i + input_len]
            Y = torch.stack([self.data[i + input_len + o] for o in target_offsets])
            self.samples.append((X, Y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Dataset and Dataloader
input_len = 10
target_offsets = [1, 5, 10]
dataset = PlayerRNNForecastDataset(normalized, input_len, target_offsets)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define RNN Model
class RNNPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_targets=3):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size * num_targets)
        self.num_targets = num_targets
        self.input_size = input_size

    def forward(self, x):
        _, (h_n, _) = self.rnn(x)
        out = self.fc(h_n.squeeze(0))
        return out.view(-1, self.num_targets, self.input_size)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = normalized.shape[1]

model = RNNPredictor(input_size=input_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(20):
    model.train()
    total_loss = 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

# Save model
torch.save(model.state_dict(), "rnn_forecaster1.pt")
