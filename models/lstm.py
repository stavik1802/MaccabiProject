import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    """
    LSTM-based regressor for single or multi-target regression.
    Supports stacked, bidirectional, and attention (optional) LSTM variants.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1,
                 bidirectional=False, use_attention=False):
        super(LSTMRegressor, self).__init__()
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_dim, output_dim)
        if use_attention:
            self.attn = nn.Linear(lstm_out_dim, 1)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)  # [batch, seq, hidden]
        if self.use_attention:
            # Attention over time dimension
            attn_weights = torch.softmax(self.attn(lstm_out), dim=1)  # [batch, seq, 1]
            context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden]
        else:
            # Use output at final timestep
            context = lstm_out[:, -1, :]  # [batch, hidden]
        out = self.fc(context)  # [batch, output_dim]
        return out