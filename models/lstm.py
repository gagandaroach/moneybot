import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_size=20, hidden_size=50, num_layers=2, 
                 prediction_horizon=1, dropout=0.2):
        """
        LSTM for stock price prediction.
        
        Args:
            input_size: Number of input features (default: 20 = OHLC for 5 stocks)
            hidden_size: Number of features in hidden state
            num_layers: Number of LSTM layers
            prediction_horizon: Steps to predict ahead
            dropout: Dropout rate for regularization
        """
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, prediction_horizon)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Tensor of shape (batch_size, prediction_horizon)
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch, seq_len, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])  # Take last time step
        out = self.fc(out)  # (batch, prediction_horizon)
        
        return out