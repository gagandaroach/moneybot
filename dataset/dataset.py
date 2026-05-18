from api.yf import getStocksDF, STOCKS_LIST
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class StocksDataset(Dataset):
    def __init__(self, sequence_length=60, prediction_horizon=1, 
                 feature_columns=None, target_column=None,
                 normalize=True, train=True, train_split=0.8):
        """
        Dataset for stock price prediction.
        
        Args:
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of time steps to predict ahead
            feature_columns: List of columns to use as features (if None, use all OHLCV)
            target_column: Column to predict (if None, predict Close price of first stock)
            normalize: Whether to normalize the data
            train: If True, returns training data; if False, returns validation data
            train_split: Fraction of data to use for training (0.0 to 1.0)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize
        self.train = train
        self.train_split = train_split
        
        # Default to using Close price for prediction if not specified
        if target_column is None:
            self.target_column = ('Close', STOCKS_LIST[0])  # Default to first stock's Close
        else:
            self.target_column = target_column
            
        # Default to using OHLC (excluding Volume) for features if not specified
        if feature_columns is None:
            # Use OHLC (Open, High, Low, Close) for all stocks
            self.feature_columns = []
            for stock in STOCKS_LIST:
                self.feature_columns.append(('Open', stock))
                self.feature_columns.append(('High', stock))
                self.feature_columns.append(('Low', stock))
                self.feature_columns.append(('Close', stock))
        else:
            self.feature_columns = feature_columns
            
        # Load and prepare data
        self.data = getStocksDF()
        self.prepare_data()
        
        # Split into train/val
        self.split_data()
        
    def prepare_data(self):
        """Prepare the data for time series prediction."""
        # Extract feature and target data
        feature_data = self.data[self.feature_columns].values
        target_data = self.data[self.target_column].values.reshape(-1, 1)
        
        # Normalize if requested
        if self.normalize:
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
            feature_data = self.feature_scaler.fit_transform(feature_data)
            target_data = self.target_scaler.fit_transform(target_data)
        else:
            self.feature_scaler = None
            self.target_scaler = None
            
        # Create sequences
        self.sequences = []
        self.targets = []
        
        for i in range(len(feature_data) - self.sequence_length - self.prediction_horizon + 1):
            # Sequence of past observations
            seq = feature_data[i:i + self.sequence_length]
            # Target: future value(s) to predict
            target = target_data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
            self.sequences.append(seq)
            self.targets.append(target)
            
        # Convert to numpy arrays
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        
    def split_data(self):
        """Split the data into training and validation sets."""
        split_idx = int(len(self.sequences) * self.train_split)
        if self.train:
            self.sequences = self.sequences[:split_idx]
            self.targets = self.targets[:split_idx]
        else:
            self.sequences = self.sequences[split_idx:]
            self.targets = self.targets[split_idx:]
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sequence = torch.from_numpy(self.sequences[idx])
        target = torch.from_numpy(self.targets[idx])
        
        return sequence, target
        
    def get_scalers(self):
        """Return the scalers used for normalization."""
        return self.feature_scaler, self.target_scaler

    