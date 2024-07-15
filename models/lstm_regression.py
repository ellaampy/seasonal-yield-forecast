import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

    
class SequentialYieldPredictors(Dataset):
    def __init__(self, data, targets=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = targets
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        if self.targets is not None:
            y = self.targets[idx]
            return x, y
        return x

class CombinedDataset(Dataset):
    def __init__(self, seq_data, static_data, targets):
        self.seq_data = torch.tensor(seq_data, dtype=torch.float32)
        self.static_data = torch.tensor(static_data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        x_seq = self.seq_data[idx]
        x_static = self.static_data[idx]
        y = self.targets[idx]
        return x_seq, x_static, y
    
    
class LSTMRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMRegression, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


class CombinedModel(nn.Module):
    def __init__(self, seq_input_size, static_input_size, hidden_size, lstm_layers, dense_size, output_size):
        super(CombinedModel, self).__init__()
        self.lstm_layers = lstm_layers
        self.hidden_size = hidden_size
        # LSTM for sequential data
        self.lstm = nn.LSTM(seq_input_size, hidden_size, lstm_layers, batch_first=True)
        
        # Feedforward network for static data
        self.static_fc = nn.Sequential(
            nn.Linear(static_input_size, dense_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dense_size, dense_size),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Combined dense layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + dense_size, dense_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dense_size, output_size)
        )
    
    def forward(self, x_seq, x_static):
        # LSTM forward pass
        h0 = torch.zeros(self.lstm_layers, x_seq.size(0), self.hidden_size).to(x_seq.device)
        c0 = torch.zeros(self.lstm_layers, x_seq.size(0), self.hidden_size).to(x_seq.device)
        lstm_out, _ = self.lstm(x_seq, (h0, c0))
        lstm_out = lstm_out[:, -1, :]  # Take the output from the last time step
        
        # Static features forward pass
        static_out = self.static_fc(x_static)
        
        # Combine LSTM and static outputs
        combined = torch.cat((lstm_out, static_out), dim=1)
        
        # Final dense layers
        out = self.fc(combined)
        return out