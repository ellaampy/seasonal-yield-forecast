import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class YieldDataset(Dataset):
    def __init__(self, features, target):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.target = torch.tensor(target.values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]



class SimpleModel(nn.Module):
    def __init__(self, dropout_rate=0.3, input_size=21):
        super(SimpleModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.fc(x)
    
    


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class MLP(nn.Module):
    def __init__(self, input_dim=15, sequence_len=46, hidden_dim1=512, hidden_dim2 =256, dropout = 0.2):
        super(MLP, self).__init__()
        self.modelname = f"FCN_input-dim={input_dim}_sequence_len={sequence_len}_hidden-dim1={hidden_dim1}_" \
                    f"hidden-dims2={hidden_dim2}_dropout={dropout}"

        self.input_dim = input_dim
        self.sequence_len = sequence_len
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.input_dim*self.sequence_len, self.hidden_dim1)
        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.fc3 = nn.Linear(self.hidden_dim2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x =  x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = x.squeeze(-1)
        return x

