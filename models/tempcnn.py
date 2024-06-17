from torch import dropout, nn
import torch.nn.functional as F
import torch


"""
TempCNN model
https://github.com/charlotte-pel/temporalCNN
https://www.mdpi.com/2072-4292/11/5/523

adapted from 
https://github.com/MarcCoru/BreizhCrops/blob/master/breizhcrops/models/TempCNN.py
"""


class TempCNN(torch.nn.Module):
    def __init__(self, input_dim=15, num_classes=1, sequencelength=45, kernel_size=7, hidden_dims=128, dropout=0.18203942949809093):
        super(TempCNN, self).__init__()
        self.modelname = f"TempCNN_input-dim={input_dim}_num-classes={num_classes}_sequencelenght={sequencelength}_" \
                         f"kernelsize={kernel_size}_hidden-dims={hidden_dims}_dropout={dropout}"

        self.hidden_dims = hidden_dims

        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(input_dim, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.flatten = Flatten()
        self.dense1 = FC_BatchNorm_Relu_Dropout(hidden_dims * sequencelength, 4 * hidden_dims, drop_probability=dropout)
        self.dense2 = nn.Sequential(nn.Linear(4 * hidden_dims, num_classes))

    def forward(self, x):
        # require NxTxD
        x = x.transpose(1,2)
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x) 
        return x.squeeze(-1)


class Conv1D_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=5, drop_probability=0.5):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class FC_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, drop_probability=0.5):
        super(FC_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )
    def forward(self, X):
        return self.block(X)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
    
