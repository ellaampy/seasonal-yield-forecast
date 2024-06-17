## adapted from 
## https://github.com/MarcCoru/BreizhCrops/blob/master/breizhcrops/models/LongShortTermMemory.py
from torch import dropout, nn
import torch.nn.functional as F
import torch

class LSTM(torch.nn.Module):
    def __init__(self, input_dim=15, num_classes=1, hidden_dims=128, num_layers=4, dropout=0.6, bidirectional=True, use_layernorm=True):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.use_layernorm = use_layernorm
        self.d_model = num_layers * hidden_dims

        if use_layernorm:
            self.inlayernorm = nn.LayerNorm(input_dim)
            self.clayernorm = nn.LayerNorm((hidden_dims + hidden_dims * bidirectional) * num_layers)

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims, num_layers=num_layers,
                            bias=False, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        if bidirectional:
            hidden_dims = hidden_dims * 2

        self.linear_class = nn.Linear(hidden_dims * num_layers, num_classes, bias=True)

    def forward(self, x):
        if self.use_layernorm:
            x = self.inlayernorm(x)

        outputs, last_state_list = self.lstm.forward(x)

        h, c = last_state_list

        nlayers, batchsize, n_hidden = c.shape
        h = self.clayernorm(c.transpose(0, 1).contiguous().view(batchsize, nlayers * n_hidden))
        logits = self.linear_class.forward(h)

        x =  x.squeeze(-1)
        return x

if __name__ == "__main__":
    _ = LSTM()
