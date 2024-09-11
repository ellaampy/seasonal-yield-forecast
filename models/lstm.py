## adapted from 
## https://github.com/MarcCoru/BreizhCrops/blob/master/breizhcrops/models/LongShortTermMemory.py
from torch import dropout, nn
import torch.nn.functional as F
import torch

class LSTM(torch.nn.Module):
    def __init__(self, input_dim=15, num_classes=1, hidden_dims=128, num_layers=4, dropout=0.5713020228087161, bidirectional=True, use_layernorm=True):
        self.modelname = f"LSTM_input-dim={input_dim}_num-classes={num_classes}_hidden-dims={hidden_dims}_" \
                         f"num-layers={num_layers}_bidirectional={bidirectional}_use-layernorm={use_layernorm}" \
                         f"_dropout={dropout}"

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


    def logits(self, x):

        if self.use_layernorm:
            x = self.inlayernorm(x)

        outputs, last_state_list = self.lstm.forward(x)

        h, c = last_state_list

        nlayers, batchsize, n_hidden = c.shape
        h = self.clayernorm(c.transpose(0, 1).contiguous().view(batchsize, nlayers * n_hidden))
        logits = self.linear_class.forward(h)

        return logits

    def forward(self, x):
        x = self.logits(x)
        x =  x.squeeze(-1)
        return x





class LSTM_branched(torch.nn.Module):
    def __init__(self, input_dim1=15, input_dim2=4, num_classes=1, hidden_dims=128, num_layers=4, dropout=0.5713020228087161, bidirectional=True, use_layernorm=True):
        self.modelname = f"LSTM_input-dim1={input_dim1}_input-dim2={input_dim2}num-classes={num_classes}_hidden-dims={hidden_dims}_" \
                         f"num-layers={num_layers}_bidirectional={bidirectional}_use-layernorm={use_layernorm}" \
                         f"_dropout={dropout}"

        super(LSTM_branched, self).__init__()

        self.num_classes = num_classes
        self.use_layernorm = use_layernorm
        self.d_model = num_layers * hidden_dims

        if use_layernorm:
            self.inlayernorm1 = nn.LayerNorm(input_dim1)
            self.clayernorm1= nn.LayerNorm((hidden_dims + hidden_dims * bidirectional) * num_layers)
            self.inlayernorm2 = nn.LayerNorm(input_dim2)
            self.clayernorm2= nn.LayerNorm((hidden_dims + hidden_dims * bidirectional) * num_layers)

        self.lstm1 = nn.LSTM(input_size=input_dim1, hidden_size=hidden_dims, num_layers=num_layers,
                            bias=False, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        
        self.lstm2 = nn.LSTM(input_size=input_dim2, hidden_size=hidden_dims, num_layers=num_layers,
                            bias=False, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        if bidirectional:
            hidden_dims = hidden_dims * 2

        self.linear_class_ = nn.Linear(hidden_dims * num_layers, 32, bias=True)
        self.linear_class = nn.Linear(32 * 2, num_classes, bias=True)



    def logits(self, x, type_lstm):

        if type_lstm == 'lstm1':
            if self.use_layernorm:
                x = self.inlayernorm1(x)
            outputs, last_state_list = self.lstm1.forward(x)

        elif type_lstm == 'lstm2':
            if self.use_layernorm:
                x = self.inlayernorm2(x)
            outputs, last_state_list = self.lstm2.forward(x)

        h, c = last_state_list
        nlayers, batchsize, n_hidden = c.shape

        if type_lstm == 'lstm1':
            h = self.clayernorm1(c.transpose(0, 1).contiguous().view(batchsize, nlayers * n_hidden))
        elif type_lstm=='lstm2':
            h = self.clayernorm2(c.transpose(0, 1).contiguous().view(batchsize, nlayers * n_hidden))
        logits = self.linear_class_.forward(h)

        return logits

    def forward(self, x):
        x0 = x[0]  #get observed
        x1 = x[1]  #get scm
        x0 = self.logits(x0, type_lstm='lstm1')
        x1 = self.logits(x1, type_lstm='lstm2')
        

        x = torch.concat([x0, x1], dim=1)
        x = self.linear_class(x)
        x =  x.squeeze(-1)
        return x
    


# if __name__ == "__main__":
#     _ = LSTM()


# model = LSTM_branched(input_dim1=15, input_dim2=4, num_classes=1, hidden_dims=128, num_layers=4, dropout=0.5)
# data1 = torch.rand((2,10,15))
# data2 = torch.rand((2,10,4))

## wrap data in a tuple
# data = (data1, data2)

# out = model(data)

