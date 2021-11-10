
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, output_him, num_layers=5, bias=False, dropout=0.1, activation=F.relu):
        super(GCNEncoder, self).__init__()
        """encoder 1"""
        # self.encoder = nn.Linear(in_dim, hidden_dim)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, int(in_dim / 2)),
            torch.nn.BatchNorm1d(int(in_dim / 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(in_dim / 2), hidden_dim)
        )

        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()

        self.gcn_layers.append(dglnn.GraphConv(hidden_dim, hidden_dim, bias=bias, norm='both', activation=activation))    #allow_zero_in_degree=True,
        # hidden layers
        for l in range(2, num_layers):
            self.gcn_layers.append(dglnn.GraphConv(hidden_dim, hidden_dim, bias=bias, norm='both', activation=activation))
        self.gcn_layers.append(dglnn.GraphConv(hidden_dim, output_him, bias=bias, norm='both'))
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        # torch.nn.init.normal_(self.encoder.weight, std=0.01)
        for m in self.encoder.children():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.01)
        for i in range(self.num_layers):
            torch.nn.init.xavier_uniform_(self.gcn_layers[i].weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, g, h):
        h = self.encoder(h)
        all_layers = []
        all_layers.append(h)
        for i, layer in enumerate(self.gcn_layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
            all_layers.append(h)
        # return h
        all_layers = torch.stack(all_layers, dim=1)
        h = torch.mean(all_layers, 1)
        return h

