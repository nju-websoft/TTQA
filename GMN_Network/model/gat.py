import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch

class GATEncoder(nn.Module):

    def __init__(self, in_dim, hidden_dim, output_him, heads, num_layers=5, bias=False, dropout=0.1, activation=F.relu):
        super(GATEncoder, self).__init__()
        """encoder 1"""
        # self.encoder = nn.Linear(in_dim, hidden_dim)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, int(in_dim / 2)),
            torch.nn.BatchNorm1d(int(in_dim / 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(in_dim / 2), hidden_dim)
        )

        """progration layer"""
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(dglnn.GATConv(hidden_dim, hidden_dim, heads[0], bias=bias, activation=activation))
        # hidden layers
        for l in range(2, num_layers):  # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(dglnn.GATConv(hidden_dim * heads[l - 1], hidden_dim, heads[l], bias=bias, activation=activation))
        self.gat_layers.append(dglnn.GATConv(hidden_dim * heads[-2], output_him, heads[-1], bias=bias))
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        # torch.nn.init.normal_(self.encoder.weight, std=0.01)
        for m in self.encoder.children():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.01)

    def forward(self, g, h):
        h = self.encoder(h)
        for i in range(self.num_layers-1):
            h = self.gat_layers[i](g, h)
            h = h.flatten(1)
            h = self.dropout(h)
        h = self.gat_layers[-1](g, h)
        h = h.mean(1)
        return h

