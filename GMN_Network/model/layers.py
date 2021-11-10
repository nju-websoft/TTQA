
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModule(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(MLPModule, self).__init__()
        self._in_dim = in_dim
        self._dropout = dropout

        self.lin0 = torch.nn.Linear(self._in_dim, self._in_dim // 2)
        nn.init.xavier_uniform_(self.lin0.weight.data)
        nn.init.zeros_(self.lin0.bias.data)

        self.lin1 = torch.nn.Linear(self._in_dim // 2, self._in_dim // 4)
        nn.init.xavier_uniform_(self.lin1.weight.data)
        nn.init.zeros_(self.lin1.bias.data)

        self.lin2 = torch.nn.Linear(self._in_dim // 4, self._in_dim // 8)
        nn.init.xavier_uniform_(self.lin2.weight.data)
        nn.init.zeros_(self.lin2.bias.data)

        self.lin3 = torch.nn.Linear(self._in_dim // 8, out_dim)
        nn.init.xavier_uniform_(self.lin3.weight.data)
        nn.init.zeros_(self.lin3.bias.data)

    def forward(self, scores):
        scores = F.relu(self.lin0(scores))
        scores = F.dropout(scores, p=self._dropout) #training=self.training
        scores = F.relu(self.lin1(scores))
        scores = F.dropout(scores, p=self._dropout)
        scores = F.relu(self.lin2(scores))
        scores = F.dropout(scores, p=self._dropout)
        # scores = torch.sigmoid(self.lin3(scores)).view(-1)
        # scores = torch.tanh(self.lin3(scores))
        scores = self.lin3(scores)
        return scores
