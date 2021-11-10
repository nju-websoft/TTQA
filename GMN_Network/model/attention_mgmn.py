
import torch
import torch.nn as nn
import torch.nn.functional as functional


class MultiLevelAttention(torch.nn.Module):

    def __init__(self, perspectives=128):
        super(MultiLevelAttention, self).__init__()
        print('Node-Graph Matching: MultiLevelAttention')
        # ---------- Node-Graph Matching ----------
        self.perspectives = perspectives  # number of perspectives for multi-perspective matching function
        self.mp_w = nn.Parameter(torch.rand(self.perspectives, self.perspectives))  # trainable weight matrix for multi-perspective matching function

    @staticmethod
    def div_with_small_value(n, d, eps=1e-8):
        # too small values are replaced by 1e-8 to prevent it from exploding.
        d = d * (d > eps).float() + eps * (d <= eps).float()
        return n / d

    def cosine_attention(self, v1, v2):
        """
        :param v1: (batch, len1, dim)
        :param v2: (batch, len2, dim)
        :return:  (batch, len1, len2)
        """
        # (batch, len1, len2)  torch.transpose(y, 1, 0)
        # a = torch.bmm(v1, v2.permute(1, 0))
        a = torch.mm(v1, v2.permute(1, 0))
        return a
        # v1_norm = v1.norm(p=2, dim=1, keepdim=True)  # (batch, len1, 1)
        # v2_norm = v2.norm(p=2, dim=1, keepdim=True).permute(1,0)  # (batch, len2, 1)
        # d = v1_norm * v2_norm
        # return self.div_with_small_value(a, d)

    def multi_perspective_match_func(self, v1, v2, w):
        """
        :param v1: (batch, len, dim)
        :param v2: (batch, len, dim)
        :param w: (perspectives, dim)
        :return: (batch, len, perspectives)
        """
        w = w.transpose(1, 0).unsqueeze(0)  # (1,  dim, perspectives)
        v1 = w * torch.stack([v1] * self.perspectives, dim=2)  # (len, dim, perspectives)
        v2 = w * torch.stack([v2] * self.perspectives, dim=2)  # (len, dim, perspectives)
        return functional.cosine_similarity(v1, v2, dim=1)  # (batch, len, perspectives)

    def forward(self, feats, graph_idx, n_graphs):
        partitions = []
        for i in range(n_graphs):
            partitions.append(feats[graph_idx == i, :])

        multi_feats_list = []
        for i in range(0, n_graphs, 2):
            feature_p = partitions[i]
            feature_h = partitions[i + 1]
            attention = self.cosine_attention(feature_p, feature_h)  # (batch, len_p, len_h)
            attention_p = torch.mm(attention, feature_h)  # 7*128
            attention_h = torch.mm(torch.transpose(attention, 1, 0), feature_p)  #3*128
            multi_p = self.multi_perspective_match_func(v1=feature_p, v2=attention_p, w=self.mp_w)
            multi_h = self.multi_perspective_match_func(v1=feature_h, v2=attention_h, w=self.mp_w)
            multi_feats_list.append(multi_p)
            multi_feats_list.append(multi_h)
        multi_feats = torch.cat(multi_feats_list, dim=0)

        return multi_feats

