import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
from allennlp.nn import util
from transformers import BertModel, XLNetModel, XLNetForSequenceClassification, BertForSequenceClassification
from transformers import BertConfig


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=False)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r, inplace=False)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class FFN(nn.Module):
    def __init__(self, hidden_size, ff_size, dropout):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=ff_size,
            out_size=hidden_size,
            dropout_r=dropout,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MHAtt(nn.Module):
    def __init__(self, head_num, hidden_size, dropout, hidden_size_head):
        super(MHAtt, self).__init__()
        self.head_num = head_num
        self.hidden_size = hidden_size
        self.hidden_size_head = hidden_size_head
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_merge = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout, inplace=False)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.head_num,
            self.hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.head_num,
            self.hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.head_num,
            self.hidden_size_head
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e4)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class SA(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head):
        super(SA, self).__init__()

        self.mhatt = MHAtt(head_num, hidden_size, dropout, hidden_size_head)
        self.ffn = FFN(hidden_size, ff_size, dropout)

        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, x, x_mask):
        output = self.mhatt(x, x, x, x_mask)
        dropout_output = self.dropout1(output)
        x = self.norm1(x + dropout_output)

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


class GA(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head):
        super(GA, self).__init__()

        self.mhatt = MHAtt(head_num, hidden_size, dropout, hidden_size_head)
        self.ffn = FFN(hidden_size, ff_size, dropout)

        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, x, y, y_mask, x_mask=None):
        if x_mask is None:
            intermediate = self.dropout1(self.mhatt(y, y, x, y_mask))
        else:
            intermediate = self.dropout1(self.mhatt(y, y, x, y_mask)) * x_mask.unsqueeze(-1)

        x = self.norm1(x + intermediate)
        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


class SAEncoder(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head, layers):
        super(SAEncoder, self).__init__()
        self.encoders = nn.ModuleList([SA(hidden_size=hidden_size, head_num=head_num, ff_size=ff_size,
                                          dropout=dropout, hidden_size_head=hidden_size // head_num) for _ in range(layers)])

    def forward(self, x, x_mask=None):
        for layer in self.encoders:
            x = layer(x, x_mask)
        return x


class GAEncoder(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head, layers):
        super(GAEncoder, self).__init__()
        self.encoders = nn.ModuleList([GA(hidden_size=hidden_size, head_num=head_num, ff_size=ff_size,
                                          dropout=dropout, hidden_size_head=hidden_size // head_num) for _ in range(layers)])

    def forward(self, x, y, y_mask, x_mask=None):
        for layer in self.encoders:
            x = layer(x, y, y_mask, x_mask)
        return x


class CREncoder(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head, layers):
        super(CREncoder, self).__init__()
        self.embedding = nn.Embedding(2, hidden_size)
        self.encoders = nn.ModuleList([SA(hidden_size=hidden_size, head_num=head_num, ff_size=ff_size,
                                          dropout=dropout, hidden_size_head=hidden_size // head_num) for _ in range(layers)])

    def forward(self, x, x_mask, y, y_mask):
        t1_mask = torch.cat([x_mask, y_mask], -1)
        t2_mask = torch.cat([x_mask, torch.zeros_like(y_mask)], -1)
        mask = torch.cat([t1_mask.unsqueeze(1).repeat(1, x.shape[1], 1),
                          t2_mask.unsqueeze(1).repeat(1, y.shape[1], 1)], 1)

        emb_x = torch.zeros(x.shape[0], x.shape[1]).long().to(x.device)
        emb_y = torch.ones(y.shape[0], y.shape[1]).long().to(y.device)
        emb_x = self.embedding(emb_x)
        emb_y = self.embedding(emb_y)

        emb_representation = torch.cat([emb_x, emb_y], 1)

        representation = torch.cat([x, y], 1) + emb_representation

        for layer in self.encoders:
            representation = layer(representation, (1 - mask).unsqueeze(1).bool())

        return representation


class NumGNN(nn.Module):

    def __init__(self, node_dim, iteration_steps=1):
        super(NumGNN, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim, 1, bias=True)
        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._dd_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._dd_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)

    def forward(self, d_node, greater_graph, smaller_graph):
        d_node_len = d_node.size(1)

        dd_graph_left = greater_graph  # [:, :d_node_len, :d_node_len]
        dd_graph_right = smaller_graph  # [:, :d_node_len, :d_node_len]

        d_node_neighbor_num = dd_graph_left.sum(-1) + dd_graph_right.sum(-1)
        d_node_neighbor_num_mask = (d_node_neighbor_num >= 1).long()
        d_node_neighbor_num = util.replace_masked_values(d_node_neighbor_num.float(), d_node_neighbor_num_mask, 1)

        for step in range(self.iteration_steps):
            d_node_weight = torch.sigmoid(self._node_weight_fc(d_node)).squeeze(-1)

            self_d_node_info = self._self_node_fc(d_node)

            dd_node_info_left = self._dd_node_fc_left(d_node)

            dd_node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                dd_graph_left,
                0)

            dd_node_info_left = torch.matmul(dd_node_weight, dd_node_info_left)

            dd_node_info_right = self._dd_node_fc_right(d_node)

            dd_node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                dd_graph_right,
                0)

            dd_node_info_right = torch.matmul(dd_node_weight, dd_node_info_right)

            agg_d_node_info = (dd_node_info_left + dd_node_info_right) / d_node_neighbor_num.unsqueeze(-1)

            d_node = F.relu(self_d_node_info + agg_d_node_info)

        return d_node


class TableEncoder(nn.Module):

    def __init__(self, dim, head, model_type, layers=4, dropout=0.1):
        super(TableEncoder, self).__init__()
        self.BASE = BertModel.from_pretrained(model_type)
        self.ROW = SAEncoder(dim, head, 4 * dim, dropout, dim // head, 3)
        self.COL = NumGNN(dim)
        self.FUSION = GAEncoder(dim, head, 4 * dim, dropout, dim // head, 3)
        self.CLASSIFIER = nn.Linear(dim, 2)

    def forward(self, forward_type, *args):
        if forward_type == 'cell':
            return self.BASE(*args)
        elif forward_type == 'row':
            return self.ROW(*args)
        elif forward_type == 'col':
            return self.COL(*args)
        else:
            x = self.FUSION(*args)
            x = self.CLASSIFIER(x)
            return x

class GNN(nn.Module):
    def __init__(self, dim, head, model_type, config, label_num, layers=3, dropout=0.1, attention='self'):
        super(GNN, self).__init__()
        self.BASE = BertModel.from_pretrained(model_type, config=config, from_tf=False, cache_dir='tmp/')
        if attention == 'self':
            self.biflow = SAEncoder(dim, head, 4 * dim, dropout, dim // head, layers)
        else:
            self.biflow = CREncoder(dim, head, 4 * dim, dropout, dim // head, layers)
        self.gnn = NumGNN(dim)
        self.embedding = nn.Embedding(13, dim)
        self.classifier = nn.Linear(dim, label_num)

    def forward(self, forward_type, **kwargs):
        if forward_type == 'row':
            outputs = self.BASE(**kwargs)
            return outputs
        elif forward_type == 'gnn':
            outputs = self.gnn(**kwargs)
            return outputs
        elif forward_type == 'emb':
            outputs = self.embedding(kwargs['x'])
            return outputs
        elif forward_type == 'sa':
            outputs = self.biflow(**kwargs)
            return self.classifier(outputs[:, 0])
        else:
            raise NotImplementedError
