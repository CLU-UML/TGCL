import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv,GATConv


class SAGE_w_feat(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, x, feat):
        super(SAGE_w_feat, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.h1_feat = nn.Linear(x.shape[-1], hidden_channels)
        self.convs.append(SAGEConv(hidden_channels + hidden_channels, out_channels))

        self.dropout = dropout
        self.add_feat = x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        add_feat = self.h1_feat(self.add_feat)  # replaced self.h1_feat(x)  with  self.h1_feat(self.add_feat)
        x = torch.cat([x, add_feat], dim=-1)

        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

class SAGE_no_feat(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, x, model_type):
        super(SAGE_no_feat, self).__init__()
        self.convs = torch.nn.ModuleList()

        if model_type == "sage":
            self.convs.append(SAGEConv(in_channels, hidden_channels))
        elif model_type == "gcn":
            self.convs.append(GCNConv(in_channels, hidden_channels))
        elif model_type == "gatconv":
            self.convs.append(GATConv(in_channels, hidden_channels))

        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):

            if model_type == "sage":
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            elif model_type == "gcn":
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            elif model_type == "gatconv":
                self.convs.append(GATConv(hidden_channels, hidden_channels))

            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        if model_type == "sage":
            self.convs.append(SAGEConv(hidden_channels, out_channels))
        elif model_type == "gcn":
            self.convs.append(GCNConv(hidden_channels, out_channels))
        elif model_type == "gatconv":
            self.convs.append(GATConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.add_feat = x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, adj_t)

        return x.log_softmax(dim=-1)

