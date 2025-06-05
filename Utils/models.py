import torch
import torch.nn as nn
from torch_geometric.nn import AttentiveFP,GCNConv,GATv2Conv,TransformerConv,global_mean_pool as gmp


class AttentiveFPModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps=8,num_layers=2, dropout=0.2):
        super(AttentiveFPModel, self).__init__()
        self.attentive_fp = AttentiveFP(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_timesteps=num_timesteps,
            edge_dim=14,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, data):
        x, edge_index,edge_attr, batch = data.x, data.edge_index,data.edge_attr, data.batch
        x = self.attentive_fp(x, edge_index, edge_attr, batch)
        return x.squeeze(-1)
                    
class GCNModel(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
    super(GCNModel, self).__init__()
    self.convs = torch.nn.ModuleList()
    self.convs.append(GCNConv(input_dim, hidden_dim))
    for _ in range(num_layers - 1):
      self.convs.append(GCNConv(hidden_dim, hidden_dim))
    self.fc = nn.Linear(hidden_dim, output_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch
    for conv in self.convs:
      x = conv(x, edge_index)
      x = torch.relu(x)
      x = self.dropout(x)
    x = gmp(x, batch)
    x = self.fc(x)
    return x.squeeze(-1)

class GATModel(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2, heads=1
    ):
        super(GATModel, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(input_dim, hidden_dim, heads=heads, edge_dim=14))
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=14)
            )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = torch.relu(x)
            x = self.dropout(x)
        x = gmp(x, batch)  # Global pooling
        x = self.fc(x)
        return x.squeeze(-1)

class TransformerModel(nn.Module):
  def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2, heads=1
    ):
        super(TransformerModel, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            TransformerConv(input_dim, hidden_dim, heads=heads, edge_dim=14)
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                TransformerConv(
                    hidden_dim * heads, hidden_dim, heads=heads, edge_dim=14
                )
            )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        self.dropout = nn.Dropout(dropout)

  def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        edge_attr = edge_attr.float()
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = torch.relu(x)
            x = self.dropout(x)
        x = gmp(x, batch)  # Global pooling
        x = self.fc(x)
        return x.squeeze(-1)