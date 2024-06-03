import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.pool import global_mean_pool


class GAT(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads, dropout=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        # Layers
        self.layers = []

        # Input layer
        self.layers.append(GATConv(self.in_channels, self.hidden_channels,
                                   heads=self.heads, dropout=self.dropout))

        # Hidden Layers
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(self.heads * self.hidden_channels,
                                       self.hidden_channels, heads=self.heads,
                                       dropout=self.dropout))
        # Output Layer
        self.linear = nn.Linear(self.heads * self.hidden_channels,
                                self.out_channels, bias=True)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)

        # Merge the heads        
        x = self.layers[-1](x)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)
