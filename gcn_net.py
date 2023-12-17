"""
Multi-layer Graph Convolutional Network

Author: Tanay Tuncer 
"""

import numpy as np
import torch
from torch import manual_seed
from torch.nn import Linear, ReLU, Module
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GraphConvNeuralNet(Module):

    def __init__(self, data, n_hidden_channels):
        super(GraphConvNeuralNet, self).__init__()
        manual_seed(2023)
        self.n_features = data.num_features
        self.n_classes = len(data.y.unique())
        self.conv1 = GCNConv(in_channels = self.n_features, out_channels = n_hidden_channels)
        self.conv2 = GCNConv(in_channels = n_hidden_channels, out_channels = self.n_classes) 

    def forward(self, x, edge_idx):
        x = F.dropout(x, p = 0.5, training = self.training)
        x = self.conv1(x, edge_idx)
        x = torch.relu(x)
        x = F.dropout(x, p = 0.5, training = self.training)
        x = self.conv2(x, edge_idx)
        x = F.log_softmax(x, dim = 1)
        return x
     