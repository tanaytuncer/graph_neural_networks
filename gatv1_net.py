import torch
import torch.nn as nn
import torch.nn.functional as F

from gat import GraphAttentionLayer

class GatNetv1(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim 
        self.out_dim = out_dim 
        self.gat_conv1 = GraphAttentionLayer(in_dim, 64, n_att_heads = 8, concat = True, dyanamic_attention = False)
        self.gat_conv2 = GraphAttentionLayer(64, out_dim, n_att_heads = 1, concat = False, dyanamic_attention = False)

    def forward(self, data):
        
        x = data.x
        adj_mat = torch.zeros((data.num_nodes, data.num_nodes))
        adj_mat[data.edge_index[0], data.edge_index[1]] = 1
        adj_mat[data.edge_index[1], data.edge_index[0]] = 1
        adj_mat = adj_mat.unsqueeze(2)

        x = F.dropout(x, p = 0.6, training=self.training)
        x = self.gat_conv1(x, adj_mat)
        x = F.elu(x)
        x = F.dropout(x, p = 0.6, training=self.training)
        x = self.gat_conv2(x, adj_mat)
        x = F.log_softmax(x, dim = 1)

        return x