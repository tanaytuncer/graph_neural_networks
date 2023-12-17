"""
Author: Tanay Tunçer
Graph Neural Network - Graph Attention Layer

"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import manual_seed
import numpy as np
#import tqdm

class GraphAttentionLayer(nn.Module):

    def __init__(self, inp_dim, out_dim, n_att_heads = 4, concat = True, dropout_rate = 0.6, dyanamic_attention = True):
        super().__init__()
        manual_seed(2023)
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.n_att_heads = n_att_heads
        self.concat = concat
        self.dyanamic_attention = dyanamic_attention

        if concat:
            assert out_dim % n_att_heads == 0
            self.n_hidden_channels = int(out_dim / n_att_heads)
        else:
            self.n_hidden_channels = out_dim
        
        self.W = nn.Linear(self.inp_dim, self.n_hidden_channels * n_att_heads, bias = False)
        self.a = nn.Linear(self.n_hidden_channels, 1, bias = False)  

        init.kaiming_uniform_(self.W.weight)
        init.kaiming_uniform_(self.a.weight)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout_rate)      

    def forward(self, x, adj_mat):

        n_nodes = x.size()[0]
        
        Wx = self.W(x).view(n_nodes, self.n_att_heads, self.n_hidden_channels)              
        Wx_i = Wx.repeat(n_nodes, 1, 1)
        Wx_j = Wx.repeat_interleave(n_nodes, dim=0)

        Wx_ij = (Wx_i + Wx_j).view(n_nodes, n_nodes, self.n_att_heads, self.n_hidden_channels) 

        if self.dyanamic_attention:
            e_ij = self.a(self.leaky_relu(Wx_ij)).squeeze(-1)
        else:
            e_ij = self.leaky_relu(self.a(Wx_ij)).squeeze(-1)
        
        e_ij = e_ij.masked_fill(adj_mat == 0, float('-inf'))
        a_ij = self.softmax(e_ij)
        a_ij = self.dropout(a_ij) 

        xh = torch.einsum('ijh,jhf->ihf', a_ij, Wx)

        if self.concat:
            return xh.reshape(n_nodes, self.n_att_heads * self.n_hidden_channels)
        else:
            return xh.mean(dim = 1)

            
        

        

    