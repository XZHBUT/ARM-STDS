import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np

from math import sqrt

import networkx as nx
from matplotlib import pyplot as plt
from torch_geometric.utils import from_networkx
from torch_geometric.utils import from_networkx, to_networkx

import torch_geometric
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from torch_geometric.data import Batch


def Random_Build_Graph(n_nodes, p):


    G = nx.erdos_renyi_graph(n_nodes, p)


    data = from_networkx(G)

    return data.edge_index



def Plot_Graph(data):


    G = to_networkx(data, to_undirected=True)


    pos = nx.spring_layout(G)  
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='k', node_size=500, alpha=0.7)
    plt.show()

class GraphPooling(nn.Module):
    def __init__(self,  node_n, device='cuda:0'):
        super().__init__()
        self.node_n = node_n
        self.device = device

    def forward(self, x):
        B, N, L = x.shape
        node = x
        batch_out_list = []
        for i in range(B):
            node_i = node[i, :, :]
            out = torch_geometric.nn.global_mean_pool(x=node_i, batch=torch.tensor([0]*self.node_n).to(self.device))
            batch_out_list.append(out)
        batch_out_list = torch.stack(batch_out_list, dim=0)
        return batch_out_list

class GraphFeatureExtraction(nn.Module):
    def __init__(self, Seq_len, node_n, GNN_head=1, dropout=0.2, device='cuda:)'):
        super().__init__()

        # self.ChebConv = torch_geometric.nn.GraphSAGE(in_channels=Seq_len, out_channels=int(Seq_len/2), heads=GNN_head,
        #                                             dropout=dropout, K=1, hidden_channels=Seq_len, num_layers=1)

        self.ChebConv = torch_geometric.nn.ChebConv(in_channels=Seq_len, out_channels=int(Seq_len/2), heads=GNN_head,
                                                    dropout=dropout, K=1)

        # self.ChebConv = torch_geometric.nn.TransformerConv(in_channels=Seq_len, out_channels=int(Seq_len / 2),heads=1,
        #                                             dropout=dropout)
        #
        # self.ChebConv = torch_geometric.nn.GATConv(in_channels=Seq_len, out_channels=int(Seq_len / 2),heads=1,
        #                                             dropout=dropout)

        # self.ChebConv = torch_geometric.nn.GCNConv(in_channels=Seq_len, out_channels=int(Seq_len / 2),heads=1,
        #                                             dropout=dropout)


        self.DirGNNConv = torch_geometric.nn.DirGNNConv(conv=self.ChebConv)

        self.Seq_len = Seq_len
        self.node_n = node_n
        self.device = device

    def forward(self, x, At, isRandomAt=False):
        B, N, L = x.shape
        # node = x.transpose(2, 1).contiguous()

        node = x


        batch_data_list = []
        for i in range(B):
            node_i = node[i, :, :]
            if isRandomAt:
                data = torch_geometric.data.Data(x=node_i,
                                                 edge_index=At.to(x.device))
                batch_data_list.append(data)
            else:
                edge_index_i = At[i].nonzero(as_tuple=False).t().contiguous()
                data = torch_geometric.data.Data(x=node_i,
                                                 edge_index=edge_index_i.to(x.device))
                batch_data_list.append(data)

        batched_data = Batch.from_data_list(batch_data_list)



        output = self.DirGNNConv(batched_data.x, batched_data.edge_index)



        return output.view(B, N, -1)


