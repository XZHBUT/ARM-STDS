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
    """
    使用networkx生成一个Erdős-Rényi随机图，然后转换为torch-geometric的格式。

    参数:
    - n_nodes: 图中节点的数量。
    - p: 任意两个节点之间形成边的概率。

    返回:
    - data: torch-geometric的图数据对象。
    """
    # 使用networkx生成随机图
    G = nx.erdos_renyi_graph(n_nodes, p)

    # 将networkx图转换为torch-geometric图数据对象
    data = from_networkx(G)

    return data.edge_index


# 假设data是已经生成的torch-geometric图数据对象
def Plot_Graph(data):
    """
    将torch-geometric的图数据对象转换为networkx图，并进行绘制。

    参数:
    - data: torch-geometric的图数据对象。
    """
    # 将torch-geometric图数据对象转换为networkx图
    G = to_networkx(data, to_undirected=True)

    # 绘制图
    pos = nx.spring_layout(G)  # 使用Spring布局
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

        # 将每个通道的向量表示为图的节点
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
        # 批处理所有的图
        batched_data = Batch.from_data_list(batch_data_list)
        # print(batched_data.x.shape)

        # 在大图上应用ARMAConv
        output = self.DirGNNConv(batched_data.x, batched_data.edge_index)



        return output.view(B, N, -1)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn((32, 8, 1024), device=device)
    model = GraphFeatureExtraction(node_n=8, Seq_len=1024).to(device)

    edge = Random_Build_Graph(n_nodes=8, p=1)
    print(edge)

    # Plot_Graph(data=edge)
    out = model(x, edge, isRandomAt=True)
    print(out.shape)

    pool1 = GraphPooling(node_n=8)
    out1 = pool1(out)
    print(out1.shape)

    out2 = out1.squeeze()
    print(out2.shape)
