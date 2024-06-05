
import networkx as nx


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_networkx

from torch.nn import init


class GraphicalStructureModeling(nn.Module):
    def __init__(self, in_channel, Seq_len, device):
        super().__init__()

        self.in_channel = in_channel
        self.Seq_len = Seq_len
        self.device = device

        self.W1 = nn.Parameter(torch.empty(self.in_channel, self.in_channel))
        self.W2 = nn.Parameter(torch.empty(self.Seq_len, self.in_channel))

        # 使用 Xavier 初始化
        init.xavier_uniform_(self.W1)
        init.xavier_uniform_(self.W2)

    def __build_edge__(self, At, Dis):
        count = At.size(1) - 2

        while count > 1:
            solo_to_clu_two, solo_to_clu_min, mask_clu, clu_to_solo_two, clu_to_solo_min, mask_solo, clu_to_min_solo_mask, D_clu_to_solo_min = self.__find_closest_clusters__(
                Dis, At)

            # 孤悬点向簇建立连接
            solo_to_clu_min_index = solo_to_clu_two[:, :1]
            solo_to_clu_max_values, max_indices = torch.max(solo_to_clu_min, dim=1)
            solo_to_clu_max_values = solo_to_clu_max_values.unsqueeze(1)  # 最近两个点平均值的最大值

            solo_to_clu_index_tensor = torch.zeros((At.size(0), At.size(1), At.size(2)), dtype=torch.float,
                                                   device=Dis.device)
            solo_to_clu_index_tensor[torch.arange(At.size(0)), solo_to_clu_min_index.squeeze(1), :] = 1
            solo_to_clu_index_tensor.permute(0, 2, 1)[mask_solo] = int(0)  # 最小孤悬值点到簇点的索引

            min_solo_to_clu_dis = Dis * solo_to_clu_index_tensor  # 获取到了最小孤悬点到所有簇点的距离

            ons = torch.ones((At.size(0), At.size(1), At.size(2)), dtype=torch.int, device=Dis.device)
            min_solo_to_clu_maxGG = ons * solo_to_clu_max_values.unsqueeze(2)
            min_solo_to_clu_dayumaxGG_index_all = min_solo_to_clu_dis - min_solo_to_clu_maxGG
            dayumaxGG_index = (min_solo_to_clu_dayumaxGG_index_all < 0).float()
            build_link_solo_to_clu = dayumaxGG_index * solo_to_clu_index_tensor
            At = At + build_link_solo_to_clu

            # 簇点向最近孤悬点建立连接
            clu_to_solo_max_values, clu_to_solo_max_indices = torch.max(clu_to_solo_min, dim=1)
            clu_to_solo_max_values = clu_to_solo_max_values.unsqueeze(1)  # 最近两个点平均值的最大值

            ons = torch.ones((At.size(0), At.size(1), At.size(2)), dtype=torch.int, device=Dis.device)
            min_clu_to_solo_maxGG = ons * clu_to_solo_max_values.unsqueeze(2)
            min_clu_to_solo_dayumaxGG_index_all = D_clu_to_solo_min - min_clu_to_solo_maxGG
            clu_dayumaxGG_index = (min_clu_to_solo_dayumaxGG_index_all < 0).float()  # 找到簇点到最近孤悬点小于maxGG的
            build_link_clu_to_solo = clu_dayumaxGG_index * clu_to_min_solo_mask
            At = At + build_link_clu_to_solo

            count -= 1

        # 处理剩下的结点
        solo_to_clu_last, solo_to_clu_min_last, mask_clu, clu_to_solo_two, clu_to_solo_min_last, mask_solo, clu_to_min_solo_mask_last, D_clu_to_solo_min_last \
            = self.__find_closest_clusters__(Dis, At, Last_Node=True)
        # 孤悬点向簇建立连接
        solo_to_clu_min_index_last = solo_to_clu_last[:, :1]
        solo_to_clu_max_values_last, max_indices = torch.max(solo_to_clu_min_last, dim=1)
        solo_to_clu_max_values_last = solo_to_clu_max_values_last.unsqueeze(1)  # 最近两个点平均值的最大值

        solo_to_clu_index_tensor_last = torch.zeros((At.size(0), At.size(1), At.size(2)), dtype=torch.float,
                                                    device=Dis.device)
        solo_to_clu_index_tensor_last[torch.arange(At.size(0)), solo_to_clu_min_index_last.squeeze(1), :] = 1
        solo_to_clu_index_tensor_last.permute(0, 2, 1)[mask_solo] = int(0)  # 最小孤悬值点到簇点的索引

        min_solo_to_clu_dis_last = Dis * solo_to_clu_index_tensor_last  # 获取到了最小孤悬点到所有簇点的距离

        ons = torch.ones((At.size(0), At.size(1), At.size(2)), dtype=torch.int, device=Dis.device)
        min_solo_to_clu_maxGG_last = ons * solo_to_clu_max_values_last.unsqueeze(2)
        min_solo_to_clu_dayumaxGG_index_all_last = min_solo_to_clu_dis_last - min_solo_to_clu_maxGG_last
        dayumaxGG_index_last = (min_solo_to_clu_dayumaxGG_index_all_last < 0).float()
        build_link_solo_to_clu_last = dayumaxGG_index_last * solo_to_clu_index_tensor_last
        At = At + build_link_solo_to_clu_last

        # 簇点向最近孤悬点建立连接
        clu_to_solo_max_values_last, clu_to_solo_max_indices_last = torch.max(clu_to_solo_min_last, dim=1)
        clu_to_solo_max_values_last = clu_to_solo_max_values_last.unsqueeze(1)  # 最近两个点平均值的最大值

        ons = torch.ones((At.size(0), At.size(1), At.size(2)), dtype=torch.int, device=Dis.device)
        min_clu_to_solo_maxGG_last = ons * clu_to_solo_max_values_last.unsqueeze(2)
        min_clu_to_solo_dayumaxGG_index_all_last = D_clu_to_solo_min_last - min_clu_to_solo_maxGG_last
        clu_dayumaxGG_index_last = (min_clu_to_solo_dayumaxGG_index_all_last < 0).float()  # 找到簇点到最近孤悬点小于maxGG的
        build_link_clu_to_solo_last = clu_dayumaxGG_index_last * clu_to_min_solo_mask_last
        At = At + build_link_clu_to_solo_last

        return At

    def __find_closest_clusters__(self, Dis, At, Last_Node=False):
        # 假设 At（2，5，5） Dis At（2，5，5）

        # 如果图里面只剩一个孤悬点
        if Last_Node:
            # 最后一个孤悬点
            degree = At.sum(dim=2) + At.sum(dim=1) - At.diagonal(dim1=1, dim2=2)

            # 确定孤悬点的索引
            isolated_nodes_last = (degree == 0).float()

            D_solo_to_clu_last = Dis.clone()
            mask_last = isolated_nodes_last == 0
            mask1_last = isolated_nodes_last == 1
            D_solo_to_clu_last[mask_last] = float('inf')
            D_solo_to_clu_last.permute(0, 2, 1)[mask1_last] = float('inf')
            row_sums_last = torch.sum(
                torch.where(torch.isfinite(D_solo_to_clu_last), D_solo_to_clu_last, torch.tensor(0.)), dim=-1)
            valid_counts_last = torch.sum(torch.isfinite(D_solo_to_clu_last), dim=-1)
            row_means_last = row_sums_last / valid_counts_last.float()
            # 使用 torch.argsort 获取排序索引
            sorted_indices_row_last = torch.argsort(row_means_last, dim=1)
            # 选择每行的前第一个索引，也就是唯一的一个的位置
            min_indices_row_last = sorted_indices_row_last[:, :1]
            # 使用索引提取对应的值
            min_values_row_last = torch.gather(row_means_last, 1, min_indices_row_last)

            # 簇到孤悬
            D_clu_to_solo_last = Dis.clone()
            D_clu_to_solo_last.permute(0, 2, 1)[mask_last] = float('inf')
            D_clu_to_solo_last[mask1_last] = float('inf')

            # 指向最近孤悬值得列索引
            clu_to_min_solo_mask_last = torch.zeros((At.size(0), At.size(1), At.size(2)), dtype=torch.float,
                                                    device=Dis.device)
            clu_to_min_solo_mask_last[torch.arange(At.size(0)), :, min_indices_row_last[:, :1].squeeze(1)] = 1
            clu_to_min_solo_mask_last[mask1_last] = 0

            # 簇所有点到最近一个孤悬点得距离
            D_clu_to_solo_min_last = D_clu_to_solo_last.clone()

            min_solo_mask_True_last = (clu_to_min_solo_mask_last == 1).float()
            D_clu_to_solo_min_last[min_solo_mask_True_last == 0] = float('inf')

            # 簇内所有点到所有孤悬点得平均距离，找最近得两个以及平均值
            row_sums_last = torch.sum(
                torch.where(torch.isfinite(D_clu_to_solo_last), D_clu_to_solo_last, torch.tensor(0.)), dim=-1)
            valid_counts_last = torch.sum(torch.isfinite(D_clu_to_solo_last), dim=-1)
            row_means_last = row_sums_last / valid_counts_last.float()

            # 使用 torch.argsort 获取排序索引
            sorted_indices_l_last = torch.argsort(row_means_last, dim=1)
            # 选择每行的第一个索引，唯一一个
            min_indices_l_last = sorted_indices_l_last[:, :2]
            # 使用索引提取对应的值
            min_values_l_last = torch.gather(row_means_last, 1, min_indices_l_last)

            return min_indices_row_last, min_values_row_last, mask_last, min_indices_l_last, min_values_l_last, mask1_last, clu_to_min_solo_mask_last, D_clu_to_solo_min_last

        # 计算每个节点的度
        degree = At.sum(dim=2) + At.sum(dim=1) - At.diagonal(dim1=1, dim2=2)
        # degree torch.Size([2, 5])
        # 确定孤悬点的索引
        isolated_nodes = (degree == 0).float()
        mask1 = isolated_nodes == 0  # 簇点掩码
        mask2 = isolated_nodes == 1  # 孤悬点掩码
        # isolated_nodes mask1 mask2  torch.Size([2, 5])

        # 计算 每个孤悬点到簇内所有点的平均距离
        D_solo_to_clu = Dis.clone()
        D_solo_to_clu[mask1] = float('inf')  # 簇到其他点的距离设置为inf
        D_solo_to_clu.permute(0, 2, 1)[mask2] = float('inf')  # 孤悬点到自己的距离设置为inf
        solo_to_clu_sums = torch.sum(torch.where(torch.isfinite(D_solo_to_clu), D_solo_to_clu, torch.tensor(0.)),
                                     dim=-1)
        valid_counts = torch.sum(torch.isfinite(D_solo_to_clu), dim=-1)
        solo_to_clu_means = solo_to_clu_sums / valid_counts.float()  # 每个孤悬点到簇的平均距离
        # solo_to_clu_means torch.Size([2, 5])

        # 获取孤悬点到簇的平均距离最小的两个点的索引位置以及数值
        solo_to_clu_sort = torch.argsort(solo_to_clu_means, dim=1)  # 获取排序索引
        solo_to_clu_two = solo_to_clu_sort[:, :2]  # 前两个
        solo_to_clu_min = torch.gather(solo_to_clu_means, 1, solo_to_clu_two)  # 按照索引获取值
        # solo_to_clu_min torch.Size([2, 2])

        # 计算 簇内每个点到所有孤悬点的平均距离       #######################################################
        D_clu_to_solo = Dis.clone()
        D_clu_to_solo.permute(0, 2, 1)[mask1] = float('inf')  # 孤悬点到其他点的距离设置为inf
        D_clu_to_solo[mask2] = float('inf')  # 簇点到自己的距离设置为inf

        # 指向最近孤悬值得列索引
        clu_to_min_solo_mask = torch.zeros((At.size(0), At.size(1), At.size(2)), dtype=torch.float, device=Dis.device)
        clu_to_min_solo_mask[torch.arange(At.size(0)), :, solo_to_clu_two[:, :1].squeeze(1)] = 1
        clu_to_min_solo_mask[mask2] = 0

        # 簇所有点到最近一个孤悬点得距离
        D_clu_to_solo_min = D_clu_to_solo.clone()
        min_solo_mask_True = (clu_to_min_solo_mask == 1).float()
        D_clu_to_solo_min[min_solo_mask_True == 0] = float('inf')

        # 簇内所有点到所有孤悬点得平均距离，找最近得两个以及平均值
        clu_to_solo_sums = torch.sum(torch.where(torch.isfinite(D_clu_to_solo), D_clu_to_solo, torch.tensor(0.)),
                                     dim=-1)
        valid_counts = torch.sum(torch.isfinite(D_clu_to_solo), dim=-1)
        clu_to_solo_means = clu_to_solo_sums / valid_counts.float()  # 每个簇点到孤悬的平均距离
        # clu_to_solo_means torch.Size([2, 5])

        # 获取簇点到孤悬点的平均距离最小的两个点的索引位置以及数值
        clu_to_solo_sort = torch.argsort(clu_to_solo_means, dim=1)  # 获取排序索引
        clu_to_solo_two = clu_to_solo_sort[:, :2]  # 前两个
        clu_to_solo_min = torch.gather(clu_to_solo_means, 1, clu_to_solo_two)  # 按照索引获取值
        # clu_to_solo_min torch.Size([2, 2])

        """
        solo_to_clu_two     孤悬点到簇最近的两个的位置
        solo_to_clu_min      孤悬点到簇最近的两个的值
        mask1              孤悬点的索引
        clu_to_solo_two       簇到所有孤悬点最近的两个的位置
        clu_to_solo_min        簇到所有孤悬点最近的两个的值
        mask2               簇点的索引
        clu_to_min_solo_mask    簇每个点到最小孤悬点的索引
        D_clu_to_solo_min   簇每个点到最小孤悬点的值
        """

        return solo_to_clu_two, solo_to_clu_min, mask1, clu_to_solo_two, clu_to_solo_min, mask2, clu_to_min_solo_mask, D_clu_to_solo_min

    def __init_At(self, Dis):
        B, N, _ = Dis.shape
        # 假如Dis (2,N,N)
        average_dis = (Dis + Dis.transpose(1, 2)) / 2
        #  获取对角线的索引 把对角线设置成最大 避免初始化成闭环
        indices = torch.arange(average_dis.size(1), device=Dis.device)
        average_dis[:, indices, indices] = float('inf')
        # 使用torch.min找到最小值及其索引
        min_values, min_indices = torch.min(average_dis.view(average_dis.size(0), -1), dim=1)
        # 将一维索引转换为二维索引
        row_indices = min_indices // average_dis.size(2)
        col_indices = min_indices % average_dis.size(2)
        min_positions = torch.stack((row_indices, col_indices), dim=1)  # 最小值坐标
        # min_positions torch.Size([2, 2])

        # 初始化一个全0的连接矩阵
        At = torch.zeros((B, N, N), device=Dis.device)
        # 按照坐标将位置为1，来去两条边
        At[torch.arange(B), min_positions[:, 0], min_positions[:, 1]] = 1
        At[torch.arange(B), min_positions[:, 1], min_positions[:, 0]] = 1

        return At

    def __Random_Build_At(self, n_nodes, p):
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

    def forward(self, x, isRandomAt=False, p=1):

        if isRandomAt:
            At = self.__Random_Build_At(n_nodes=self.in_channel, p=p)
            return x, At
        else:
            node = x.contiguous()
            term1 = torch.tanh((self.W1.matmul(node)).matmul(self.W2))
            term2 = torch.tanh(((self.W2.T).matmul(node.permute(0, 2, 1))).matmul(self.W1.T))

            sim = F.softplus(term1 - term2)  # 相似矩阵

            diag_elements = sim.diagonal(dim1=1, dim2=2)
            Dis = torch.sqrt(F.relu(diag_elements.unsqueeze(2) + diag_elements.unsqueeze(1) - sim))  # 距离矩阵

            At = self.__init_At(Dis)  # 初始化一个全0的连接矩阵，但是有两个结点前后相连为1
            At = self.__build_edge__(At, Dis)  # 结点邻接矩阵



            return x, At


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn((64, 8, 1024), device=device)
    model = GraphicalStructureModeling(in_channel=8, Seq_len=1024, device=device).to(device)

    x, At = model(x,isRandomAt=True)

    print(At[0])
