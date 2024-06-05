import time
import math

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.nn import functional as F

import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms


import torch
from thop import profile

from data.DataProcess import Read_Data_From_SEU, Read_Data_From_HUST
from torch.utils.data import TensorDataset, DataLoader

from sklearn.manifold import TSNE


class MFE(nn.Module):
    def __init__(self, out_c, gamma=2, b=1, **kwargs):
        super(MFE, self).__init__(**kwargs)

        # 创建多个分支self.create_branch()
        self.branches = nn.ModuleList()
        for i in range(1, out_c + 1):
            branch = self.create_branch(i)
            self.branches.append(branch)

        self.batch_norm = nn.BatchNorm1d(out_c)

        # self.ECABlock_1d = ECABlock_1d(out_c)

    def forward(self, x):
        # 对每个分支进行前向传播
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        # 在这里可以对各个分支的输出进行合并或其他操作
        # 这里简单地将各个分支的输出放入一个列表返回
        x1 = torch.cat(branch_outputs, dim=1)

        # freq_Weight = self.ECABlock_1d(x1)
        #
        # x2 = x1 * freq_Weight.expand_as(x1)
        x2 = x1

        x3 = F.relu(x2)
        # x3 = x2

        x4 = self.batch_norm(x3)

        return x4

    def create_branch(self, i):
        # 这个方法用于创建一个分支，你可以根据需要自定义每个分支的结构
        branch = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=2 * i - 1, padding=(2 * i - 1 - 1) // 2, stride=1),
        )
        return branch


class Add(nn.Module):
    '''
    实现加权和
    inputs: 【a, b】
    '''

    def __init__(self, epsilon=1e-12):
        super(Add, self).__init__()
        self.epsilon = epsilon
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_relu = nn.ReLU()

    def forward(self, x):
        w = self.w_relu(self.w)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return weight[0] * x[0] + weight[1] * x[1]


class MHLSA(nn.Module):
    def __init__(self, emb_dim, heads):
        super(MHLSA, self).__init__()
        self.dim, self.heads = emb_dim, heads

    def forward(self, q, k, v):
        # print(q.shape)
        # print(torch.flatten(q, 2).shape) torch.Size([1, 8, 1024])
        q = torch.flatten(q, 2).transpose(1, 2)
        k = torch.flatten(k, 2).transpose(1, 2)
        v = torch.flatten(v, 2).transpose(1, 2)
        # print(q.shape) torch.Size([1, 1024, 8])
        # print(k.shape) torch.Size([1, 256, 8])
        # print(v.shape) torch.Size([1, 256, 8])
        if self.heads == 1:
            q, k = F.softmax(q, dim=2), F.softmax(k, dim=1)
            # print(q.bmm(k.transpose(2, 1).bmm(v)).transpose(1, 2).shape)
            return q.bmm(k.transpose(2, 1).bmm(v)).transpose(1, 2)

        else:
            q = q.split(self.dim // self.heads, dim=2)
            k = k.split(self.dim // self.heads, dim=2)

            v = v.split(self.dim // self.heads, dim=2)
            atts = []
            for i in range(self.heads):
                att = F.softmax(q[i], dim=2).bmm(F.softmax(k[i], dim=1).transpose(2, 1).bmm(v[i]))
                atts.append(att.transpose(1, 2))
            return torch.cat(atts, dim=1)


class FFN(nn.Module):
    def __init__(self, dim, ratio=4):
        super(FFN, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(dim, dim // ratio), nn.GELU(),
            nn.Linear(dim // ratio, dim), nn.GELU(), )
        self.add = Add()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        # print(x.shape) torch.Size([1, 8, 1024])
        # FNN在维度上进行MLP
        feature = self.MLP(x.transpose(1, 2))
        return self.bn(self.add([feature.transpose(1, 2), x]))


class ECABlock_1d(nn.Module):
    def __init__(self, in_C, gamma=2, b=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        t = int(abs((math.log(in_C, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        # k = 3
        print(k)
        self.avg_pool1d = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)

    def forward(self, x):
        y = self.avg_pool1d(x)
        # [1, 10, 1]
        # 将 y 的最后一维（大小为1）去除，并交换倒数第一维和倒数第二维，以适应卷积操作的输入要求
        y = self.conv(y.transpose(-1, -2))
        # [1, 1, 10]
        y = y.transpose(-1, -2)
        # [1, 10, 1]

        return y


class LinearAttentionBlock(nn.Module):
    def __init__(self, heads, dim, *args, **kwargs):
        # 输入 bx8x1024 1024是时间步，8是维度，所以计算 线性注意力时要.transpose(1, 2)
        super(LinearAttentionBlock, self).__init__()
        self.q_k_v = nn.ModuleList([
            # 分别改变 QKV的尺度， 其中 KV 时间长度/4
            nn.Sequential(
                # groups=dim 时，表示使用深度卷积，每个通道的输入与输出通道相对应。分组卷积可以减少模型的参数量，但通常在特定情况下使用，比如用于模型的轻量化或加速计算。
                nn.Conv2d(dim, dim, 3,
                          stride=1 if i == 0 else 2,
                          padding=1, groups=dim,
                          bias=False),
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, dim, 1, 1, 0), nn.GELU())
            for i in range(3)])

        # 计算时间尺度注意力
        self.MHLSA = MHLSA(dim, heads)
        # 权值加和
        self.add = Add()
        self.bn = nn.BatchNorm1d(dim)
        self.FFN = FFN(dim)

    def forward(self, x):
        # x [1, 8, 1024]
        b, c, l = x.size()
        maps = x.view(-1, c, int(l ** 0.5), int(l ** 0.5))
        # print(maps.shape) torch.Size([1, 8, 32, 32])
        # print(self.q_k_v[0](maps).shape) torch.Size([1, 8, 32, 32])
        # print(self.q_k_v[2](maps).shape) torch.Size([1, 8, 16, 16])
        MHLSA = self.MHLSA(
            self.q_k_v[0](maps),
            self.q_k_v[1](maps),
            self.q_k_v[2](maps))
        # print(MHLSA.shape) torch.Size([1, 8, 1024])

        att_Out = self.bn(self.add([MHLSA, x]))
        FNN_out = self.FFN(att_Out)

        return FNN_out


class LFTFormer_block(nn.Module):
    def __init__(self, d_in, d_out, gamma=2, b=1, heads=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 频率尺度注意力
        self.ECABlock = ECABlock_1d(d_in)

        # 时间尺度的注意力
        self.LinearAttentionBlock = LinearAttentionBlock(heads, d_in)
        self.batch_norm = nn.BatchNorm1d(d_out)
        self.convs_L = nn.Conv1d(d_in, d_in, kernel_size=5, padding=2, stride=4)
        self.convs_D = nn.Conv1d(d_in, d_out, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        # 通道维权值
        freq_Weight = self.ECABlock(x)
        # [1, 10, 1]

        # 时间尺度的注意力
        Att_x = self.LinearAttentionBlock(x)
        # print(Att_x.shape) torch.Size([1, 8, 1024])

        x1 = self.convs_L(Att_x)
        # 通道注意力赋值
        DouBle_Att = x1 * freq_Weight.expand_as(x1)

        x2 = self.convs_D(DouBle_Att)
        x3 = F.gelu(x2)

        LFTFormer_block_out = self.batch_norm(x3)

        return LFTFormer_block_out


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob, lastWeightZero=True):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.BN = nn.BatchNorm1d(hidden_dim)
        if lastWeightZero:
            self.zero_last_layer_weight()

    def zero_last_layer_weight(self):
        self.fc2.weight.data = torch.zeros_like(self.fc2.weight)

    def forward(self, x):
        x = x.squeeze(2)
        x = self.fc1(x)
        x = self.BN(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LFTFormer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LFTFormer, self).__init__(*args, **kwargs)
        self.MFE1 = MFE(4)  # torch.Size([1, 8, 1024])
        self.Encoder = nn.Sequential(
            LFTFormer_block(4, 16, heads=1),  # torch.Size([1, 16, 256])
            LFTFormer_block(16, 32, heads=1),  # torch.Size([1, 32, 64])
            LFTFormer_block(32, 64, heads=1),  # torch.Size([1, 64, 16])
            nn.AdaptiveAvgPool1d(1)  # torch.Size([1, 64, 1])
        )
        self.Feedforward = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            # nn.Linear(64, 10)
        )
        self.lastLinear = nn.Linear(64, 9)

        nn.init.xavier_uniform_(self.Feedforward[0].weight)

        self.zero_last_layer_weight()

    def zero_last_layer_weight(self):
        self.lastLinear.weight.data = torch.zeros_like(self.lastLinear.weight)
        self.lastLinear.bias.data = torch.zeros_like(self.lastLinear.bias)

    def forward(self, data):
        # FeatureHead = self.MFE1(data)
        FeatureHead = data
        EncodeFeature = self.Encoder(FeatureHead)  # torch.Size([1, 64, 1])
        EncodeFeature = EncodeFeature.squeeze(2)  # torch.Size([1, 64])
        x1 = self.Feedforward(EncodeFeature)
        out = self.lastLinear(x1)
        return out


def start_tsne(x_train, y_train, filepath):
    print('x_train' + str(x_train.shape))
    # 每一个点代表一个数据样本
    # 不同颜色代表不同类
    # 越接近代表越相关
    print("正在进行初始输入数据的可视化...")
    # 假设你已经有了 x_train（PyTorch张量）和 y_train（NumPy数组）作为数据和标签

    # 将 x_train 转换为 NumPy 数组并进行形状重塑
    x_train_np = x_train.numpy()
    x_train1 = x_train_np.reshape(len(x_train_np), -1)

    # 使用 Scikit-Learn 执行 TSNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(x_train1)

    # 绘制 TSNE 图
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train)
    plt.colorbar()
    plt.savefig(filepath)
    plt.show()


def End_tsne(model, x_traino, y_traino, filepath, device):
    x_train = x_traino
    y_train = y_traino

    print('x_train' + str(x_train.shape))
    print("正在进行初始训练完数据的可视化...")
    # 获取模型的输出特征
    model.eval()
    with torch.no_grad():
        x_train_embed = model(x_train.to(device)).cpu().numpy()

    # 使用 TSNE 执行降维
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(x_train_embed)

    # 绘制 TSNE 图
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train)
    plt.colorbar()
    plt.savefig(filepath)
    plt.show()


if __name__ == '__main__':
    # net = nn.Sequential(MFE(2), LFTFormer_block(2, 4), LFTFormer_block(4, 8), LFTFormer_block(8, 16),
    #                     nn.AdaptiveAvgPool1d(1), MLPClassifier(16, 32, 10, 0.5))
    #
    # X = torch.rand(size=(10, 1, 1024))
    # for layer in net:
    #     X = layer(X)
    #     print(layer.__class__.__name__, ' output shape:\t', X.shape)

    model = LFTFormer()


    folder_path = "../data/HUST"

    Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = Read_Data_From_HUST(filepath=folder_path, SampleNum=200, SampleLength=1024, Rate=None, isMaxMin=False)

    # 将NumPy数组转换为PyTorch张量，标签使用dtype=torch.int64
    Train_X = torch.tensor(Train_X, dtype=torch.float32)
    Train_Y = torch.tensor(Train_Y, dtype=torch.int64)
    Valid_X = torch.tensor(Valid_X, dtype=torch.float32)
    Valid_Y = torch.tensor(Valid_Y, dtype=torch.int64)
    Test_X = torch.tensor(Test_X, dtype=torch.float32)
    Test_Y = torch.tensor(Test_Y, dtype=torch.int64)

    print(Train_X.shape)

    print(Train_Y)

    # 创建TensorDataset对象
    train_dataset = TensorDataset(Train_X, Train_Y)
    valid_dataset = TensorDataset(Valid_X, Valid_Y)
    test_dataset = TensorDataset(Test_X, Test_Y)

    # 创建DataLoader对象，你可以设置batch_size等参数
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    def compute_accuracy(output, target):
        _, predicted = torch.max(output, 1)
        correct = (predicted == target).sum().item()
        accuracy = correct / target.size(0)
        return accuracy

    # 记录训练集和验证集的损失和准确率
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    num_epochs, batch_size = 200, 64
    # 设置初始学习率和最小学习率
    max_lr = 0.001
    min_lr = 0.0001

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=max_lr)

    start_time = time.time()
    for epoch in range(num_epochs):
        print(f'\r Progress: {epoch + 1}/{num_epochs}', end='\n')
        # 使用余弦退火公式计算学习率
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(epoch / num_epochs * math.pi))
        # 设置优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # 训练模型
        model.train()
        train_loss = 0.0
        # 用于在训练过程中跟踪正确分类的样本数量
        correct_train = 0

        a = 0
        for inputs, labels in train_loader:
            a = a + 1

            inputs = inputs.to(device)  # 移动输入数据到GPU
            labels = labels.to(device)  # 移动标签到GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  # 首先更新模型参数
            # 然后更新学习率
            # 更新学习率

            train_loss += loss.item()
            correct_train += (compute_accuracy(outputs, labels) * len(inputs))
            print('batch:' + str(a) + '----' +
                  'loss:' + str(loss.item()) + '----' +
                  'correct_train:' + str(compute_accuracy(outputs, labels)))

        train_loss /= len(train_loader)
        train_accuracy = correct_train / len(train_dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 验证模型
        model.eval()
        val_loss = 0.0
        correct_val = 0

        with torch.no_grad():
            # 在验证过程中，通常不需要计算梯度，因为只是用来评估模型的性能，而不是进行训练。
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                correct_val += (compute_accuracy(outputs, labels) * len(inputs))

        val_loss /= len(valid_loader)
        val_accuracy = correct_val / len(valid_dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')


    end_time = time.time()