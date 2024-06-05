import math
import time

import networkx as nx
from thop import profile
from torch import optim
from torch.nn import init

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from Modules.LSTMChannel import ChannelWiseLSTM
from Modules.MambaChannel import ChannelWiseMamba

from data.DataProcess import Read_Data_From_SEU, Read_Data_From_HUST, Read_Data_From_HIT, Read_Data_From_THU
from torch.utils.data import TensorDataset, DataLoader

from data.addnoise import addNoiseBatch



from Modules.GraphEncoder import GraphFeatureExtraction, GraphPooling

from Modules.GraphicalStructureModeling import GraphicalStructureModeling
from Modules.HDGSMBlock import HDGSMBlock


from Comparativemodeli.A_TSGNN.GraphBuild import GraphBuild

def compute_formula(x, a, b):
    # 计算 log2(x)
    log_x = math.log2(x)

    # 计算 (log2(x) - b) / a 并向下取整
    floor_value = math.floor((log_x - b) / a)

    # 计算 2 的 floor_value 次幂
    result = 2 ** floor_value

    return result
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


class HDGSM(nn.Module):
    def __init__(self, in_channel, Seq_len,num_classes, device):
        super().__init__()

        self.in_channel = in_channel
        self.Seq_len = Seq_len
        self.device = device
        self.num_classes = num_classes

        self.GrapPool = GraphPooling(node_n=self.in_channel)

        self.GrapStructure = GraphicalStructureModeling(in_channel=self.in_channel, Seq_len=self.Seq_len, device=device)

        self.patch_size_1 = self.hidden_size_1 = compute_formula(self.Seq_len, 2, 0)
        self.patch_size_2 = self.hidden_size_2 = compute_formula(self.Seq_len / 2, 2, 0)
        self.block1 = HDGSMBlock(in_channel=self.in_channel, Seq_len=self.Seq_len, hidden_size=self.hidden_size_1, patch_size=self.patch_size_1,
                                 stride=int(self.patch_size_1) , device=device)

        self.block2 = HDGSMBlock(in_channel=self.in_channel, Seq_len=int(self.Seq_len / 2), hidden_size=self.hidden_size_2,
                                 patch_size=int(self.patch_size_2), stride=self.patch_size_2, device=device)

        self.Feedforward = nn.Sequential(
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.6),
        )
        self.lastLinear = nn.Linear(32, self.num_classes)

        nn.init.xavier_uniform_(self.Feedforward[0].weight)

        self.zero_last_layer_weight()

        self.GraphBuild = GraphBuild(epsilon=0)

    def zero_last_layer_weight(self):
        self.lastLinear.weight.data = torch.zeros_like(self.lastLinear.weight)
        self.lastLinear.bias.data = torch.zeros_like(self.lastLinear.bias)

    def forward(self, x, isRandomAt=False, p=1):
        # At = self.GraphBuild(x)
        x, At = self.GrapStructure(x)

        out1 = self.block1(x, At)

        out1 = self.block2(out1, At)

        out2 = out1

        out3 = self.GrapPool(out2)

        out4 = out3.squeeze(dim=1)

        # out5 = self.fc1(out4)
        # out5 = self.bn1(out5)
        # out5 = self.dropout(out5)
        #
        # out6 = self.fc2(out5)

        x1 = self.Feedforward(out4)
        out6 = self.lastLinear(x1)

        return out6, At

def min_max_normalization(data):
    min_vals = data.min(dim=2, keepdim=True)[0]  # 沿L维度计算最小值，保持维度用于广播
    max_vals = data.max(dim=2, keepdim=True)[0]  # 沿L维度计算最大值，保持维度用于广播

    # 避免分母为0的情况，可以添加一个很小的数epsilon
    epsilon = 1e-12
    normalized_data = (data - min_vals) / (max_vals - min_vals + epsilon)  # 归一化公式
    return normalized_data

if __name__ == '__main__':

    in_channel = 8
    Seq_len = 1024
    SampleNum = 3600
    num_epochs, batch_size = 100, 2048
    # 设置初始学习率和最小学习率
    max_lr = 0.0005
    min_lr = 0.0001
    num_classes = 8
    # folder_path = "../data/SEU"
    # SEU channel 8 out 9
    # HUST channel 5 out 11
    # HIT channel 6 out 3
    # THU channel 8 out 8 ps: 一个类别36个文件夹，建议一个类别3600个样本
    dataset = 'THU'
    folder_path = "../data/{}".format(dataset)

    Noise = False
    snr = 0


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn((64, in_channel, Seq_len), device=device)
    # model = HDGSM(in_channel=8, Seq_len=1024,device=device).to(device)
    model = HDGSM(in_channel=in_channel, Seq_len=Seq_len, num_classes=num_classes, device=device).to(device)

    # model.load_state_dict(torch.load('../result/best_model_SEU_1024.pth'),  strict=False)

    out, At = model(x)
    print(out.shape)
    print(At)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    # 计算MFLOPs
    input_data = torch.randn(1, in_channel, Seq_len).to(device)  # 例如，使用输入数据的示例
    macs, params = profile(model, inputs=(input_data,), verbose=False)
    print(f"Total FLOPs: {macs / 10 ** 6} MFLOPs")  # 将FLOPs转换为MFLOPs

    if dataset == 'SEU':
        Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = Read_Data_From_SEU(folder_path,
                                                                                SampleNum=SampleNum,
                                                                                SampleLength=Seq_len,
                                                                                Rate=None,
                                                                                isMaxMin=True)
    elif dataset == 'HUST':
        Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = Read_Data_From_HUST(filepath=folder_path,
                                                                                 SampleNum=SampleNum,
                                                                                 SampleLength=Seq_len,
                                                                                 Rate=None,
                                                                                 isMaxMin=True)
    elif dataset == 'HIT':
        Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = Read_Data_From_HIT(folder_path,
                                                                                SampleNum=SampleNum,
                                                                                SampleLength=Seq_len,
                                                                                Rate=None)
    elif dataset == 'THU':
        Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = Read_Data_From_THU(folder_path,
                                                                                SampleNum=SampleNum,
                                                                                SampleLength=Seq_len,
                                                                                Rate=None,
                                                                                isMaxMin=True)
    else:
        print("不存在数据集")


    if Noise:
        Train_X = addNoiseBatch(Train_X, snr)
        Valid_X = addNoiseBatch(Valid_X, snr)
        Test_X = addNoiseBatch(Test_X, snr)




    # 将NumPy数组转换为PyTorch张量，标签使用dtype=torch.int64
    Train_X = torch.tensor(Train_X, dtype=torch.float32)
    Train_Y = torch.tensor(Train_Y, dtype=torch.int64)
    Valid_X = torch.tensor(Valid_X, dtype=torch.float32)
    Valid_Y = torch.tensor(Valid_Y, dtype=torch.int64)
    Test_X = torch.tensor(Test_X, dtype=torch.float32)
    Test_Y = torch.tensor(Test_Y, dtype=torch.int64)


    # Train_X = min_max_normalization(Train_X)
    # Valid_X = min_max_normalization(Valid_X)
    # Test_X = min_max_normalization(Test_X)

    print(Train_X.shape)

    # print(Train_Y)

    # 创建TensorDataset对象
    train_dataset = TensorDataset(Train_X, Train_Y)
    valid_dataset = TensorDataset(Valid_X, Valid_Y)
    test_dataset = TensorDataset(Test_X, Test_Y)

    # 创建DataLoader对象，你可以设置batch_size等参数
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


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

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=max_lr)

    start_time = time.time()
    curr_val_loss = 999.0
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
            outputs, At = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  # 首先更新模型参数
            # 然后更新学习率
            # 更新学习率

            train_loss += loss.item()
            correct_train += (compute_accuracy(outputs, labels) * len(inputs))
            print('\r','batch:' + str(a) + '----' +
                  'loss:' + str(loss.item()) + '----' +
                  'correct_train:' + str(compute_accuracy(outputs, labels)),end='', flush=True)

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
                outputs, At = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                correct_val += (compute_accuracy(outputs, labels) * len(inputs))

        val_loss /= len(valid_loader)
        val_accuracy = correct_val / len(valid_dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print('\n' + f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        print(f'Best Model loss: {curr_val_loss:.4f}')

        if curr_val_loss > val_loss:
            curr_val_loss = val_loss
            torch.save(model.state_dict(), '../result/best_model_{}_{}.pkl'.format(dataset, Seq_len))  # 保存模型参数到文件 'best_model.pth'

            print(f'Save Best Model loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    end_time = time.time()
    # 计算执行时间（以分钟为单位）
    execution_time = end_time - start_time
    execution_time_minutes = execution_time / 60
    # 打印执行时间
    print(f"执行时间：{execution_time_minutes:.2f} min")

    # 使用模型进行测试集评估
    with torch.no_grad():
        outputs,_ = model(Test_X.to(device))
    # 获取模型的预测类别（假设是分类任务）
    _, predicted = torch.max(outputs, 1)

    # 计算准确率
    correct = (predicted == Test_Y.to(device)).sum().item()
    total = Test_Y.size(0)
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')


    def save_losses(train_losses, test_losses, filename="losses.txt"):
        with open(filename, 'w') as f:
            f.write("train_loss,val_loss\n")  # 写入头部，便于理解数据
            for train_loss, test_loss in zip(train_losses, test_losses):
                f.write(f"{train_loss},{test_loss}\n")  # 写入数据


    # 保存损失
    save_losses(train_losses, val_losses, filename="../result/Loss/Model_{}_BastLoss_{:.4f}_{}.txt".format(dataset, curr_val_loss, Seq_len))
    print("损失已保存")



