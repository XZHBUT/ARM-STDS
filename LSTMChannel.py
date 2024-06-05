import torch
import torch.nn as nn


def Build_Patches(input_tensor, patch_size, stride=None):
    """
    从输入张量中提取一维补丁。

    参数:
    - input_tensor: 输入张量，维度为 [B, L, 1]。
    - patch_size: 每个补丁的长度。
    - stride: 补丁之间的步长，默认为 patch_size。

    返回:
    - patches: 补丁张量，维度为 [B, Y, patch_size]。
    """
    if stride is None:
        stride = patch_size  # 如果未指定步长，则步长等于补丁长度

    B, L, _ = input_tensor.shape
    # 计算可以提取多少个补丁
    Y = (L - patch_size) // stride + 1

    # 初始化补丁张量
    patches = torch.zeros(B, Y, patch_size, device=input_tensor.device)

    for i in range(Y):
        start = i * stride
        end = start + patch_size
        patches[:, i, :] = input_tensor[:, start:end].squeeze(-1)

    return patches


def Restore_Patches(patches, original_length,stride=None):
    """
    从补丁张量中恢复原始序列。

    参数:
    - patches: 补丁张量，维度为 [B, Y, patch_size]。
    - original_length: 原始序列的长度 L。

    返回:
    - restored_tensor: 恢复的张量，维度为 [B, L, 1]。
    """
    B, Y, patch_size = patches.shape
    if stride == None:
        stride = patch_size  # 假设步长等于补丁长度

    # 初始化恢复后的张量
    restored_tensor = torch.zeros(B, original_length, 1, device=patches.device)

    for i in range(Y):
        start = i * stride
        end = start + patch_size
        restored_tensor[:, start:end, 0] = patches[:, i, :]

    return restored_tensor

class ChannelWiseLSTM(nn.Module):
    def __init__(self,in_channel, Patch_SeqLen, hidden_size, num_layers, patch_size, stride=None):
        super(ChannelWiseLSTM, self).__init__()
        self.in_channel = in_channel
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.Patch_SeqLen = Patch_SeqLen
        if stride is None:
            self.stride = patch_size
        else:
            self.stride = stride

        # 创建8个独立的LSTM模块，每个模块对应一个通道
        self.lstms = nn.ModuleList([nn.LSTM(self.patch_size, hidden_size, num_layers, batch_first=True) for _ in range(self.in_channel)])
        # 创建8个独立的全连接层，用于将LSTM的输出调整为原始序列长度
        self.fcs = nn.ModuleList([nn.Linear(hidden_size, self.patch_size) for _ in range(self.in_channel)])


        self.norm1 = nn.LayerNorm(self.patch_size)
        self.dropout = nn.Dropout(0.2)

        # self.fcs = nn.ModuleList([nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size * 4),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_size * 4, self.patch_size)
        # ) for _ in range(self.in_channel)])





    def forward(self, x):
        # x的形状为(B, 8, L)
        B, C, L = x.shape
        outputs = []

        for i in range(C):
            # 对每个通道独立处理
            # 提取当前通道的数据，形状为(B, L)，并增加一个维度变为(B, L, 1)以匹配LSTM的输入要求
            channel_data = x[:, i, :].view(B, L, 1)

            patch_data = Build_Patches(channel_data, patch_size=self.patch_size,stride=self.stride)
            # 通过对应的LSTM模块
            lstm_out, _ = self.lstms[i](patch_data)

            # 将LSTM的输出通过对应的全连接层，将输出调整为原始序列长度
            channel_out = self.fcs[i](lstm_out)

            # channel_out = self.norm1(channel_out)

            channel_out = Restore_Patches(channel_out, original_length=L,stride=self.stride)

            # channel_out = channel_data + channel_out
            # 移除最后一个维度，因为output_size=1
            channel_out = channel_out.squeeze(-1)
            outputs.append(channel_out)

        # 将所有通道的输出堆叠起来，形状为(B, 8, L)
        outputs = torch.stack(outputs, dim=1)


        return outputs

if __name__ == '__main__':


    # 输入参数
    B, C, L = 32, 8, 1024  # 示例：批次大小32，8个通道，序列长度10
    input_size = 1  # 每个时间步的输入维度
    hidden_size = 64  # LSTM隐藏层的大小
    num_layers = 1  # LSTM堆叠的层数
    output_size = 1  # 输出与输入的最后一维相同，这里是因为我们预计通过全连接层匹配维度
    in_channel = 8

    # 创建模型
    model = ChannelWiseLSTM(in_channel=8, Patch_SeqLen=32, hidden_size=32, num_layers=1, patch_size=32)

    # 示例输入
    x = torch.randn(B, C, L)

    # 前向传播
    output = model(x)

    # 输出形状
    print(output.shape)  # 预期的输出形状：(B, 8, L)
