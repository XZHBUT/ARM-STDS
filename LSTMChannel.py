import torch
import torch.nn as nn


def Build_Patches(input_tensor, patch_size, stride=None):

    if stride is None:
        stride = patch_size 

    B, L, _ = input_tensor.shape

    Y = (L - patch_size) // stride + 1


    patches = torch.zeros(B, Y, patch_size, device=input_tensor.device)

    for i in range(Y):
        start = i * stride
        end = start + patch_size
        patches[:, i, :] = input_tensor[:, start:end].squeeze(-1)

    return patches


def Restore_Patches(patches, original_length,stride=None):

    B, Y, patch_size = patches.shape
    if stride == None:
        stride = patch_size  


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


        self.lstms = nn.ModuleList([nn.LSTM(self.patch_size, hidden_size, num_layers, batch_first=True) for _ in range(self.in_channel)])

        self.fcs = nn.ModuleList([nn.Linear(hidden_size, self.patch_size) for _ in range(self.in_channel)])


        self.norm1 = nn.LayerNorm(self.patch_size)
        self.dropout = nn.Dropout(0.2)

        # self.fcs = nn.ModuleList([nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size * 4),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_size * 4, self.patch_size)
        # ) for _ in range(self.in_channel)])





    def forward(self, x):

        B, C, L = x.shape
        outputs = []

        for i in range(C):

            channel_data = x[:, i, :].view(B, L, 1)

            patch_data = Build_Patches(channel_data, patch_size=self.patch_size,stride=self.stride)

            lstm_out, _ = self.lstms[i](patch_data)


            channel_out = self.fcs[i](lstm_out)

            # channel_out = self.norm1(channel_out)

            channel_out = Restore_Patches(channel_out, original_length=L,stride=self.stride)

            # channel_out = channel_data + channel_out

            channel_out = channel_out.squeeze(-1)
            outputs.append(channel_out)


        outputs = torch.stack(outputs, dim=1)


        return outputs


