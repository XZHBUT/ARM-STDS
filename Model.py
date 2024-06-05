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

    log_x = math.log2(x)


    floor_value = math.floor((log_x - b) / a)


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
    min_vals = data.min(dim=2, keepdim=True)[0]  
    max_vals = data.max(dim=2, keepdim=True)[0]  


    epsilon = 1e-12
    normalized_data = (data - min_vals) / (max_vals - min_vals + epsilon)
    return normalized_data

