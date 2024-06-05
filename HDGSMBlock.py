import math
import time

import networkx as nx
from torch import optim
from torch.nn import init

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_networkx

from Modules.LSTMChannel import ChannelWiseLSTM
from Modules.MambaChannel import ChannelWiseMamba
from Modules.TransformerChannel import ChannelWiseTransformer

from data.DataProcess import Read_Data_From_SEU
from torch.utils.data import TensorDataset, DataLoader
from Modules.GraphEncoder import GraphFeatureExtraction, GraphPooling


class HDGSMBlock(nn.Module):
    def __init__(self, in_channel, Seq_len, hidden_size, patch_size, device, stride=None):
        super().__init__()

        self.in_channel = in_channel
        self.Seq_len = Seq_len
        self.hidden_size = hidden_size
        self.device = device

        self.patch_size = patch_size
        if stride == None:
            self.stride = patch_size
        else:
            self.stride = stride


        self.Patch_SeqLen = (self.Seq_len - self.patch_size) // self.stride + 1

        self.LstmChannel = ChannelWiseLSTM(self.in_channel,
                                           Patch_SeqLen=self.Patch_SeqLen,
                                           hidden_size=self.hidden_size,
                                           num_layers=1,
                                           patch_size=self.patch_size,
                                           stride=self.stride)




        self.GrapChannel = GraphFeatureExtraction(node_n=self.in_channel, Seq_len=self.Seq_len)

        self.bn = torch.nn.BatchNorm1d(self.in_channel)

        self.dropout = nn.Dropout(0.4)

    def forward(self, x, At, isRandomAt=False):
        Timeout = self.LstmChannel(x)
        # Timeout =self.MambaChannel(x)
        # Timeout = self.TransformerChannel(x)

        Spatialout = self.GrapChannel(x=Timeout, At=At, isRandomAt=isRandomAt)

        # Spatialout  = self.MambaChannel(x) + self.GrapChannel(x=x, At=At, isRandomAt=isRandomAt)

        Bnout = self.dropout(F.relu(self.bn(Spatialout)))

        return Bnout
