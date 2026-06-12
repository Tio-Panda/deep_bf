import torch.nn as nn

class BasicConv2dModule(nn.Module):
    def __init__(self, conv2d, activation):
        super().__init__()
        self.conv2d = conv2d
        self.activation = activation

    def forward(self, x1, x2=None, x3=None):
        return self.activation(self.conv2d(x1))

class BeamformerModule(nn.Module):
    def __init__(self, bf):
        super().__init__()
        self.bf = bf

    def forward(self, data, ids, angles):
        return self.bf(data, ids, angles)
