import torch
import torch.nn as nn
from deep_bf.models.bf_layers.das import WDAS

class WConv2d(nn.Module):
    def __init__(self, conv, activation):
        super().__init__()
        self.conv = conv
        self.activation = activation

    def forward(self, x1, x2=None):
        return self.activation(self.conv(x1))

def def_conv2d(in_ch, out_ch, kernel_size, padding):
    m = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=True)
    nn.init.xavier_uniform_(m.weight)
    nn.init.zeros_(m.bias)

    return m

class BasicModel(nn.Module):
    ARCHITECTURE = []

    def __init__(self, id, gsi, chunk_size=64, negative_slope=0.01, is_train=True, device="cuda", dtype=torch.float32):
        super().__init__()
        act = nn.LeakyReLU(negative_slope=negative_slope)
        architecture = self.ARCHITECTURE[id-1]

        last_ch = 1
        self.layers = nn.ModuleList()
        for ch, kernel in architecture:
            if ch == -1:
                self.layers.append(WDAS(is_train, gsi, chunk_size, device, dtype))
                continue
            
            self.layers.append(WConv2d(def_conv2d(last_ch, ch, kernel, "same"), act))
            last_ch = ch

    def forward(self, rfs, x2):
        '''
        - is_train => x2 = ids
        - not is_train x2 = samples_idx
        '''
        x = rfs
        for layer in self.layers:
                x = layer(x, x2)

        return x
