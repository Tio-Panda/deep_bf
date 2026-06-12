import torch.nn as nn

class ModelBase(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers 

    def forward(self, batch_data, ids, angles):
        x = batch_data

        for layer in self.layers:
            x = layer(x, ids, angles)
        return x

class Binn(ModelBase):
    pass

class BinnOG(ModelBase):
    pass

class Sandwich(ModelBase):
    pass
