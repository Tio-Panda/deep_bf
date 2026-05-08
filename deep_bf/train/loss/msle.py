import torch
import torch.nn as nn

class MSLELoss(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.mse = nn.MSELoss(**params)

    def forward(self, pred, actual):
        return self.mse(torch.log(pred + 1), torch.log(actual + 1))
