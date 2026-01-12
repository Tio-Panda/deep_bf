import numpy as np
from ..beamformers import get_rf_bmode

# TODO: implementar calculo de tiempo y agregarlo

class Reconstruction:
    def __init__(self, data, Z, X, metadata):
        self.data = data.cpu()
        Z = Z.cpu()
        X = X.cpu()

        self.zlims = np.array([Z[0, 0], Z[-1, 0]]) * 1e3
        self.xlims = np.array([X[0, 0], X[0, -1]]) * 1e3

        self.metadata = metadata

    def get_bmode(self, vmin=-60, vmax=0):
        data = self.data.mean(axis=0)
        return get_rf_bmode(data, vmin=vmin, vmax=vmax)

