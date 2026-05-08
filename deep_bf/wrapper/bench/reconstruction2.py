import numpy as np
from ...beamformers.utils.b_mode import get_bmode
from ...constants.bf import PWDataType


class Reconstruction:
    def __init__(self, pw, data, mode, times):
        self.zlims = np.array(pw.zlims) * 1e3
        self.xlims = np.array(pw.xlims) * 1e3
        self.data = data.cpu().numpy()
        self.mode = mode
        self.times = times

    def get_bmode(self, mean=True, vmin=-60, vmax=0, eps=1e-10):
        d = self.data.ndim

        if (d == 3 and self.mode == PWDataType.RF) or (d == 4 and self.mode == PWDataType.IQ_SPLIT):
            if mean:
                _data = np.mean(self.data, axis=0)
            else:
                _data = np.sum(self.data, axis=0)
        else:
            _data = self.data

        b_mode = get_bmode(_data, vmin=vmin, vmax=vmax, eps=eps, mode=self.mode)

        return b_mode

    def get_total_time(self):
        pass
