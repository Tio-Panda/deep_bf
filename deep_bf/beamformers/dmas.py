import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from .utils.bp_filter import get_freqs, pass_band_filter

class FDMAS(nn.Module):
    def __init__(self, BW=0.7, for_dmas=True, batch_size=5, device="cuda", dtype=torch.float32):
        """
        Args:
            angles_idx: Índices de los ángulos a procesar
            batch_size: Número de ángulos a procesar simultáneamente (default: 10)
        """
        super().__init__()
        self.BW = BW
        self.for_dmas = for_dmas

        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

    def forward(self, rf, t0, d_tx, d_rx, fs, f0, c0, apod):
        """
        Args:
            d_tx: Transmit distances con shape [n_angles, nz, nx]
            d_rx: Receive distances con shape [n_elements, nz, nx]
            apod: Apodization con shape [n_elements, nz, nx]

        Returns:
            das: Beamformed data con shape [n_angles, nz, nx]
        """

        _, nz, nx = d_tx.shape
        n_angles, n_elements, n_samp = rf.shape
        norm_factor = 2.0 / (n_samp - 1)

        freqs = get_freqs(fs, f0, self.BW, self.for_dmas)
        H = pass_band_filter(nz, freqs, "tukey50", device=self.device, dtype=self.dtype)

        fdmas = torch.zeros(n_angles, nz, nx, device=self.device, dtype=self.dtype)

        for i in range(0, n_elements, self.batch_size):
            end_idx = min(i + self.batch_size, n_elements)
            batch_size_elem = end_idx - i

            rf_batch = rf[:, i:end_idx, :]  # [n_angles, batch_size_elem, n_samples]
            d_rx_batch = d_rx[i:end_idx, :, :]  # [batch_size_elem, nz, nx]
            apod_batch = apod[i:end_idx, :, :]  # [batch_size_elem, nz, nx]

            total_delay = (d_tx.unsqueeze(1) + d_rx_batch.unsqueeze(0)) / c0 - t0.view(-1, 1, 1, 1)
            delays_batch = total_delay * fs # [n_angles, batch_size_elem, nz, nx]

            x_folded = rf_batch.reshape(n_angles * batch_size_elem, 1, 1, n_samp)
            delays_folded = delays_batch.reshape(n_angles * batch_size_elem, nz, nx)

            grid = torch.zeros(n_angles * batch_size_elem, nz, nx, 2,
                             dtype=self.dtype, device=self.device)
            grid[..., 0] = delays_folded * norm_factor - 1.0

            sampled = F.grid_sample(
                x_folded,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )

            sampled = sampled.view(n_angles, batch_size_elem, nz, nx)
            sampled = sampled * apod_batch.unsqueeze(0)  # [n_angles, batch_size_elem, nz, nx] * [1, batch_size_elem, nz, nx]

            s_hat = torch.sign(sampled) * torch.sqrt(torch.abs(sampled))

            if i == 0:
                sum_s_hat = torch.sum(s_hat, dim=1)
                sum_abs_s = torch.sum(torch.abs(sampled), dim=1)
            else:
                sum_s_hat += torch.sum(s_hat, dim=1)
                sum_abs_s += torch.sum(torch.abs(sampled), dim=1)

        fdmas = 0.5 * (sum_s_hat**2 - sum_abs_s)  # [n_angles, nz, nx]
        fdmas = torch.fft.rfft(fdmas, dim=1) # [n_angles, nz//2+1, nx]
        H = H[:nz//2+1]
        fdmas *= H[None, :, None]
        fdmas = torch.fft.irfft(fdmas, n=nz, dim=1).real

        return fdmas
