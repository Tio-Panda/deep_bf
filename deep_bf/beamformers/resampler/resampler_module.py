from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .resampler import GridSample, LinearInterpolation, IntegerDelayFractionalFIR
    from ...webdataset.gsi.gsi_for_training import GlobalSamplesIdxForTraining
    from ...webdataset.gsi.gsi_fast_bench import GlobalSamplesIdxFastBench

# TODO: Hay que borrar esta cuando se implemente el ToFC por id o algo para reemplazar la logica de ir por ids y angulos.

class ResamplerByIdsAndAngles(nn.Module):
    def __init__(
        self,
        gsi: "GlobalSamplesIdxForTraining",
        resampler: "GridSample | LinearInterpolation | IntegerDelayFractionalFIR",
        batch_size: int = 1,
        channel_batch_size: int = 4,
    ):
        super().__init__()
        self.gsi = gsi
        self.batch_size = batch_size
        self.channel_batch_size = channel_batch_size
        self.resampler = resampler
        self.nz, self.nx = gsi.nz, gsi.nx

    def forward(self, data, ids, angles):
        B, C, nc, ns = data.shape

        output_sorted = torch.empty(
            B, C, nc, self.nz, self.nx, device=data.device, dtype=data.dtype
        )

        ids_long = ids.to(torch.long)
        angles_long = angles.to(torch.long)

        pairs = torch.stack((ids_long, angles_long), dim=1)
        unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)
        perm = inverse.argsort(stable=True)
        inv_perm = perm.argsort(stable=True)
        data_sorted = data[perm]  # [B, C, nc, ns]
        inverse_sorted = inverse[perm]

        counts = torch.bincount(
            inverse_sorted, minlength=unique_pairs.shape[0]
        ).tolist()

        unique_pairs_cpu = unique_pairs.cpu().tolist()

        start = 0
        for (uid, angle), cnt in zip(unique_pairs_cpu, counts):
            end = start + cnt
            group_data = data_sorted[start:end]  # [cnt, C, nc, ns]
            samples_idx = self.gsi.get_samples_idx(
                uid, angle, device=data.device, dtype=data.dtype
            )  # [nc, nz, nx]
            for s in range(0, cnt, self.batch_size):
                e = min(s + self.batch_size, cnt)
                b = e - s
                for c_start in range(0, C, self.channel_batch_size):
                    c_end = min(c_start + self.channel_batch_size, C)
                    c_batch = c_end - c_start
                    chunk = group_data[s:e, c_start:c_end]  # [b, c_batch, nc, ns]
                    chunk_flat = chunk.reshape(
                        b * c_batch, nc, ns
                    )  # [b*c_batch, nc, ns]
                    sampled_flat = self.resampler(
                        chunk_flat, samples_idx, b * c_batch, nc, ns, self.nz, self.nx
                    )  # [b*c_batch, nc, nz, nx]
                    sampled = sampled_flat.reshape(b, c_batch, nc, self.nz, self.nx)
                    output_sorted[start + s : start + e, c_start:c_end] = sampled
            start = end
        return output_sorted[inv_perm]


class ResamplerByIdsAndAngles2(nn.Module):
    def __init__(
        self,
        mode,
        gsi: GlobalSamplesIdxForTraining,
        resampler: GridSample | LinearInterpolation | IntegerDelayFractionalFIR,
        batch_size=4,
    ):
        super().__init__()

        self.mode = mode
        self.gsi = gsi
        self.batch_size = batch_size
        self.resampler = resampler

        self.nz, self.nx = gsi.nz, gsi.nx

    def forward(self, data, ids, angles):
        if self.mode == "IQ":
            B, nc, ns, iq_ch = data.shape
            output_sorted = torch.empty(
                B, nc, self.nz, self.nx, iq_ch, device=data.device, dtype=data.dtype
            )
        else:
            B, nc, ns = data.shape
            output_sorted = torch.empty(
                B, nc, self.nz, self.nx, device=data.device, dtype=data.dtype
            )

        ids_long = ids.to(torch.long)
        angles_long = angles.to(torch.long)
        pairs = torch.stack((ids_long, angles_long), dim=1)

        unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)

        perm = inverse.argsort(stable=True)
        inv_perm = perm.argsort(stable=True)

        data_sorted = data[perm]
        inverse_sorted = inverse[perm]

        counts = torch.bincount(
            inverse_sorted, minlength=unique_pairs.shape[0]
        ).tolist()
        unique_pairs_cpu = unique_pairs.cpu().tolist()

        start = 0
        for (uid, angle), cnt in zip(unique_pairs_cpu, counts):
            end = start + cnt
            group_data = data_sorted[start:end]

            samples_idx = self.gsi.get_samples_idx(
                uid, angle, device=data.device, dtype=data.dtype
            )

            for s in range(0, cnt, self.batch_size):
                e = min(s + self.batch_size, cnt)
                b = e - s
                sampled_data = self.resampler(
                    group_data[s:e], samples_idx, b, nc, ns, self.nz, self.nx
                )
                output_sorted[start + s : start + e] = sampled_data

            start = end

        return output_sorted[inv_perm]


class ResamplerSimple(nn.Module):
    def __init__(
        self,
        gsi: GlobalSamplesIdxFastBench,
        resampler: GridSample | LinearInterpolation | IntegerDelayFractionalFIR,
        batch_size=4,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.resampler = resampler
        _, self.nc, self.nz, self.nx = gsi.samples_idx.shape

        self.register_buffer(
            "samples_idx", gsi.samples_idx.to(non_blocking=True), persistent=False
        )

    def forward(self, rfs, ids):
        is_iq = rfs.dim() == 4

        if is_iq:
            B, _, ns, iq_ch = rfs.shape
            output = torch.empty(
                B, self.nc, self.nz, self.nx, iq_ch, device=rfs.device, dtype=rfs.dtype
            )
        else:
            B, _, ns = rfs.shape
            output = torch.empty(
                B, self.nc, self.nz, self.nx, device=rfs.device, dtype=rfs.dtype
            )

        for s in range(0, B, self.batch_size):
            e = min(s + self.batch_size, B)
            b = e - s
            _rfs = rfs[s:e]
            _ids = ids[s:e]

            samples_idx = self.samples_idx[_ids]

            sampled_data = self.resampler(
                _rfs, samples_idx, b, self.nc, ns, self.nz, self.nx
            )

            output[s:e] = sampled_data

        return output
