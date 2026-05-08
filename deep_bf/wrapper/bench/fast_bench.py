from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn

from dataclasses import dataclass, field
from typing import List

from ...beamformers.utils.delays import compute_meshgrid, compute_d_rx
from ...beamformers.apod.apod_builder import apod_builder
from ...beamformers.bf.bf_builder import bf_builder
from ...webdataset.gsi.gsi_fast_bench import GlobalSamplesIdxFastBench
from ...config_registery import PathCenter, BeamformerPacking

from ..reconstruction import Reconstruction

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config_registery import BeamformerConfig, ResamplerConfig, ApodConfig


@dataclass
class FastBenchTime:
    meshgrid: float
    d_rx: float
    apod: float
    batch_gsi: List[float] = field(default_factory=list)
    batch_resampler: List[float] = field(default_factory=list)
    batch_bf: List[float] = field(default_factory=list)


class FastBench(nn.Module):
    def __init__(
        self,
        pw,
        nz: int,
        nx: int,
        bfs: List[BeamformerConfig],
        resamplers: List[ResamplerConfig],
        apods: List[ApodConfig],
        meshgrid_nyquist=False,
        mode="RF",
        batch_size=10,
    ):
        super().__init__()
        self.pw = pw
        self.nz = nz
        self.nx = nx
        self.batch_size = batch_size
        self.mode = mode

        self.bfs = bfs
        self.resamplers = resamplers
        self.apods = apods

        self.times = {}
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.Z, self.X = compute_meshgrid(pw, nz, nx, nyquist=meshgrid_nyquist)
        torch.cuda.synchronize()
        self.times["meshgrid"] = time.perf_counter() - start

        if meshgrid_nyquist:
            self.nz, self.nx = self.Z.shape

        torch.cuda.synchronize()
        start = time.perf_counter()
        self.d_rx = compute_d_rx(pw, self.Z, self.X)
        torch.cuda.synchronize()
        self.times["d_rx"] = time.perf_counter() - start

        self.gsi = GlobalSamplesIdxFastBench(
            pw, self.Z, self.X, self.d_rx, batch_size=batch_size
        )

        # with PathCenter(location="local") as pc:
        #     paths = pc.dataset_paths
        # query = f"name == '{pw.name}'"
        # self.gsi = GlobalSamplesIdx(paths.samples_idx, query, multiple_mode=True, batch_size=batch_size)
        # self.gsi.set_samples_idx_by_pw(self.pw, self.Z, self.X, self.d_rx, np.arange(batch_size))

    def forward(self):
        output = {}
        times = {}
        bf_data = {}

        for bC in self.bfs:
            bf_name = bC.type

            output[bf_name] = {}
            times[bf_name] = {}
            bf_data[bf_name] = {}

            for aC in self.apods:
                apod_name = aC.type

                output[bf_name][apod_name] = {}
                times[bf_name][apod_name] = {}
                bf_data[bf_name][apod_name] = {}

                torch.cuda.synchronize()
                start = time.perf_counter()
                apod = apod_builder(self.Z, self.X, self.pw.probe_geometry, aC)

                torch.cuda.synchronize()
                apod_time = time.perf_counter() - start

                for rC in self.resamplers:
                    resampler_name = rC.type

                    output[bf_name][apod_name][resampler_name] = {}
                    times[bf_name][apod_name][resampler_name] = {
                        "meshgrid": self.times["meshgrid"],
                        "d_rx": self.times["d_rx"],
                        "apod": apod_time,
                    }

                    if self.mode == "IQ":
                        bf_data[bf_name][apod_name][resampler_name] = torch.empty(
                            self.pw.na,
                            self.nz,
                            self.nx,
                            2,
                            device="cpu",
                            pin_memory=True,
                        )
                    else:
                        bf_data[bf_name][apod_name][resampler_name] = torch.empty(
                            self.pw.na, self.nz, self.nx, device="cpu", pin_memory=True
                        )

                    bP = BeamformerPacking(bC, rC)
                    bf = bf_builder(self.gsi, bP, self.nz, self.nx, self.batch_size)

                    batch_gsi_times = []
                    batch_resampler_times = []
                    batch_bf_times = []

                    for s in range(0, self.pw.na, self.batch_size):
                        e = min(s + self.batch_size, self.pw.na)
                        angles = np.arange(s, e)

                        torch.cuda.synchronize()
                        start = time.perf_counter()

                        self.gsi.set_samples_idx_by_pw(angles)

                        torch.cuda.synchronize()
                        batch_gsi_times.append(time.perf_counter() - start)

                        rfs = torch.from_numpy(self.pw.data[s:e]).to("cuda")
                        ids = torch.from_numpy(self.gsi[self.pw.name]).to("cuda")

                        torch.cuda.synchronize()
                        start = time.perf_counter()

                        resampler = get_resampler_for_fast_bench(
                            self.gsi, rC, self.batch_size
                        )
                        sampled_data = resampler(rfs, ids)

                        torch.cuda.synchronize()
                        batch_resampler_times.append(time.perf_counter() - start)

                        with torch.no_grad():
                            torch.cuda.synchronize()
                            start = time.perf_counter()

                            bf_data[bf_name][apod_name][resampler_name][s:e].copy_(
                                bf(sampled_data, apod), non_blocking=True
                            )

                            torch.cuda.synchronize()
                            batch_bf_times.append(time.perf_counter() - start)

                    times[bf_name][apod_name][resampler_name]["batch_gsi"] = (
                        batch_gsi_times
                    )
                    times[bf_name][apod_name][resampler_name]["batch_resampler"] = (
                        batch_resampler_times
                    )
                    times[bf_name][apod_name][resampler_name]["batch_bf"] = (
                        batch_bf_times
                    )

                    _times = FastBenchTime(**times[bf_name][apod_name][resampler_name])

                    reconstruction = Reconstruction(
                        self.pw,
                        bf_data[bf_name][apod_name][resampler_name],
                        self.mode,
                        _times,
                    )

                    output[bf_name][apod_name][resampler_name] = reconstruction

        return output
