from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from typing import List

from ...beamformers.apod.apod_builder import apod_builder
from ...beamformers.bf.bf_builder import bf_builder
from ...beamformers.tofc.tofc import ToFCRealTime
from ...beamformers.compounding.compounder_builder import compounder_builder

from ...wrapper.timer import Timer

from ...config_registery import (
    BeamformerConfig,
    ResamplerConfig,
    ApodConfig,
    DataSizeConfig,
    CompoundingConfig,
)
from ...constants.bf import PWDataType
from ..reconstruction import Reconstruction, ReconstructionCatalog

class ClassicBench(nn.Module):
    def __init__(
        self,
        pw,
        data_sizes: List[DataSizeConfig],
        bfs: List[BeamformerConfig],
        resamplers: List[ResamplerConfig],
        apods: List[ApodConfig],
        compoundings: List[CompoundingConfig],
        angles: List[int] = [-1],
        angle_batch_size=5,
        device="cuda",
    ):
        super().__init__()
        self.pw = pw
        if isinstance(self.pw.data, np.ndarray):
            self.pw.data = torch.from_numpy(self.pw.data)

        self.data_sizes = data_sizes
        self.bfs = bfs
        self.apods = apods
        self.resamplers = resamplers
        self.compoundings = compoundings

        self.angles = angles
        self.angle_batch_size = angle_batch_size
        self.device = device
        self.timer = Timer()

    def forward(self):
        catalog = ReconstructionCatalog()

        for dsC in self.data_sizes:
            nz = dsC.nz
            nx = dsC.nx

            meshgrid_nyquist = nz == -1
            self.tofc_gen = ToFCRealTime(self.pw, self.resamplers, nz, nx, meshgrid_nyquist, self.angle_batch_size).to(self.device)

            for aC in self.apods:
                apod_type = aC.type
                with self.timer.measure(f"{apod_type}_{nz}"):
                    Z, X = self.tofc_gen.get_ZX()
                    apod = apod_builder(Z, X, self.pw.probe_geometry, aC)

                for bfC in self.bfs:
                    bf_type = bfC.type
                    bf = bf_builder(bfC, self.pw.type).to(self.device)
                    for rC in self.resamplers:
                        resampler_type = rC.type

                        self.tofc_gen.set_resampler(resampler_type)

                        times = []
                        if self.angles[0] == -1:
                            if self.pw.type != PWDataType.IQ_SPLIT:
                                # TODO: implementar ya que float32 no sirviria para Complex32
                                bf_data = torch.zeros(self.pw.na, nz, nx, device=self.device, dtype=self.pw.data.dtype)
                            else:
                                bf_data = torch.zeros(self.pw.na, nz, nx, 2, device=self.device, dtype=self.pw.data.dtype)

                            for s in range(0, self.pw.na, self.angle_batch_size):
                                e = min(s + self.angle_batch_size, self.pw.na)
                                angles = np.arange(s, e)

                                tofc_data, tofc_times = self.tofc_gen(angles)

                                timer = Timer()
                                with timer.measure(f"{bf_type}_{nz}"):
                                    with torch.no_grad():
                                        bf_data[s:e] = bf(tofc_data, apod)

                                _times = (
                                    np.array(tofc_times)
                                    + np.array([timer.times[f"{bf_type}_{nz}"] / (e - s)])
                                    * (e - s)
                                    + np.array([self.timer.times[f"{apod_type}_{nz}"]])
                                    * (e - s)
                                )
                                times.append(_times)

                            for cC in self.compoundings:
                                cmp = compounder_builder(cC, self.pw)
                                _data = cmp(bf_data)

                                reconstruction = Reconstruction(
                                    self.pw,
                                    _data,
                                    self.pw.type,
                                    times,
                                    data_size_config=dsC,
                                    beamformer_config=bfC,
                                    resampler_config=rC,
                                    apod_config=aC,
                                    compounding_config=cC,
                                )

                                catalog.add(reconstruction)

        return catalog
