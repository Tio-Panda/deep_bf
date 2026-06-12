from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from typing import List

from ...beamformers.apod.apod import apply_apodization
from ...beamformers.apod.apod_builder import apod_builder
from ...beamformers.bf.bf_builder import bf_builder
from ...beamformers.tofc.tofc import ToFCClassic
from ...beamformers.compounding.compounder_builder import compounder_builder

from ...wrapper.timer import Timer

from ...config_registery import (
    BeamformerConfig,
    ResamplerConfig,
    ApodConfig,
    DataSizeConfig,
    CompoundingConfig,
)

from ...data_handler import DataLoader
from ...constants.bf import PWDataType
from ..reconstruction import Reconstruction, ReconstructionCatalog
from ...config_registery import PathCenter

class FullClassicBench(nn.Module):
    def __init__(
        self,
        names,
        data_type,
        data_sizes: List[DataSizeConfig],
        bfs: List[BeamformerConfig],
        resamplers: List[ResamplerConfig],
        apods: List[ApodConfig],
        compoundings: List[CompoundingConfig],
        angles: List[int] = [-1],
        angle_batch_size=5,
        device="cuda",
        location="local"
    ):
        super().__init__()
        self.names = names
        self.data_type = data_type
        with PathCenter(location) as pc:
            dl_path = pc.dl
            self.dl = DataLoader(dl_path)

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
        
        for name in self.names:

            pw = self.dl.get_defined_pwdata(name, self.data_type)
            if isinstance(pw.data, np.ndarray):
                pw.data = torch.from_numpy(pw.data)

            for dsC in self.data_sizes:
                nz = dsC.nz
                nx = dsC.nx

                meshgrid_nyquist = nz == -1
                self.tofc_gen = ToFCClassic(pw, self.resamplers, nz, nx, meshgrid_nyquist, self.angle_batch_size).to(self.device)

                for aC in self.apods:
                    apod_type = aC.type
                    with self.timer.measure(f"{name}_{apod_type}_{nz}"):
                        Z, X = self.tofc_gen.get_ZX()
                        apod = apod_builder(Z, X, pw.probe_geometry, aC)

                    for bfC in self.bfs:
                        bf_type = bfC.type
                        bf = bf_builder(bfC, pw).to(self.device)
                        for rC in self.resamplers:
                            resampler_type = rC.type

                            self.tofc_gen.set_resampler(resampler_type)

                            times = []
                            if self.angles[0] == -1:
                                if pw.type != PWDataType.IQ_SPLIT:
                                    # TODO: implementar ya que float32 no sirviria para Complex32
                                    bf_data = torch.zeros(pw.na, nz, nx, device=self.device, dtype=pw.data.dtype)
                                else:
                                    bf_data = torch.zeros(pw.na, nz, nx, 2, device=self.device, dtype=pw.data.dtype)

                                for s in range(0, pw.na, self.angle_batch_size):
                                    e = min(s + self.angle_batch_size, pw.na)
                                    angles = np.arange(s, e)

                                    tofc_data, tofc_times = self.tofc_gen(angles)

                                    timer = Timer()
                                    with timer.measure(f"{name}_{bf_type}_{nz}"):
                                        with torch.no_grad():
                                            tofc_data = apply_apodization(tofc_data, apod)
                                            bf_data[s:e] = bf(tofc_data)

                                    _times = (
                                        np.array(tofc_times)
                                        + np.array([timer.times[f"{name}_{bf_type}_{nz}"] / (e - s)])
                                        * (e - s)
                                        + np.array([self.timer.times[f"{name}_{apod_type}_{nz}"]])
                                        * (e - s)
                                    )
                                    times.append(_times)

                                for cC in self.compoundings:

                                    # TODO: Implementar esto mejor
                                    if cC.type != "NONE":
                                        cmp = compounder_builder(cC, pw)
                                        _data = cmp(bf_data)
                                    else:
                                        center_angle = pw.na // 2
                                        _data = bf_data[center_angle]

                                    reconstruction = Reconstruction(
                                        pw,
                                        _data,
                                        pw.type,
                                        times,
                                        data_size_config=dsC,
                                        beamformer_config=bfC,
                                        resampler_config=rC,
                                        apod_config=aC,
                                        compounding_config=cC,
                                    )

                                    catalog.add(reconstruction)

        return catalog
