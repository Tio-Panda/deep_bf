from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from contextlib import contextmanager

from typing import List

from ...beamformers.apod.apod import apply_apodization
from ...beamformers.apod.apod_builder import apod_builder
from ...beamformers.bf.bf_builder import bf_builder
from ...beamformers.tofc.tofc import ToFCClassic
from ...beamformers.compounding.compounder_builder import compounder_builder
from ...beamformers.utils.upsampler import RFUpsampler1D

from ...wrapper.timer import Timer

from ...config_registery import (
    BeamformerConfig,
    ResamplerConfig,
    ApodConfig,
    DataSizeConfig,
    CompoundingConfig,
)
from ...constants.bf import BeamformerType, PWDataType
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
    
        self.upsample_factor = 2
        self.upsample_numtaps = 65
        self.upsample_window = "hamming"

        self.angles = angles
        self.angle_batch_size = angle_batch_size
        self.device = device
        self.timer = Timer()

    @contextmanager
    def _temporary_fdmas_upsampled_pw(self):
        original_data = self.pw.data
        original_fs = self.pw.fs
        original_ns = self.pw.ns

        try:
            if self.upsample_factor > 1:
                upsampler = RFUpsampler1D(
                    factor=self.upsample_factor,
                    numtaps=self.upsample_numtaps,
                    window=self.upsample_window,
                )
                self.pw.data = upsampler(original_data)
                self.pw.fs = original_fs * self.upsample_factor
                self.pw.ns = original_ns * self.upsample_factor
            yield
        finally:
            self.pw.data = original_data
            self.pw.fs = original_fs
            self.pw.ns = original_ns

    def _run_bf_pipeline(
        self,
        catalog: ReconstructionCatalog,
        bfC: BeamformerConfig,
        build_bfC: BeamformerConfig | None = None,
    ):
        if build_bfC is None:
            build_bfC = bfC

        bf_type = bfC.type
        bf = bf_builder(build_bfC, self.pw).to(self.device)

        for dsC in self.data_sizes:
            nz = dsC.nz
            nx = dsC.nx

            meshgrid_nyquist = nz == -1
            self.tofc_gen = ToFCClassic(
                self.pw,
                self.resamplers,
                nz,
                nx,
                meshgrid_nyquist,
                self.angle_batch_size,
            ).to(self.device)

            for aC in self.apods:
                apod_type = aC.type
                with self.timer.measure(f"{apod_type}_{nz}"):
                    Z, X = self.tofc_gen.get_ZX()
                    apod = apod_builder(Z, X, self.pw.probe_geometry, aC)
                    # TODO: Crear una clase como Compounding que se encargue de la apodizacion

                for rC in self.resamplers:
                    resampler_type = rC.type

                    self.tofc_gen.set_resampler(resampler_type)

                    times = []
                    if self.angles[0] == -1:
                        if self.pw.type != PWDataType.IQ_SPLIT:
                            # TODO: implementar ya que float32 no sirviria para Complex32
                            bf_data = torch.zeros(
                                self.pw.na,
                                nz,
                                nx,
                                device=self.device,
                                dtype=self.pw.data.dtype,
                            )
                        else:
                            bf_data = torch.zeros(
                                self.pw.na,
                                nz,
                                nx,
                                2,
                                device=self.device,
                                dtype=self.pw.data.dtype,
                            )

                        for s in range(0, self.pw.na, self.angle_batch_size):
                            e = min(s + self.angle_batch_size, self.pw.na)
                            angles = np.arange(s, e)

                            tofc_data, tofc_times = self.tofc_gen(angles)

                            timer = Timer()
                            with timer.measure(f"{bf_type}_{nz}"):
                                with torch.no_grad():
                                    tofc_data = apply_apodization(tofc_data, apod)
                                    bf_data[s:e] = bf(tofc_data)

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

    def forward(self):
        catalog = ReconstructionCatalog()

        for bfC in self.bfs:
            if bfC.type == BeamformerType.FDMAS:
                with self._temporary_fdmas_upsampled_pw():
                    self._run_bf_pipeline(catalog, bfC)
            else:
                self._run_bf_pipeline(catalog, bfC)

        return catalog
