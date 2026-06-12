from __future__ import annotations

from contextlib import contextmanager
from typing import List

import numpy as np
import torch
import torch.nn as nn

from ...beamformers.apod.apod import apply_apodization
from ...beamformers.apod.apod_builder import apod_builder
from ...beamformers.bf.bf_builder import bf_builder
from ...beamformers.tofc.tofc import ToFCClassic
from ...beamformers.utils.upsampler import RFUpsampler1D
from ...config_registery import (
    ApodConfig,
    BeamformerConfig,
    CompoundingConfig,
    DataSizeConfig,
    ResamplerConfig,
)
from ...constants.bf import BeamformerType, CompooundingType
from ...wrapper.timer import Timer
from ..reconstruction import Reconstruction, ReconstructionCatalog


class ClassicBench2(nn.Module):
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

    def _angle_indices(self) -> np.ndarray:
        if self.angles[0] == -1:
            return np.arange(self.pw.na)
        return np.asarray(self.angles, dtype=int)

    def _validate_pre_bf_compounding(self, cC: CompoundingConfig):
        if cC.type not in (
            CompooundingType.NONE,
            CompooundingType.CPWC_SUM,
            CompooundingType.CPWC_MEAN,
        ):
            raise NotImplementedError(
                "ClassicBench2 applies compounding before beamforming and currently "
                "supports only NONE, CPWC_SUM and CPWC_MEAN."
            )

    def _compound_batch(self, compound_tofc, tofc_data: torch.Tensor):
        batch_sum = torch.sum(tofc_data, dim=0)
        if compound_tofc is None:
            compound_tofc = torch.zeros_like(batch_sum)
        compound_tofc += batch_sum
        return compound_tofc

    def _compound_raw_rf(
        self, angle_indices: np.ndarray, cC: CompoundingConfig
    ) -> torch.Tensor:
        self._validate_pre_bf_compounding(cC)

        raw_rf = None
        n_angles = 0
        for s in range(0, len(angle_indices), self.angle_batch_size):
            e = min(s + self.angle_batch_size, len(angle_indices))
            angles = angle_indices[s:e]
            raw_batch = self.pw.data[angles].to(self.device)
            batch_sum = torch.sum(raw_batch, dim=0)
            raw_rf = batch_sum if raw_rf is None else raw_rf + batch_sum
            n_angles += e - s

        if raw_rf is None or n_angles == 0:
            raise ValueError("CB requires at least one angle to compound RAW RF")

        if cC.type == CompooundingType.CPWC_MEAN:
            raw_rf = raw_rf / n_angles

        return raw_rf.unsqueeze(0)

    def _run_cb_pipeline(self, catalog: ReconstructionCatalog, bfC: BeamformerConfig):
        if len(self.resamplers) == 0:
            raise ValueError("ClassicBench2 requires at least one resampler config for CB metadata")
        if len(self.apods) == 0:
            raise ValueError("ClassicBench2 requires at least one apod config for CB metadata")

        n_angles = int(self.pw.na if hasattr(self.pw, "na") else self.pw.n_angles)
        angle_indices = np.asarray([n_angles // 2], dtype=int)
        rC = self.resamplers[0]
        aC = self.apods[0]

        for dsC in self.data_sizes:
            nz = dsC.nz
            nx = dsC.nx

            bf = bf_builder(bfC, self.pw, nz=nz, nx=nx).to(self.device)

            for cC in self.compoundings:
                self._validate_pre_bf_compounding(cC)
                times = []

                compound_timer = Timer()
                with compound_timer.measure(f"{cC.type}_{nz}"):
                    if cC.type == CompooundingType.NONE:
                        raw_rf = self.pw.data[angle_indices[0]].to(self.device).unsqueeze(0)
                    else:
                        raw_rf = self._compound_raw_rf(angle_indices, cC)

                timer = Timer()
                with timer.measure(f"{bfC.type}_{nz}"):
                    with torch.no_grad():
                        _data = bf(raw_rf)

                times.append(np.array([compound_timer.times[f"{cC.type}_{nz}"]]))
                times.append(np.array([timer.times[f"{bfC.type}_{nz}"]]))

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

    def _run_bf_pipeline(self, catalog: ReconstructionCatalog, bfC: BeamformerConfig):
        if bfC.type == BeamformerType.CB:
            self._run_cb_pipeline(catalog, bfC)
            return

        bf_type = bfC.type
        bf = bf_builder(bfC, self.pw).to(self.device)
        angle_indices = self._angle_indices()
        needs_center_tofc = any(
            cC.type == CompooundingType.NONE for cC in self.compoundings
        )

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

                for rC in self.resamplers:
                    resampler_type = rC.type
                    self.tofc_gen.set_resampler(resampler_type)

                    compound_tofc = None
                    center_tofc = None
                    times = []
                    n_angles = 0
                    center_angle = int(self.pw.na // 2)

                    for s in range(0, len(angle_indices), self.angle_batch_size):
                        e = min(s + self.angle_batch_size, len(angle_indices))
                        angles = angle_indices[s:e]

                        tofc_data, tofc_times = self.tofc_gen(angles)
                        tofc_data = apply_apodization(tofc_data, apod)
                        if needs_center_tofc:
                            center_idx = np.where(angles == center_angle)[0]
                            if center_idx.size > 0:
                                center_tofc = tofc_data[int(center_idx[0])]
                        compound_tofc = self._compound_batch(compound_tofc, tofc_data)
                        n_angles += e - s

                        _times = (
                            np.array(tofc_times)
                            + np.array([self.timer.times[f"{apod_type}_{nz}"]])
                            * (e - s)
                        )
                        times.append(_times)

                    if needs_center_tofc and center_tofc is None:
                        center_angles = np.asarray([center_angle], dtype=int)
                        tofc_data, _ = self.tofc_gen(center_angles)
                        center_tofc = apply_apodization(tofc_data, apod).squeeze(0)

                    for cC in self.compoundings:
                        self._validate_pre_bf_compounding(cC)
                        if cC.type == CompooundingType.NONE:
                            bf_input = center_tofc
                        elif cC.type == CompooundingType.CPWC_MEAN:
                            bf_input = compound_tofc
                            bf_input = bf_input / n_angles
                        else:
                            bf_input = compound_tofc

                        timer = Timer()
                        with timer.measure(f"{bf_type}_{nz}"):
                            with torch.no_grad():
                                _data = bf(bf_input.unsqueeze(0)).squeeze(0)

                        times.append(np.array([timer.times[f"{bf_type}_{nz}"]]))

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
