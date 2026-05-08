import json
import torch
import numpy as np
import h5py
import hdf5plugin
from pathlib import Path
from functools import lru_cache
import shutil

from .gsi_base import GlobalSamplesIdx
from ...config_registery import WebDatasetBeamformerPacking, PathCenter
from ...data_handler import DataLoader
from ...beamformers.utils.delays import compute_gsi_samples_idx


# TODO: Dejar estas variables globales en otro lado como ConfigRegistery para que no dependan del archivo,
# Tambien para que sirvan a la hora de agregar nuevas filas en las tablas.
RESIZE_GT_RESIZE = "resize"

class GlobalSamplesIdxForTraining(GlobalSamplesIdx):
    def __init__(
        self,
        config: WebDatasetBeamformerPacking,
        cache_limit=10,
        location="local",
        reset=False,
    ):
        dsC = config.data_size_config
        rC = config.resize_gt_config

        if rC.type == RESIZE_GT_RESIZE:
            self.nz = rC.params["new_nz"]
            self.nx = rC.params["new_nx"]
        else:
            self.nz = dsC.nz
            self.nx = dsC.nx

        self.cache_limit = cache_limit
        query = config.samples_organization_config.query
        super().__init__(query)

        self.config = config
        self.location = location

        with PathCenter(location=location) as pc:
            self.samples_idx_path = pc.dataset_paths.samples_idx

        if reset:
            print("Computing samples_idx")
            p = Path(self.samples_idx_path)
            shutil.rmtree(p, ignore_errors=True)
            p.mkdir(parents=True, exist_ok=True)

            self._compute_samples_idx()
            print("Computing samples_idx done")

        @lru_cache(maxsize=self.cache_limit)
        def _load_hdf5(id, angle):
            with h5py.File(
                f"{self.samples_idx_path}/{id}_{angle}.hdf5", "r", swmr=True
            ) as f:
                return f["samples_idx"][:]

        self._load_samples_idx = _load_hdf5

    def get_samples_idx(self, id, angle, device="cuda", dtype=torch.float32):
        sample = self._load_samples_idx(int(id), int(angle))
        sample = torch.from_numpy(sample).to(device=device, dtype=dtype)
        return sample

    def _compute_samples_idx(self):
        dsC = self.config.data_size_config
        rC = self.config.resize_gt_config

        if rC.type == RESIZE_GT_RESIZE:
            nz = rC.params["new_nz"]
            nx = rC.params["new_nx"]
        else:
            nz = dsC.nz
            nx = dsC.nx

        with PathCenter(location=self.location) as pc:
            dl = DataLoader(str(pc.dataset_paths.raw))

        id = 0
        for group_name, mini_df in self.groups:
            name = mini_df["name"].iloc[0]
            pw = dl.get_defined_pwdata(name, "RF")

            # TODO: implementar que con extra_angles expandimos en [-extra_angles, center, +extra_angles] para aug
            angles = np.array([pw.na // 2])

            samples, times = compute_gsi_samples_idx(pw, nz, nx, angles)

            for angle, sample, time in zip(angles, samples, times):
                with h5py.File(f"{self.samples_idx_path}/{id}_{angle}.hdf5", "w") as f:
                    f.create_dataset(
                        "samples_idx",
                        data=sample.cpu(),
                        chunks=True,
                        **hdf5plugin.Bitshuffle(nelems=0, cname="zstd"),
                    )
                    f.attrs["time"] = time

            id += 1
