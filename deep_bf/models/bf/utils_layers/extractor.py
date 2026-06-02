import numpy as np
import h5py
import hdf5plugin
import torch
import torch.nn as nn
from collections import OrderedDict

import importlib.resources
import pandas as pd

from ....config_registery import PathCenter
from ....data_handler import DataLoader
from ....constants.bf import PWDataType
from ....beamformers.utils.delays import compute_d_rx, compute_meshgrid, compute_samples_idx_by_angles

GROUP_KEYS = ["na", "fs", "aperture_width", "element_width", "pitch", "nc", "zlims", "fc"]

def get_name2id_dic(query):
    with importlib.resources.files("deep_bf.data_handler.data").joinpath("data.csv").open("r") as f:
        df = pd.read_csv(f)

        df = df.query(query)
        df = df.sort_values("name")

    groups = df.groupby(GROUP_KEYS)

    id = 0
    name2id = {}

    for _, mini_df in groups:
        group_names = mini_df["name"]

        for name in group_names:
            name2id[name] = id

        id += 1

    return name2id

def save_samples_idx_hdf5(nz, nx, query, location="local", verbose=False):
    with PathCenter(location=location) as pc:
        dl = DataLoader(str(pc.dataset_paths.raw))
        samples_idx_path = pc.dataset_paths.samples_idx
    
    name2id = get_name2id_dic(query)

    seen_ids = set()
    for name, id in name2id.items():
        if id not in seen_ids:
            seen_ids.add(id)
            pw = dl.get_defined_pwdata(name, PWDataType.RF)
            if verbose: print(name)

            # TODO: implementar que con extra_angles expandimos en [-extra_angles, center, +extra_angles] para aug
            angles = np.array([pw.na // 2])
            
            Z, X = compute_meshgrid(pw, nz, nx, nyquist=False)
            d_rx = compute_d_rx(pw, Z, X)
            array_samples_idx = compute_samples_idx_by_angles(pw, Z, X, d_rx, angles)

            for angle, samples_idx in zip(angles, array_samples_idx):
                with h5py.File(f"{samples_idx_path}/{id}_{angle}.hdf5", "w") as f:
                    f.create_dataset(
                        "samples_idx",
                        data=samples_idx.cpu(),
                        chunks=True,
                        **hdf5plugin.Bitshuffle(nelems=0, cname="zstd"),
                    )
    

class SamplesIdxExtractor2(nn.Module):
    def __init__(self, location="local"):
        super().__init__()
        with PathCenter(location=location) as pc:
            self.samples_idx_path = pc.dataset_paths.samples_idx


    def forward(self, ids, angles):
        ids = ids.to(dtype=torch.int64).cpu()
        angles = angles.to(dtype=torch.int64).cpu()

        batch = []
        for id, angle in zip(ids, angles):
            with h5py.File(f"{self.samples_idx_path}/{id}_{angle}.hdf5", "r", swmr=True) as f:
                batch.append(torch.from_numpy(f["samples_idx"][:])) # (nz, nx)

        return torch.stack(batch, dim=0).to(device="cuda", dtype=torch.float32)  # (B, nz, nx)


class SamplesIdxExtractor(nn.Module):
    def __init__(self, location="local", device="cuda"):
        super().__init__()
        with PathCenter(location=location) as pc:
            self.samples_idx_path = pc.dataset_paths.samples_idx

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(
                f"SamplesIdxExtractorGPU requiere CUDA, recibido device={device}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA no esta disponible para SamplesIdxExtractorGPU")

        self.max_cache_size = 15
        self._gpu_cache: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()

    def _get_samples_idx_gpu(self, id_int: int, angle_int: int) -> torch.Tensor:
        key = (id_int, angle_int)

        cached = self._gpu_cache.get(key)
        if cached is not None:
            self._gpu_cache.move_to_end(key)
            return cached

        with h5py.File(
            f"{self.samples_idx_path}/{id_int}_{angle_int}.hdf5", "r", swmr=True
        ) as f:
            value_cpu = torch.from_numpy(f["samples_idx"][:]).to(
                dtype=torch.float32, device="cpu"
            )

        value_gpu = value_cpu.to(device=self.device, dtype=torch.float32)

        self._gpu_cache[key] = value_gpu
        self._gpu_cache.move_to_end(key)

        if len(self._gpu_cache) > self.max_cache_size:
            _, evicted = self._gpu_cache.popitem(last=False)
            del evicted

        return value_gpu

    def forward(self, ids, angles):
        # TODO: Es necesaria esta transformacion si del shard salen como numpy?
        ids_cpu = ids.to(dtype=torch.int64).cpu()
        angles_cpu = angles.to(dtype=torch.int64).cpu()

        batch = []
        for id_tensor, angle_tensor in zip(ids_cpu, angles_cpu):
            id_int = int(id_tensor.item())
            angle_int = int(angle_tensor.item())
            batch.append(self._get_samples_idx_gpu(id_int, angle_int))

        return torch.stack(batch, dim=0)
