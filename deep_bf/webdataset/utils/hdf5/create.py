from __future__ import annotations

import h5py
import hdf5plugin

from ....config_registery import PathCenter
from ....wrapper.reconstruction.reconstruction import Reconstruction

def save_ground_truth_hdf5(reconstruction: Reconstruction, location="local"):
    name = reconstruction.name
    data = reconstruction.data
    data_type = reconstruction.mode
    bfC = reconstruction.beamformer_config
    rC = reconstruction.resampler_config
    aC = reconstruction.apod_config
    cC = reconstruction.compounding_config
    dC = reconstruction.data_size_config

    with PathCenter(location=location) as pc:
        path = f"{pc.dataset_paths.ground_truth}/{name}.hdf5"
    
    group_name = f"{data_type}/{bfC.type}/{rC.type}/{aC.type}/{cC.type}/{dC.nz}x{dC.nx}"

    with h5py.File(path, "a") as f:
        g = f.require_group(group_name)

        if "ground_truth" in g:
            del g["ground_truth"] 

        g.create_dataset("ground_truth", data=data, chunks=True, **hdf5plugin.Bitshuffle(nelems=0, cname="zstd"))
