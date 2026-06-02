from __future__ import annotations

import h5py
import hdf5plugin
import numpy as np

from ....config_registery import PathCenter, BeamformerConfig, ResamplerConfig, ApodConfig, CompoundingConfig, DataSizeConfig

def get_ground_truth_hdf5(
        name, 
        data_type,
        bfC:BeamformerConfig, 
        rC: ResamplerConfig,
        aC: ApodConfig,
        cC: CompoundingConfig,
        dC: DataSizeConfig,
        location="local"
    ):

    with PathCenter(location=location) as pc:
        path = f"{pc.dataset_paths.ground_truth}/{name}.hdf5"
    
    group_name = f"{data_type}/{bfC.type}/{rC.type}/{aC.type}/{cC.type}/{dC.nz}x{dC.nx}"
    
    with h5py.File(path, "r") as f:
        g = f.require_group(group_name)
        data = g["ground_truth"][:]

    return data
