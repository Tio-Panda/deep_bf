import numpy as np
from skimage.transform import resize
import h5py
import hdf5plugin
from dataclasses import asdict
import json

from ...config_registery import WebDatasetBeamformerPacking, PathCenter

def resize_gt(gt, new_nz, new_nx, mode="reflect"):
    new_gt = resize(
        gt,
        (new_nz, new_nx),
        order=3,
        mode=mode,
        anti_aliasing=True,
        preserve_range=True
    )
    return np.asarray(new_gt, dtype=np.float32)

RESIZE_GT_ORIGINAL = "original"
RESIZE_GT_RESIZE = "resize"
def get_transformed_ground_truth(name, config: WebDatasetBeamformerPacking, location="local"):
    dsC = config.data_size_config
    nz = dsC.nz
    nx = dsC.nx
    data_type = config.data_type_config.type
    data_transformation = config.transform_data_config.type
    source = config.webdataset_beamformer_config.gt_source

    gt, _ = get_hdf5_ground_truth(name, data_type, data_transformation, nz, nx, source, location)

    resize_mode = config.resize_gt_config.type
    if resize_mode == RESIZE_GT_RESIZE:
        params = config.resize_gt_config.params
        gt = resize_gt(gt, **params)

    return gt

def save_hdf5_ground_truth(data, times, name, data_type, data_transformation, nz, nx, source, location="local"):
    with PathCenter(location=location) as pc:
        path = f"{pc.dataset_paths.ground_truth}/{name}.hdf5"

    times = asdict(times)
    times_str = json.dumps(times)

    with h5py.File(path, "a") as f:
        g = f.require_group(f"{data_type}/{data_transformation}/{nz}x{nx}/{source}")

        if 'times' in g:
            del g['times']
        else:
            g.create_dataset('times', data=times_str, dtype=h5py.string_dtype(encoding='utf-8'))

        if 'ground_truth' in g:
            del g['ground_truth']
        else:
            g.create_dataset("ground_truth", data=data, chunks=True, **hdf5plugin.Bitshuffle(nelems=0, cname="zstd"))

def get_hdf5_ground_truth(name, data_type, data_transformation, nz, nx, source, location="local"):
    with PathCenter(location=location) as pc:
        path = f"{pc.dataset_paths.ground_truth}/{name}.hdf5"

    with h5py.File(path, "a") as f:
        g = f.require_group(f"{data_type}/{data_transformation}/{nz}x{nx}/{source}")
        data = g["ground_truth"][:]
        times = g["times"]

    return data, times
