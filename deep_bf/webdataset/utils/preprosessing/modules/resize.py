# from __future__ import annotations
#
# import json
# from dataclasses import asdict
#
# import h5py
# import hdf5plugin
# import numpy as np
# from skimage.transform import resize

# def resize_gt(gt, new_nz, new_nx, mode="reflect"):
#     new_gt = resize(
#         gt,
#         (new_nz, new_nx),
#         order=3,
#         mode=mode,
#         anti_aliasing=True,
#         preserve_range=True
#     )
#     return np.asarray(new_gt, dtype=np.float32)

# TODO: Revisar esto para mandarlo al sistema de variables globales

# RESIZE_GT_ORIGINAL = "original"
# RESIZE_GT_RESIZE = "resize"
# def get_transformed_ground_truth(name, config: WebDatasetBeamformerPacking, location="local"):
#     dsC = config.data_size_config
#     nz = dsC.nz
#     nx = dsC.nx
#     data_type = config.data_type_config.type
#     data_transformation = config.transform_data_config.type
#     source = config.webdataset_beamformer_config.gt_source
#
#     gt, _ = get_hdf5_ground_truth(name, data_type, data_transformation, nz, nx, source, location)
#
#     resize_mode = config.resize_gt_config.type
#     if resize_mode == RESIZE_GT_RESIZE:
#         params = config.resize_gt_config.params
#         gt = resize_gt(gt, **params)
#
#     return gt
