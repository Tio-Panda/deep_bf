import random
import torch
import numpy as np
import h5py
import hdf5plugin
import importlib.resources
import pandas as pd
from pathlib import Path
import shutil

# from ..wrapper.config import WebDatasetConfig
# from ..beamformers import compute_samples_idx_by_angles, compute_meshgrid, compute_d_rx

# from ..data_handler import DataLoader

# DATALOADER_PATH = "/home/panda/rf_data/"
# DL = DataLoader(DATALOADER_PATH)

GROUP_KEYS = ["na", "fs", "aperture_width", "element_width", "pitch", "nc", "zlims", "fc"]

def get_names_groups(query):
    with importlib.resources.files("deep_bf.data_handler.data").joinpath("data.csv").open("r") as f:
        df = pd.read_csv(f)

        df = df.query(query)
        df = df.sort_values("name")

    group = df.groupby(GROUP_KEYS)
    return group
