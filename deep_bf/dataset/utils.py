import random
import torch
import numpy as np
import h5py
import hdf5plugin
import importlib.resources
import pandas as pd

def set_seed(seed, id):
    seed = seed + id
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

SAMPLES_IDX_PATH = "/home/panda/rf_data/dataset/samples_idx"
# SAMPLES_IDX_PATH = "/mnt/workspace/sgutierrezm/deep_bf/dataset/samples_idx"

class GlobalSamplesIdx():
    def __init__(self, samples_idx_path=SAMPLES_IDX_PATH):

        print("loading gsi")

        with importlib.resources.files("deep_bf.data_handler.data").joinpath("data.csv").open("r") as f:
            df = pd.read_csv(f)

            query = "RF == 1 and nc == 128 and source == 'CUBDL'"
            df = df.query(query)
            df = df[df["name"].str[:3] != "JHU"]
            # df = df[df["name"] != "OSL010"]
            df = df[df["name"].str[:3] != "UFL"]
            df = df.sort_values("name")

        self.df = df
        
        keys = ["na", "fs", "aperture_width", "element_width", "pitch", "nc", "zlims", "fc"]
        group = self.df.groupby(keys)

        id = 0
        matrices_idx = {}

        for _, mini_df in group:
            group_names = mini_df["name"]

            for name in group_names:
                matrices_idx[name] = id
            
            id += 1

        self.matrix = matrices_idx
        
        self.samples_idx_path = samples_idx_path

        all_ids = set(self.matrix.values())
        n = len(all_ids)

        with h5py.File(f"{self.samples_idx_path}/{0}.hdf5", "r", swmr=True) as f:
            nc, nz, nx = f["samples_idx"][:].shape
            samples_idx = torch.empty((n, nc, nz, nx))

        for id in all_ids:
            with h5py.File(f"{self.samples_idx_path}/{id}.hdf5", "r", swmr=True) as f:
                _samples_idx = torch.from_numpy(f["samples_idx"][:])
                samples_idx[id] = _samples_idx

        # self.samples_idx = samples_idx.share_memory_()
        self.samples_idx = samples_idx

        print("gsi loaded")
        # self.samples_idx = samples_idx


    def __getitem__(self, key):
        return self.matrix[key]

    def get_samples_idx(self, id):
        id = int(id)
        return self.samples_idx[id]
