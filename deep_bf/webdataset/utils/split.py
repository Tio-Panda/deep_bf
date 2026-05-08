import random
# from pathlib import Path
# import json
# from dataclasses import asdict

from ...data_handler import DataLoader
from ...config_registery import WebDatasetBeamformerPacking, PathCenter
from .idxs import decode_idxs

SPLIT_MODE_SELECT_IDX = "select_idxs"
SPLIT_MODE_RANDOM = "random_split"
def beamformer_split_names(config: WebDatasetBeamformerPacking, location="local"):
    soC = config.samples_organization_config
    with PathCenter(location=location) as pc:
        raw_path = str(pc.dataset_paths.raw)
        dl = DataLoader(raw_path)
        df = dl.get_df()

        query = soC.query
        df = df.query(query)
        df = df.sort_values("name")

        names = df["name"]

    if soC.select_mode == SPLIT_MODE_SELECT_IDX:
        train_idxs = soC.train_idxs
        train_names = decode_idxs(names, train_idxs)

        val_idxs = soC.val_idxs
        val_names = decode_idxs(names, val_idxs)
    else:
        seed = soC.seed
        rng = random.Random(seed)

        names = list(names)
        rng.shuffle(names)
        n = len(names)
        ratio = soC.ratio
        train = int(n * ratio)

        train_names = names[:train]
        val_names = names[train:]

    return train_names, val_names
