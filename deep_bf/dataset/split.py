import random
from pathlib import Path
import webdataset as wds
import h5py
import numpy as np
import json
import shutil

from .utils import GlobalSamplesIdx

def shard_writer(gsi, path, files, transform):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

    with wds.ShardWriter(f"{path}/dataset-%03d.tar", maxcount=100) as sink:
        for file in files:
            name = file.stem
            with h5py.File(file, "r") as f:
                rf = f["rf"][:]

                if transform:
                    rf_max = rf / np.max(np.abs(rf))
                    sigma = np.std(rf_max) + 1e-8
                    rf = rf_max / sigma

                gt  = f["ground_truth"][:]
                samples_idx_id = str(gsi[name])

                sink.write({
                    "__key__": name,
                    "rf.npy": rf,
                    "gt.npy": gt,
                    "sii.txt": samples_idx_id
                })

def split_webdataset(raw_path, webdataset_path, seed=42, ratio=0.9, transform=True):
    metadata = { "seed": seed, "ratio": ratio, "transform": transform }
    rng = random.Random(seed)
    gsi = GlobalSamplesIdx()

    raw_path = Path(raw_path)
    webdataset_path = Path(webdataset_path)

    train_path = webdataset_path / "train"
    val_path = webdataset_path / "val"

    files = list(raw_path.glob("*.hdf5"))
    rng.shuffle(files)
    n_files = len(files)
    metadata["n_dataset"] = n_files

    train = int(n_files * ratio)
    train_files = files[:train]
    val_files = files[train:]
    metadata["n_train"] = len(train_files)
    metadata["n_val"] = len(val_files)

    shard_writer(gsi, train_path, train_files, transform)
    shard_writer(gsi, val_path, val_files, transform)

    with open(f"{webdataset_path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
