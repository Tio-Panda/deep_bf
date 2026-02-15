from pathlib import Path
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader

def split_data(root_path, factor, seed=42):
    root_dir = Path(root_path)
    files = np.array(sorted([f for f in root_dir.rglob('*.hdf5') if f.is_file()]))

    rng = np.random.default_rng(seed)
    rng.shuffle(files)

    n = int(len(files) * factor)
    train = files[:n].tolist()
    val = files[n:].tolist()

    return train, val

class DatasetRF(Dataset):
    def __init__(self, data_path, transform=True, dtype=np.float32):
        self.path = data_path
        self.transform = transform
        self.dtype = dtype

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        path = self.path[idx]

        with h5py.File(path, "r", swmr=True) as f:
            rf = np.array(f["rfdata"], dtype=self.dtype)
            grid = np.array(f["grid"], dtype=self.dtype)
            probe_geometry = np.array(f["probe_geometry"], dtype=self.dtype)
            img = np.array(f["img"], dtype=self.dtype)

            if self.transform:
                rf_max = rf / np.max(np.abs(rf))
                sigma = np.std(rf_max) + 1e-8
                rf = rf_max / sigma

            c0 = np.squeeze(f.attrs["c0"])
            fs = np.squeeze(f.attrs["fs"])
            t0 = np.squeeze(f.attrs["t0"])
            angle = np.squeeze(f.attrs["angle"])

            rf = rf[..., np.newaxis]
            params = np.array([c0, fs, t0, angle], dtype=self.dtype)

            inputs = {
                "rf": rf,
                "grid": grid,
                "probe": probe_geometry,
                "params": params
            }

            ground_truth = img

            return inputs, ground_truth

def get_dataset(data_path, factor=0.9, seed= 42, batch_size=1, transform=True, pin_memory=False, num_workers=4, dtype=np.float32):
    train_path, val_path = split_data(data_path, factor, seed)

    train_dataset = DatasetRF(train_path, transform, dtype)
    val_dataset = DatasetRF(val_path, transform, dtype)

    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

    return train, val
