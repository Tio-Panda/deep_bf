import webdataset as wds
import random
from torch.utils.data import DataLoader

from ..config_registery import HyperparametersConfig, PathCenter, WebDatasetBeamformerPack

def prepare(sample):
    sample["sii.txt"] = int(sample["sii.txt"])
    sample["angle.txt"] = int(sample["angle.txt"])
    return sample

def define_dataset(urls, batch_size, seed, is_train=True):
    dataset = wds.WebDataset(
        urls,
        nodesplitter=wds.split_by_node,
        workersplitter=wds.split_by_worker,
        resampled=False,
        shardshuffle=seed if is_train else False,
    )

    if is_train:
        dataset = dataset.shuffle(1000, rng=random.Random(seed))

    dataset = (
        dataset.decode("torch")
        .map(prepare)
        .to_tuple("data.npy", "sii.txt", "angle.txt", "gt.npy", "__key__")
        .batched(batch_size, partial=not is_train)
    )

    return dataset

def get_datasets(hC: HyperparametersConfig, wdbP: WebDatasetBeamformerPack, num_workers, pin_memory, mode="general", location="local"):
    seed = hC.seed
    batch_size = hC.batch_size

    with PathCenter(location=location) as pc:
        wp = pc.get_webdataset_beamformer_paths(wdbP, mode=mode)
        train_path = wp.train_path
        val_path = wp.val_path
    
    urls = []
    for _path in [train_path, val_path]:
        n = len(list(_path.glob("*.tar")))
        last_idx = f"{n-1:03d}"
        urls.append(f"{str(_path)}/dataset-{{000..{last_idx}}}.tar")

    train_urls, val_urls = urls

    train_ds = define_dataset(
        train_urls, batch_size=batch_size, is_train=True, seed=seed
    )
    val_ds = define_dataset(val_urls, batch_size=batch_size, is_train=False, seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=None, num_workers=1, persistent_workers=True
    )

    return train_loader, val_loader
