import webdataset as wds
import random
from torch.utils.data import DataLoader
import numpy as np

def prepare(sample):
    sample["sii.txt"] = int(sample["sii.txt"])
    return sample

def define_dataset(urls, batch_size, seed, is_train=True):

    dataset = (
        wds.WebDataset(
            urls,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            # resampled=is_train,
            resampled=False,
            shardshuffle=seed if is_train else False
        )
    )

    if is_train:
        dataset = dataset.shuffle(1000, rng=random.Random(seed))

    dataset = (
        dataset
        .decode("torch")
        .map(prepare)
        .to_tuple("rf.npy", "sii.txt", "gt.npy", "__key__")
        .batched(batch_size, partial=not is_train)
    )

    return dataset

def get_datasets(base_url, seed, batch_size=1, num_workers=4, pin_memory=True):
    train_urls = base_url + "/train/dataset-{000..004}.tar"
    val_urls = base_url + "/val/dataset-000.tar"

    train_ds = define_dataset(train_urls, batch_size=batch_size, is_train=True, seed=seed)
    val_ds = define_dataset(val_urls, batch_size=batch_size, is_train=False, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=None, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=1, persistent_workers=True)

    return train_loader, val_loader
