import webdataset as wds
import random
from torch.utils.data import DataLoader
import numpy as np

def prepare(sample):
    sample["sii.txt"] = int(sample["sii.txt"])
    sample["rf.npy"] = np.expand_dims(sample["rf.npy"], axis=-1)
    sample["gt.npy"] = np.expand_dims(sample["gt.npy"], axis=-1)

    return sample

def define_dataset(urls, batch_size, seed, is_train=True):

    dataset = (
        wds.WebDataset(
            urls,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            resampled=is_train,
            shardshuffle=False
        )
    )

    if is_train:
        dataset = dataset.shuffle(1000, rng=random.Random(seed))

    dataset = (
        dataset
        .decode("torch")
        .map(prepare)
        .batched(batch_size)
        .to_tuple("rf.npy", "sii.txt", "gt.npy", "__key__")
    )

    return dataset

def get_datasets(base_url, seed, batch_size=1, num_workers=4, pin_memory=True):
    train_urls = base_url + "/train/dataset-{000..004}.tar"
    val_urls = base_url + "/val/dataset-{000..001}.tar"


    train_ds = define_dataset(train_urls, batch_size=batch_size, is_train=True, seed=seed)
    val_ds = define_dataset(val_urls, batch_size=batch_size, is_train=False, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=None, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=num_workers)

    return train_loader, val_loader
