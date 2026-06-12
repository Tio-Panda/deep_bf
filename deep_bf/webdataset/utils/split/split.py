import random
from ....data_handler import DataLoader
from ....config_registery import WebDatasetBeamformerPack, PathCenter
from ....constants.webdataset import SplitOptions
from .idx_fn import decode_idxs

def get_webdataset_train_val_names(config: WebDatasetBeamformerPack, location="local"):
    soC = config.samples_organization_config
    with PathCenter(location=location) as pc:
        raw_path = str(pc.dataset_paths.raw)
        dl = DataLoader(raw_path)
        df = dl.get_df()

        query = soC.query
        df = df.query(query)
        df = df.sort_values("name")

        names = df["name"]

    if soC.select_mode == SplitOptions.SPLIT_MODE_SELECT_IDX:
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
