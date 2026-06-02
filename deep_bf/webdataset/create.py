import webdataset as wds
import shutil

import json
from pathlib import Path
from dataclasses import asdict

from ..config_registery import WebDatasetBeamformerPack, PathCenter

from .utils.preprosessing.preprosessing import get_preprosessed_input, get_preprosessed_gt
from .utils.split.split import get_webdataset_train_val_names

def webdataset_beamformer_shard_writer(path, names, config: WebDatasetBeamformerPack, maxcount=100, location="local"):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

    with wds.ShardWriter(f"{path}/dataset-%03d.tar", maxcount=maxcount) as sink:
        for name in names:
            aug_inputs = get_preprosessed_input(name, config, location=location)
            gt = get_preprosessed_gt(name, config, location=location)

            for pack in aug_inputs:
                sii, angle, data = pack.values()
                sink.write(
                    {
                        "__key__": f"{name}_{angle}",
                        "data.npy": data,
                        "gt.npy": gt,
                        "angle.txt": str(angle),
                        "sii.txt": str(sii)
                    }
                )

def create_webdataset_beamformer(config: WebDatasetBeamformerPack, mode="general", location="local"):
    with PathCenter(location=location) as pc:
        wp = pc.get_webdataset_beamformer_paths(config, mode=mode)
        train_path = wp.train_path
        val_path = wp.val_path
        metadata_path = wp.metadata_path
    
    train_names, val_names = get_webdataset_train_val_names(config, location=location)

    webdataset_beamformer_shard_writer(train_path, train_names, config, maxcount=100, location=location)
    webdataset_beamformer_shard_writer(val_path, val_names, config, maxcount=100, location=location)

    metadata = asdict(config)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
