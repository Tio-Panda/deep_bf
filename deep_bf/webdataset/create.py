import json
from pathlib import Path
from dataclasses import asdict

from .gsi.gsi_for_training import GlobalSamplesIdxForTraining
from .utils.split import beamformer_split_names
from .utils.shard_writer import beamformer_shard_writer

from ..config_registery import WebDatasetBeamformerPacking, PathCenter

def create_webdataset_beamformer(config: WebDatasetBeamformerPacking, location="local"):
    with PathCenter(location=location) as pc:
        webdataset_path = pc.dataset_paths.webdataset_beamformer
        train_path = Path(webdataset_path) / "train"
        val_path = Path(webdataset_path) / "val"

        metadata_path = (
            Path(pc.dataset_paths.webdataset_beamformer) / "metadata.json"
        )
    delete_samples_idx_folder = False if metadata_path.exists() else True

    with open(metadata_path, "r") as archivo:
        metadata = json.load(archivo)
        metadata_dataset_id = metadata["webdataset_beamformer_config"]["id"]
        dataset_id = config.webdataset_beamformer_config.id

        if metadata_dataset_id != dataset_id:
            delete_samples_idx_folder = True

    train_names, val_names = beamformer_split_names(config, location)
    
    gsi = GlobalSamplesIdxForTraining(config, cache_limit=12, location=location, reset=delete_samples_idx_folder)

    beamformer_shard_writer(gsi, train_path, train_names, config, location)
    beamformer_shard_writer(gsi, val_path, val_names, config, location)

    metadata = asdict(config)
    with open(f"{webdataset_path}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
