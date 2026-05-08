from dataclasses import dataclass
from pathlib import Path
import shutil

from .packing import ExperimentPacking

@dataclass(frozen=True)
class ModelPaths:
    backup: Path
    best: Path
    epochs: Path
    logs: Path

@dataclass(frozen=True)
class DatasetPaths:
    raw: Path
    ground_truth: Path
    webdataset_beamformer: str
    samples_idx: Path
    img: Path

class PathCenter():
    def __init__(self, location="local"):
        server_dataset_base = Path("/mnt/workspace/sgutierrezm/deep_bf/dataset")
        server_models_base = Path("/mnt/workspace/sgutierrezm/deep_bf/models")

        local_dataset_base = Path("/home/panda/rf_data/dataset")
        local_models_base = Path("/home/panda/code/usm/deep_bf/models/")

        if location == "server":
            self.dataset_base = server_dataset_base
            self.models_base = server_models_base
            self.location = "server"
        else:
            self.dataset_base = local_dataset_base
            self.models_base = local_models_base
            self.location = "local"

        raw = self.dataset_base / "raw" 
        ground_truth = self.dataset_base / "ground_truth" 

        webdataset_beamformer = self.dataset_base / "webdataset_beamformer"
        samples_idx = webdataset_beamformer / "samples_idx"

        img = self.dataset_base / "img"

        self.dataset_paths = DatasetPaths(raw, ground_truth, str(webdataset_beamformer), samples_idx, img)
        self.dl = "/home/panda/rf_data"

    def get_model_paths(self, config: ExperimentPacking):
        mC = config.model
        aC = mC.architecture_configs[0]
        dtC = config.webdataset_beamformer.data_type_config

        model_group = f"{aC.family}-{aC.model_id}"
        model_description = f"e{config.id}-{dtC.type}"
        # TODO: Dejar un nombre mas completo con mas ids y el hash del commit

        model_base_dir = self.models_base / self.location / model_group / model_description

        backup = model_base_dir / "backup"
        backup.mkdir(parents=True, exist_ok=True)

        best = model_base_dir / "best"
        best.mkdir(parents=True, exist_ok=True)

        epochs = model_base_dir / "epochs"
        epochs.mkdir(parents=True, exist_ok=True)

        logs = model_base_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)

        return ModelPaths(backup, best, epochs, logs)

    def reset_backup(self, model_paths: ModelPaths):
        print("Reseting backup checkpoint")
        p = model_paths.backup
        shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True, exist_ok=True)
        print("Reseting done")

    def close(self) -> None:
        pass
    def __enter__(self) -> "PathCenter":
        return self
    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
