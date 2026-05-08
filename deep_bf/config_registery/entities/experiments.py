from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    id: int
    version: int
    webdataset_beamformer_id: int
    trainloop_id: int
    model_id: int
    commit_hash: str = "unknown"
    commit_msg: str = ""
