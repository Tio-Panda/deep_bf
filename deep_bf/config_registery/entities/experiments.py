from dataclasses import dataclass

from .model import ModelPack
from .trainloop import TrainLoopSetup
from .webdataset_beamformer import WebDatasetBeamformerPack


@dataclass
class Experiment:
    id: int
    description: str
    model_pack: ModelPack
    trainloop_setup: TrainLoopSetup
    webdataset_beamformer_pack: WebDatasetBeamformerPack
