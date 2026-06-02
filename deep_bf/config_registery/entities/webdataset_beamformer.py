from dataclasses import dataclass
from typing import Any

from .beamformer import BeamformerSetup


@dataclass
class DataSizeConfig:
    id: int
    nz: int
    nx: int
    ns: int


@dataclass
class DataPreprocessingConfig:
    id: int
    type: str
    params: dict[str, Any]


@dataclass
class SamplesOrganizationConfig:
    id: int
    seed: int
    ratio: float
    order: str
    select_mode: str
    n_train: int
    n_val: int
    query: str
    train_idxs: str
    val_idxs: str


@dataclass
class WebDatasetBeamformerPack:
    id: int
    beamformer_setup: BeamformerSetup
    data_size_config: DataSizeConfig
    data_preprocessing_config: DataPreprocessingConfig
    samples_organization_config: SamplesOrganizationConfig
