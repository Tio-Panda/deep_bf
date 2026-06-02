from dataclasses import dataclass
from typing import Any


@dataclass
class DataTypeConfig:
    id: int
    type: str
    params: dict[str, Any]


@dataclass
class ApodConfig:
    id: int
    type: str
    params: dict[str, Any]


@dataclass
class ResamplerConfig:
    id: int
    type: str
    params: dict[str, Any]


@dataclass
class BeamformerConfig:
    id: int
    type: str
    params: dict[str, Any]


@dataclass
class CompoundingConfig:
    id: int
    type: str
    params: dict[str, Any]


@dataclass
class BeamformerSetup:
    id: int
    data_type_config: DataTypeConfig
    beamformer_config: BeamformerConfig
    resampler_config: ResamplerConfig
    compounding_config: CompoundingConfig
    apod_config: ApodConfig
