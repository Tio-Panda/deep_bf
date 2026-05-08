from dataclasses import dataclass
from typing import Any


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
    resampler_id: int
    params: dict[str, Any]


@dataclass
class CompoundingConfig:
    id: int
    type: str
    params: dict[str, Any]
