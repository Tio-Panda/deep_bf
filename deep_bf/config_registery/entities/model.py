from dataclasses import dataclass
from typing import Any

from .beamformer import BeamformerSetup


@dataclass
class Conv2dInitConfig:
    id: int
    init_weights: str
    init_bias: str


@dataclass
class ActivationConfig:
    id: int
    type: str
    params: dict[str, Any]


@dataclass
class ArchitectureCnnBfConfig:
    model_id: int
    family: str
    pos: int
    type: str
    ch_in: int
    ch_out: int
    kernel: tuple[int, int]
    padding: str
    bias: bool


@dataclass
class ModelPack:
    id: int
    family: str
    model_id: int
    conv2d_init_config: Conv2dInitConfig
    activation_config: ActivationConfig
    architecture_configs: list[ArchitectureCnnBfConfig]
    beamformer_setup: BeamformerSetup
