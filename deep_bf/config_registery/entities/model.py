from dataclasses import dataclass
from typing import Any


@dataclass
class ModelConfig:
    id: int
    family: str
    model_id: int
    conv2d_init_id: int
    activation_id: int
    beamformer_id: int


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
