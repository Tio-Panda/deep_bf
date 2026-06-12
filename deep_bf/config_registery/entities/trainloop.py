from dataclasses import dataclass
from typing import Any


@dataclass
class CriterionConfig:
    id: int
    type: str
    params: dict[str, Any]


@dataclass
class OptimizerConfig:
    id: int
    type: str
    params: dict[str, Any]


@dataclass
class SchedulerConfig:
    id: int
    type: str
    params: dict[str, Any]


@dataclass
class HyperparametersConfig:
    id: int
    seed: int
    n_epoch: int
    batch_size: int
    learning_rate: float


@dataclass
class TrainLoopSetup:
    id: int
    criterion_config: CriterionConfig
    optimizer_config: OptimizerConfig
    scheduler_config: SchedulerConfig
    hyperparameters_config: HyperparametersConfig
