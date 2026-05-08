from dataclasses import dataclass

from .entities import (
    ActivationConfig,
    ArchitectureCnnBfConfig,
    BeamformerConfig,
    Conv2dInitConfig,
    CriterionConfig,
    DataSizeConfig,
    DataTypeConfig,
    HyperparametersConfig,
    ModelConfig,
    OptimizerConfig,
    ResizeGtConfig,
    ResamplerConfig,
    SamplesOrganizationConfig,
    SchedulerConfig,
    TransformDataConfig,
    TrainLoopConfig,
    WebDatasetBeamformerConfig,
)


@dataclass
class BeamformerPacking:
    beamformer_config: BeamformerConfig
    resampler_config: ResamplerConfig


@dataclass
class ModelPacking:
    model_config: ModelConfig
    conv2d_init_config: Conv2dInitConfig
    activation_config: ActivationConfig
    architecture_configs: list[ArchitectureCnnBfConfig]
    beamformer: BeamformerPacking


@dataclass
class TrainLoopPacking:
    trainloop_config: TrainLoopConfig
    criterion_config: CriterionConfig
    optimizer_config: OptimizerConfig
    scheduler_config: SchedulerConfig
    hyperparameters_config: HyperparametersConfig


@dataclass
class WebDatasetBeamformerPacking:
    webdataset_beamformer_config: WebDatasetBeamformerConfig
    data_type_config: DataTypeConfig
    data_size_config: DataSizeConfig
    samples_organization_config: SamplesOrganizationConfig
    transform_data_config: TransformDataConfig
    resize_gt_config: ResizeGtConfig


@dataclass
class ExperimentPacking:
    id: int
    version: int
    webdataset_beamformer: WebDatasetBeamformerPacking
    trainloop: TrainLoopPacking
    model: ModelPacking
    commit_hash: str = "unknown"
    commit_msg: str = ""
