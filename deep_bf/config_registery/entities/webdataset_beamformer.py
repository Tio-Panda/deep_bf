from dataclasses import dataclass
from typing import Any


@dataclass
class DataSizeConfig:
    id: int
    nz: int
    nx: int
    ns: int


@dataclass
class DataTypeConfig:
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
class ResizeGtConfig:
    id: int
    type: str
    params: dict[str, Any]


@dataclass
class TransformDataConfig:
    id: int
    type: str
    params: dict[str, Any]


@dataclass
class WebDatasetBeamformerConfig:
    id: int
    gt_source: str
    data_type_id: int
    data_size_id: int
    samples_organization_id: int
    transform_data_id: int
    resize_gt_id: int
