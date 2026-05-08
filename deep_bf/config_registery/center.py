from pathlib import Path
import re

import pandas as pd

from .db.connection import create_connection, ensure_schema_initialized
from .entities import (
    ActivationConfig,
    ApodConfig,
    ArchitectureCnnBfConfig,
    BeamformerConfig,
    CompoundingConfig,
    Conv2dInitConfig,
    CriterionConfig,
    DataSizeConfig,
    DataTypeConfig,
    ExperimentConfig,
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
from .errors import ConfigNotFoundError
from .packing import (
    BeamformerPacking,
    ExperimentPacking,
    ModelPacking,
    TrainLoopPacking,
    WebDatasetBeamformerPacking,
)
from .repository import (
    BeamformerRepository,
    ExperimentsRepository,
    ModelRepository,
    TrainLoopRepository,
    WebDatasetBeamformerRepository,
)


class ConfigRegisteryCenter:
    _IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else None
        self._conn = create_connection(self._db_path)
        ensure_schema_initialized(self._conn, self._db_path)

        self._experiments_repo = ExperimentsRepository(self._conn)
        self._webdataset_beamformer_repo = WebDatasetBeamformerRepository(self._conn)
        self._model_repo = ModelRepository(self._conn)
        self._beamformer_repo = BeamformerRepository(self._conn)
        self._trainloop_repo = TrainLoopRepository(self._conn)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ConfigRegisteryCenter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def list_tables(self) -> list[str]:
        rows = self._conn.execute(
            (
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name ASC"
            )
        ).fetchall()
        return [str(row["name"]) for row in rows]

    def show_table(
        self,
        table_name: str,
        limit: int | None = 50,
        order_by: str | None = "id",
    ) -> pd.DataFrame:
        self._validate_identifier(table_name, "table name")
        available_tables = self.list_tables()
        if table_name not in available_tables:
            tables = ", ".join(available_tables)
            raise ConfigNotFoundError(
                f"table '{table_name}' was not found. Available tables: {tables}"
            )

        query = f"SELECT * FROM {table_name}"

        if order_by is not None:
            self._validate_identifier(order_by, "order_by")
            table_columns = self._get_table_columns(table_name)
            if order_by not in table_columns:
                cols = ", ".join(table_columns)
                raise ConfigNotFoundError(
                    (
                        f"column '{order_by}' was not found in table '{table_name}'. "
                        f"Available columns: {cols}"
                    )
                )
            query += f" ORDER BY {order_by} ASC"

        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be a positive integer")
            query += f" LIMIT {int(limit)}"

        return pd.read_sql_query(query, self._conn)

    def _get_table_columns(self, table_name: str) -> list[str]:
        rows = self._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return [str(row["name"]) for row in rows]

    def _validate_identifier(self, value: str, context: str) -> None:
        if not self._IDENTIFIER_RE.fullmatch(value):
            raise ValueError(
                (
                    f"Invalid {context} '{value}'. "
                    "Only letters, numbers, and underscores are allowed."
                )
            )

    def get_experiment_config(self, id: int) -> ExperimentConfig:
        return self._experiments_repo.get_experiment_config(id)

    def get_experiment_packing(self, id: int) -> ExperimentPacking:
        config = self.get_experiment_config(id)
        webdataset_beamformer = self.get_webdataset_beamformer_packing(
            config.webdataset_beamformer_id
        )
        trainloop = self.get_trainloop_packing(config.trainloop_id)
        model = self.get_model_packing(config.model_id)

        return ExperimentPacking(
            id=config.id,
            version=config.version,
            commit_hash=config.commit_hash,
            commit_msg=config.commit_msg,
            webdataset_beamformer=webdataset_beamformer,
            trainloop=trainloop,
            model=model,
        )

    def get_webdataset_beamformer_config(self, id: int) -> WebDatasetBeamformerConfig:
        return self._webdataset_beamformer_repo.get_webdataset_beamformer_config(id)

    def get_webdataset_beamformer_packing(self, id: int) -> WebDatasetBeamformerPacking:
        webdataset_beamformer_config = self.get_webdataset_beamformer_config(id)
        data_type_config = self.get_data_type_config(
            webdataset_beamformer_config.data_type_id
        )
        data_size_config = self.get_data_size_config(
            webdataset_beamformer_config.data_size_id
        )
        samples_organization_config = self.get_samples_organization_config(
            webdataset_beamformer_config.samples_organization_id
        )
        transform_data_config = self.get_transform_data_config(
            webdataset_beamformer_config.transform_data_id
        )
        resize_gt_config = self.get_resize_gt_config(
            webdataset_beamformer_config.resize_gt_id
        )

        return WebDatasetBeamformerPacking(
            webdataset_beamformer_config=webdataset_beamformer_config,
            data_type_config=data_type_config,
            data_size_config=data_size_config,
            samples_organization_config=samples_organization_config,
            transform_data_config=transform_data_config,
            resize_gt_config=resize_gt_config,
        )

    def get_data_type_config(self, id: int) -> DataTypeConfig:
        return self._webdataset_beamformer_repo.get_data_type_config(id)

    def get_data_size_config(self, id: int) -> DataSizeConfig:
        return self._webdataset_beamformer_repo.get_data_size_config(id)

    def get_samples_organization_config(self, id: int) -> SamplesOrganizationConfig:
        return self._webdataset_beamformer_repo.get_samples_organization_config(id)

    def get_transform_data_config(self, id: int) -> TransformDataConfig:
        return self._webdataset_beamformer_repo.get_transform_data_config(id)

    def get_resize_gt_config(self, id: int) -> ResizeGtConfig:
        return self._webdataset_beamformer_repo.get_resize_gt_config(id)

    def get_model_config(self, id: int) -> ModelConfig:
        return self._model_repo.get_model_config(id)

    def get_model_packing(self, id: int) -> ModelPacking:
        model_config = self.get_model_config(id)
        conv2d_init_config = self.get_conv2d_init_config(model_config.conv2d_init_id)
        activation_config = self.get_activation_config(model_config.activation_id)
        architecture_configs = self.get_architecture_config(
            model_config.family,
            model_config.model_id,
        )
        beamformer = self.get_beamformer_packing(model_config.beamformer_id)

        return ModelPacking(
            model_config=model_config,
            conv2d_init_config=conv2d_init_config,
            activation_config=activation_config,
            architecture_configs=architecture_configs,
            beamformer=beamformer,
        )

    def get_conv2d_init_config(self, id: int) -> Conv2dInitConfig:
        return self._model_repo.get_conv2d_init_config(id)

    def get_activation_config(self, id: int) -> ActivationConfig:
        return self._model_repo.get_activation_config(id)

    def get_architecture_config(
        self, family: str, model_id: int
    ) -> list[ArchitectureCnnBfConfig]:
        configs = self._model_repo.get_architecture_configs(
            family=family, model_id=model_id
        )
        if not configs:
            raise ConfigNotFoundError(
                f"architecture configs were not found for family={family} and model_id={model_id}"
            )

        return configs

    def get_beamformer_packing(self, id: int) -> BeamformerPacking:
        beamformer_config = self.get_beamformer_config(id)
        resampler_config = self.get_resampler_config(beamformer_config.resampler_id)

        return BeamformerPacking(
            beamformer_config=beamformer_config,
            resampler_config=resampler_config,
        )

    def get_beamformer_config(self, id: int) -> BeamformerConfig:
        return self._beamformer_repo.get_beamformer_config(id)

    def get_resampler_config(self, id: int) -> ResamplerConfig:
        return self._beamformer_repo.get_resampler_config(id)

    def get_apod_config(self, id: int) -> ApodConfig:
        return self._beamformer_repo.get_apod_config(id)

    def get_compounding_config(self, id: int) -> CompoundingConfig:
        return self._beamformer_repo.get_compounding_config(id)

    def get_trainloop_config(self, id: int) -> TrainLoopConfig:
        return self._trainloop_repo.get_trainloop_config(id)

    def get_trainloop_packing(self, id: int) -> TrainLoopPacking:
        trainloop_config = self.get_trainloop_config(id)
        criterion_config = self.get_criterion_config(trainloop_config.criterion_id)
        optimizer_config = self.get_optimizer_config(trainloop_config.optimizer_id)
        scheduler_config = self.get_scheduler_config(trainloop_config.scheduler_id)
        hyperparameters_config = self.get_hyperparameters_config(
            trainloop_config.hyperparameters_id
        )

        return TrainLoopPacking(
            trainloop_config=trainloop_config,
            criterion_config=criterion_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            hyperparameters_config=hyperparameters_config,
        )

    def get_criterion_config(self, id: int) -> CriterionConfig:
        return self._trainloop_repo.get_criterion_config(id)

    def get_optimizer_config(self, id: int) -> OptimizerConfig:
        return self._trainloop_repo.get_optimizer_config(id)

    def get_scheduler_config(self, id: int) -> SchedulerConfig:
        return self._trainloop_repo.get_scheduler_config(id)

    def get_hyperparameters_config(self, id: int) -> HyperparametersConfig:
        return self._trainloop_repo.get_hyperparameters_config(id)
