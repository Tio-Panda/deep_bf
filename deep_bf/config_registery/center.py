from pathlib import Path
import re
import warnings

import pandas as pd

from .db.connection import create_connection, ensure_schema_initialized
from .db.seed_export import export_delta_seed
from .entities import (
    ActivationConfig,
    ApodConfig,
    ArchitectureCnnBfConfig,
    BeamformerConfig,
    BeamformerSetup,
    CompoundingConfig,
    Conv2dInitConfig,
    CriterionConfig,
    DataPreprocessingConfig,
    DataSizeConfig,
    DataTypeConfig,
    Experiment,
    HyperparametersConfig,
    ModelPack,
    OptimizerConfig,
    ResamplerConfig,
    SamplesOrganizationConfig,
    SchedulerConfig,
    TrainLoopSetup,
    WebDatasetBeamformerPack,
)
from .errors import ConfigNotFoundError
from .repository import (
    BeamformerRepository,
    ExperimentsRepository,
    ModelRepository,
    TrainLoopRepository,
    WebDatasetBeamformerRepository,
)


class ConfigRegisteryCenter:
    _IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(
        self,
        db_path: str | Path | None = None,
        auto_export_delta_on_add: bool = True,
        delta_seeds_out_dir: str | Path | None = None,
    ) -> None:
        self._db_path = Path(db_path) if db_path is not None else None
        self._auto_export_delta_on_add = auto_export_delta_on_add
        self._delta_seeds_out_dir = (
            Path(delta_seeds_out_dir) if delta_seeds_out_dir is not None else None
        )
        self._conn = create_connection(self._db_path)
        ensure_schema_initialized(self._conn, self._db_path)

        self._experiments_repo = ExperimentsRepository(self._conn)
        self._webdataset_repo = WebDatasetBeamformerRepository(self._conn)
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

    def _export_delta_seed(self, table_name: str, pk_value: int | dict[str, object]) -> None:
        if not self._auto_export_delta_on_add:
            return

        try:
            export_delta_seed(
                table_name=table_name,
                pk_value=pk_value,
                db_path=self._db_path,
                out_dir=self._delta_seeds_out_dir,
            )
        except Exception as exc:
            warnings.warn(
                f"Could not export delta seed for table '{table_name}' and pk={pk_value}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    def get_conv2d_init(self, id: int) -> Conv2dInitConfig:
        return self._model_repo.get_conv2d_init_config(id)

    def get_activation(self, id: int) -> ActivationConfig:
        return self._model_repo.get_activation_config(id)

    def get_architecture(self, family: str, model_id: int) -> list[ArchitectureCnnBfConfig]:
        return self._model_repo.get_architecture_configs(family=family, model_id=model_id)

    def get_criterion(self, id: int) -> CriterionConfig:
        return self._trainloop_repo.get_criterion_config(id)

    def get_optimizer(self, id: int) -> OptimizerConfig:
        return self._trainloop_repo.get_optimizer_config(id)

    def get_scheduler(self, id: int) -> SchedulerConfig:
        return self._trainloop_repo.get_scheduler_config(id)

    def get_hyperparameters(self, id: int) -> HyperparametersConfig:
        return self._trainloop_repo.get_hyperparameters_config(id)

    def get_data_size(self, id: int) -> DataSizeConfig:
        return self._webdataset_repo.get_data_size_config(id)

    def get_data_preprocessing(self, id: int) -> DataPreprocessingConfig:
        return self._webdataset_repo.get_data_preprocessing_config(id)

    def get_samples_organization(self, id: int) -> SamplesOrganizationConfig:
        return self._webdataset_repo.get_samples_organization_config(id)

    def get_data_type(self, id: int) -> DataTypeConfig:
        return self._beamformer_repo.get_data_type_config(id)

    def get_resampler(self, id: int) -> ResamplerConfig:
        return self._beamformer_repo.get_resampler_config(id)

    def get_beamformer(self, id: int) -> BeamformerConfig:
        return self._beamformer_repo.get_beamformer_config(id)

    def get_compounding(self, id: int) -> CompoundingConfig:
        return self._beamformer_repo.get_compounding_config(id)

    def get_apod(self, id: int) -> ApodConfig:
        return self._beamformer_repo.get_apod_config(id)

    def get_beamformer_setup(self, id: int) -> BeamformerSetup:
        return self._beamformer_repo.get_beamformer_setup(id)

    def get_model_pack(self, id: int) -> ModelPack:
        row = self._model_repo.get_model_pack_row(id)
        conv2d_init_config = self.get_conv2d_init(int(row["conv2d_init_config_id"]))
        activation_config = self.get_activation(int(row["activation_config_id"]))
        architecture_configs = self.get_architecture(
            family=str(row["family"]), model_id=int(row["model_id"])
        )
        beamformer_setup = self.get_beamformer_setup(int(row["beamformer_setup_id"]))
        return ModelPack(
            id=int(row["id"]),
            family=str(row["family"]),
            model_id=int(row["model_id"]),
            conv2d_init_config=conv2d_init_config,
            activation_config=activation_config,
            architecture_configs=architecture_configs,
            beamformer_setup=beamformer_setup,
        )

    def get_trainloop_setup(self, id: int) -> TrainLoopSetup:
        row = self._trainloop_repo.get_trainloop_setup_row(id)
        criterion_config = self.get_criterion(int(row["criterion_config_id"]))
        optimizer_config = self.get_optimizer(int(row["optimizer_config_id"]))
        scheduler_config = self.get_scheduler(int(row["scheduler_config_id"]))
        hyperparameters_config = self.get_hyperparameters(
            int(row["hyperparameters_config_id"])
        )
        return TrainLoopSetup(
            id=int(row["id"]),
            criterion_config=criterion_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            hyperparameters_config=hyperparameters_config,
        )

    def get_webdataset_beamformer_pack(self, id: int) -> WebDatasetBeamformerPack:
        row = self._webdataset_repo.get_webdataset_beamformer_pack_row(id)
        beamformer_setup = self.get_beamformer_setup(int(row["beamformer_setup_id"]))
        data_size_config = self.get_data_size(int(row["data_size_config_id"]))
        data_preprocessing_config = self.get_data_preprocessing(
            int(row["data_preprocessing_config_id"])
        )
        samples_organization_config = self.get_samples_organization(
            int(row["samples_organization_config_id"])
        )
        return WebDatasetBeamformerPack(
            id=int(row["id"]),
            beamformer_setup=beamformer_setup,
            data_size_config=data_size_config,
            data_preprocessing_config=data_preprocessing_config,
            samples_organization_config=samples_organization_config,
        )

    def get_experiment(self, id: int) -> Experiment:
        row = self._experiments_repo.get_experiment_row(id)
        return Experiment(
            id=int(row["id"]),
            description=str(row["description"]),
            model_pack=self.get_model_pack(int(row["model_pack_id"])),
            trainloop_setup=self.get_trainloop_setup(int(row["trainloop_setup_id"])),
            webdataset_beamformer_pack=self.get_webdataset_beamformer_pack(
                int(row["webdataset_beamformer_pack_id"])
            ),
        )

    def add_conv2d_init(self, config: Conv2dInitConfig) -> Conv2dInitConfig:
        created = self._model_repo.add_conv2d_init(config)
        self._export_delta_seed("conv2d_init_config", created.id)
        return created

    def add_activation(self, config: ActivationConfig) -> ActivationConfig:
        created = self._model_repo.add_activation(config)
        self._export_delta_seed("activation_config", created.id)
        return created

    def add_architecture(
        self, config: ArchitectureCnnBfConfig
    ) -> ArchitectureCnnBfConfig:
        created = self._model_repo.add_architecture(config)
        self._export_delta_seed(
            "architecture_cnn_bf_config",
            {
                "model_id": created.model_id,
                "family": created.family,
                "pos": created.pos,
            },
        )
        return created

    def add_criterion(self, config: CriterionConfig) -> CriterionConfig:
        created = self._trainloop_repo.add_criterion(config)
        self._export_delta_seed("criterion_config", created.id)
        return created

    def add_optimizer(self, config: OptimizerConfig) -> OptimizerConfig:
        created = self._trainloop_repo.add_optimizer(config)
        self._export_delta_seed("optimizer_config", created.id)
        return created

    def add_scheduler(self, config: SchedulerConfig) -> SchedulerConfig:
        created = self._trainloop_repo.add_scheduler(config)
        self._export_delta_seed("scheduler_config", created.id)
        return created

    def add_hyperparameters(
        self, config: HyperparametersConfig
    ) -> HyperparametersConfig:
        created = self._trainloop_repo.add_hyperparameters(config)
        self._export_delta_seed("hyperparameters_config", created.id)
        return created

    def add_data_size(self, config: DataSizeConfig) -> DataSizeConfig:
        created = self._webdataset_repo.add_data_size(config)
        self._export_delta_seed("data_size_config", created.id)
        return created

    def add_data_preprocessing(
        self, config: DataPreprocessingConfig
    ) -> DataPreprocessingConfig:
        created = self._webdataset_repo.add_data_preprocessing(config)
        self._export_delta_seed("data_preprocessing_config", created.id)
        return created

    def add_samples_organization(
        self, config: SamplesOrganizationConfig
    ) -> SamplesOrganizationConfig:
        created = self._webdataset_repo.add_samples_organization(config)
        self._export_delta_seed("samples_organization_config", created.id)
        return created

    def add_data_type(self, config: DataTypeConfig) -> DataTypeConfig:
        created = self._beamformer_repo.add_data_type(config)
        self._export_delta_seed("data_type_config", created.id)
        return created

    def add_resampler(self, config: ResamplerConfig) -> ResamplerConfig:
        created = self._beamformer_repo.add_resampler(config)
        self._export_delta_seed("resampler_config", created.id)
        return created

    def add_beamformer(self, config: BeamformerConfig) -> BeamformerConfig:
        created = self._beamformer_repo.add_beamformer(config)
        self._export_delta_seed("beamformer_config", created.id)
        return created

    def add_compounding(self, config: CompoundingConfig) -> CompoundingConfig:
        created = self._beamformer_repo.add_compounding(config)
        self._export_delta_seed("compounding_config", created.id)
        return created

    def add_apod(self, config: ApodConfig) -> ApodConfig:
        created = self._beamformer_repo.add_apod(config)
        self._export_delta_seed("apod_config", created.id)
        return created

    def add_beamformer_setup(
        self,
        data_type_config_id: int,
        beamformer_config_id: int,
        resampler_config_id: int,
        compounding_config_id: int,
        apod_config_id: int,
    ) -> BeamformerSetup:
        setup = self._beamformer_repo.add_beamformer_setup(
            data_type_config_id=data_type_config_id,
            beamformer_config_id=beamformer_config_id,
            resampler_config_id=resampler_config_id,
            compounding_config_id=compounding_config_id,
            apod_config_id=apod_config_id,
        )
        self._export_delta_seed("beamformer_setup", setup.id)
        return self.get_beamformer_setup(setup.id)

    def add_trainloop_setup(
        self,
        criterion_config_id: int,
        optimizer_config_id: int,
        scheduler_config_id: int,
        hyperparameters_config_id: int,
    ) -> TrainLoopSetup:
        new_id = self._trainloop_repo.add_trainloop_setup(
            criterion_config_id=criterion_config_id,
            optimizer_config_id=optimizer_config_id,
            scheduler_config_id=scheduler_config_id,
            hyperparameters_config_id=hyperparameters_config_id,
        )
        self._export_delta_seed("trainloop_setup", new_id)
        return self.get_trainloop_setup(new_id)

    def add_model_pack(
        self,
        family: str,
        model_id: int,
        conv2d_init_config_id: int,
        activation_config_id: int,
        beamformer_setup_id: int,
    ) -> ModelPack:
        new_id = self._model_repo.add_model_pack(
            family=family,
            model_id=model_id,
            conv2d_init_config_id=conv2d_init_config_id,
            activation_config_id=activation_config_id,
            beamformer_setup_id=beamformer_setup_id,
        )
        self._export_delta_seed("model_pack", new_id)
        return self.get_model_pack(new_id)

    def add_webdataset_beamformer_pack(
        self,
        beamformer_setup_id: int,
        data_size_config_id: int,
        data_preprocessing_config_id: int,
        samples_organization_config_id: int,
    ) -> WebDatasetBeamformerPack:
        new_id = self._webdataset_repo.add_webdataset_beamformer_pack(
            beamformer_setup_id=beamformer_setup_id,
            data_size_config_id=data_size_config_id,
            data_preprocessing_config_id=data_preprocessing_config_id,
            samples_organization_config_id=samples_organization_config_id,
        )
        self._export_delta_seed("webdataset_beamformer_pack", new_id)
        return self.get_webdataset_beamformer_pack(new_id)

    def add_experiment(
        self,
        model_pack_id: int,
        trainloop_setup_id: int,
        webdataset_beamformer_pack_id: int,
        description: str = "",
    ) -> Experiment:
        new_id = self._experiments_repo.add_experiment(
            description=description,
            model_pack_id=model_pack_id,
            trainloop_setup_id=trainloop_setup_id,
            webdataset_beamformer_pack_id=webdataset_beamformer_pack_id,
        )
        self._export_delta_seed("experiment", new_id)
        return self.get_experiment(new_id)
