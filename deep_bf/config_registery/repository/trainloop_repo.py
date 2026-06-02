from ..codecs import decode_json_dict, encode_json
from ..entities import (
    CriterionConfig,
    HyperparametersConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from .base import BaseRepository


class TrainLoopRepository(BaseRepository):
    def get_criterion_config(self, id: int) -> CriterionConfig:
        row = self._fetch_one(
            "SELECT id, type, params_json FROM criterion_config WHERE id = ?",
            (id,),
            f"criterion_config id={id}",
        )
        return CriterionConfig(
            id=int(row["id"]),
            type=str(row["type"]),
            params=decode_json_dict(str(row["params_json"])),
        )

    def get_optimizer_config(self, id: int) -> OptimizerConfig:
        row = self._fetch_one(
            "SELECT id, type, params_json FROM optimizer_config WHERE id = ?",
            (id,),
            f"optimizer_config id={id}",
        )
        return OptimizerConfig(
            id=int(row["id"]),
            type=str(row["type"]),
            params=decode_json_dict(str(row["params_json"])),
        )

    def get_scheduler_config(self, id: int) -> SchedulerConfig:
        row = self._fetch_one(
            "SELECT id, type, params_json FROM scheduler_config WHERE id = ?",
            (id,),
            f"scheduler_config id={id}",
        )
        return SchedulerConfig(
            id=int(row["id"]),
            type=str(row["type"]),
            params=decode_json_dict(str(row["params_json"])),
        )

    def get_hyperparameters_config(self, id: int) -> HyperparametersConfig:
        row = self._fetch_one(
            "SELECT id, seed, n_epoch, batch_size, learning_rate FROM hyperparameters_config WHERE id = ?",
            (id,),
            f"hyperparameters_config id={id}",
        )
        return HyperparametersConfig(
            id=int(row["id"]),
            seed=int(row["seed"]),
            n_epoch=int(row["n_epoch"]),
            batch_size=int(row["batch_size"]),
            learning_rate=float(row["learning_rate"]),
        )

    def get_trainloop_setup_row(self, id: int):
        return self._fetch_one(
            (
                "SELECT id, criterion_config_id, optimizer_config_id, scheduler_config_id, hyperparameters_config_id "
                "FROM trainloop_setup WHERE id = ?"
            ),
            (id,),
            f"trainloop_setup id={id}",
        )

    def add_criterion(self, config: CriterionConfig) -> CriterionConfig:
        new_id = self._insert(
            "INSERT INTO criterion_config (type, params_json) VALUES (?, ?)",
            (config.type, encode_json(config.params)),
        )
        return self.get_criterion_config(new_id)

    def add_optimizer(self, config: OptimizerConfig) -> OptimizerConfig:
        new_id = self._insert(
            "INSERT INTO optimizer_config (type, params_json) VALUES (?, ?)",
            (config.type, encode_json(config.params)),
        )
        return self.get_optimizer_config(new_id)

    def add_scheduler(self, config: SchedulerConfig) -> SchedulerConfig:
        new_id = self._insert(
            "INSERT INTO scheduler_config (type, params_json) VALUES (?, ?)",
            (config.type, encode_json(config.params)),
        )
        return self.get_scheduler_config(new_id)

    def add_hyperparameters(self, config: HyperparametersConfig) -> HyperparametersConfig:
        new_id = self._insert(
            (
                "INSERT INTO hyperparameters_config (seed, n_epoch, batch_size, learning_rate) "
                "VALUES (?, ?, ?, ?)"
            ),
            (config.seed, config.n_epoch, config.batch_size, config.learning_rate),
        )
        return self.get_hyperparameters_config(new_id)

    def add_trainloop_setup(
        self,
        criterion_config_id: int,
        optimizer_config_id: int,
        scheduler_config_id: int,
        hyperparameters_config_id: int,
    ) -> int:
        return self._insert(
            (
                "INSERT INTO trainloop_setup "
                "(criterion_config_id, optimizer_config_id, scheduler_config_id, hyperparameters_config_id) "
                "VALUES (?, ?, ?, ?)"
            ),
            (
                criterion_config_id,
                optimizer_config_id,
                scheduler_config_id,
                hyperparameters_config_id,
            ),
        )
