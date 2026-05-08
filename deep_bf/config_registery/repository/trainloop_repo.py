from ..codecs import decode_json_dict
from ..entities import (
    CriterionConfig,
    HyperparametersConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainLoopConfig,
)
from .base import BaseRepository


class TrainLoopRepository(BaseRepository):
    def get_trainloop_config(self, id: int) -> TrainLoopConfig:
        row = self._fetch_one(
            (
                "SELECT id, criterion_id, optimizer_id, scheduler_id, hyperparameters_id "
                "FROM trainloop WHERE id = ?"
            ),
            (id,),
            f"trainloop id={id}",
        )

        return TrainLoopConfig(
            id=int(row["id"]),
            criterion_id=int(row["criterion_id"]),
            optimizer_id=int(row["optimizer_id"]),
            scheduler_id=int(row["scheduler_id"]),
            hyperparameters_id=int(row["hyperparameters_id"]),
        )

    def get_criterion_config(self, id: int) -> CriterionConfig:
        row = self._fetch_one(
            "SELECT id, type, params_json FROM criterion WHERE id = ?",
            (id,),
            f"criterion id={id}",
        )

        return CriterionConfig(
            id=int(row["id"]),
            type=str(row["type"]),
            params=decode_json_dict(str(row["params_json"])),
        )

    def get_optimizer_config(self, id: int) -> OptimizerConfig:
        row = self._fetch_one(
            "SELECT id, type, params_json FROM optimizer WHERE id = ?",
            (id,),
            f"optimizer id={id}",
        )

        return OptimizerConfig(
            id=int(row["id"]),
            type=str(row["type"]),
            params=decode_json_dict(str(row["params_json"])),
        )

    def get_scheduler_config(self, id: int) -> SchedulerConfig:
        row = self._fetch_one(
            "SELECT id, type, params_json FROM scheduler WHERE id = ?",
            (id,),
            f"scheduler id={id}",
        )

        return SchedulerConfig(
            id=int(row["id"]),
            type=str(row["type"]),
            params=decode_json_dict(str(row["params_json"])),
        )

    def get_hyperparameters_config(self, id: int) -> HyperparametersConfig:
        row = self._fetch_one(
            "SELECT id, seed, n_epoch, batch_size, learning_rate FROM hyperparameters WHERE id = ?",
            (id,),
            f"hyperparameters id={id}",
        )

        return HyperparametersConfig(
            id=int(row["id"]),
            seed=int(row["seed"]),
            n_epoch=int(row["n_epoch"]),
            batch_size=int(row["batch_size"]),
            learning_rate=float(row["learning_rate"]),
        )
