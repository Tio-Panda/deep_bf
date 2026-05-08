from ..entities import ExperimentConfig
from .base import BaseRepository


class ExperimentsRepository(BaseRepository):
    def get_experiment_config(self, id: int) -> ExperimentConfig:
        row = self._fetch_one(
            (
                "SELECT id, version, webdataset_beamformer_id, trainloop_id, model_id, commit_hash, commit_msg "
                "FROM experiments WHERE id = ?"
            ),
            (id,),
            f"experiment id={id}",
        )

        return ExperimentConfig(
            id=int(row["id"]),
            version=int(row["version"]),
            webdataset_beamformer_id=int(row["webdataset_beamformer_id"]),
            trainloop_id=int(row["trainloop_id"]),
            model_id=int(row["model_id"]),
            commit_hash=str(row["commit_hash"]),
            commit_msg=str(row["commit_msg"]),
        )
