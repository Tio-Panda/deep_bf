from .base import BaseRepository


class ExperimentsRepository(BaseRepository):
    def add_experiment(
        self,
        description: str,
        model_pack_id: int,
        trainloop_setup_id: int,
        webdataset_beamformer_pack_id: int,
    ) -> int:
        return self._insert(
            (
                "INSERT INTO experiment "
                "(description, model_pack_id, trainloop_setup_id, webdataset_beamformer_pack_id) "
                "VALUES (?, ?, ?, ?)"
            ),
            (
                description,
                model_pack_id,
                trainloop_setup_id,
                webdataset_beamformer_pack_id,
            ),
        )

    def get_experiment_row(self, id: int):
        return self._fetch_one(
            (
                "SELECT id, description, model_pack_id, trainloop_setup_id, webdataset_beamformer_pack_id "
                "FROM experiment WHERE id = ?"
            ),
            (id,),
            f"experiment id={id}",
        )
