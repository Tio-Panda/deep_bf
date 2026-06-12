from ..codecs import decode_json_dict, encode_json
from ..entities import (
    DataPreprocessingConfig,
    DataSizeConfig,
    SamplesOrganizationConfig,
)
from .base import BaseRepository


class WebDatasetBeamformerRepository(BaseRepository):
    def get_data_size_config(self, id: int) -> DataSizeConfig:
        row = self._fetch_one(
            "SELECT id, nz, nx, ns FROM data_size_config WHERE id = ?",
            (id,),
            f"data_size_config id={id}",
        )
        return DataSizeConfig(
            id=int(row["id"]),
            nz=int(row["nz"]),
            nx=int(row["nx"]),
            ns=int(row["ns"]),
        )

    def get_data_preprocessing_config(self, id: int) -> DataPreprocessingConfig:
        row = self._fetch_one(
            "SELECT id, type, params_json FROM data_preprocessing_config WHERE id = ?",
            (id,),
            f"data_preprocessing_config id={id}",
        )
        return DataPreprocessingConfig(
            id=int(row["id"]),
            type=str(row["type"]),
            params=decode_json_dict(str(row["params_json"])),
        )

    def get_samples_organization_config(self, id: int) -> SamplesOrganizationConfig:
        row = self._fetch_one(
            (
                'SELECT id, seed, ratio, "order", select_mode, n_train, n_val, query, '
                "train_idxs, val_idxs FROM samples_organization_config WHERE id = ?"
            ),
            (id,),
            f"samples_organization_config id={id}",
        )
        return SamplesOrganizationConfig(
            id=int(row["id"]),
            seed=int(row["seed"]),
            ratio=float(row["ratio"]),
            order=str(row["order"]),
            select_mode=str(row["select_mode"]),
            n_train=int(row["n_train"]),
            n_val=int(row["n_val"]),
            query=str(row["query"]),
            train_idxs=str(row["train_idxs"]),
            val_idxs=str(row["val_idxs"]),
        )

    def get_webdataset_beamformer_pack_row(self, id: int):
        return self._fetch_one(
            (
                "SELECT id, beamformer_setup_id, data_size_config_id, data_preprocessing_config_id, "
                "samples_organization_config_id FROM webdataset_beamformer_pack WHERE id = ?"
            ),
            (id,),
            f"webdataset_beamformer_pack id={id}",
        )

    def add_data_size(self, config: DataSizeConfig) -> DataSizeConfig:
        new_id = self._insert(
            "INSERT INTO data_size_config (nz, nx, ns) VALUES (?, ?, ?)",
            (config.nz, config.nx, config.ns),
        )
        return self.get_data_size_config(new_id)

    def add_data_preprocessing(
        self, config: DataPreprocessingConfig
    ) -> DataPreprocessingConfig:
        new_id = self._insert(
            "INSERT INTO data_preprocessing_config (type, params_json) VALUES (?, ?)",
            (config.type, encode_json(config.params)),
        )
        return self.get_data_preprocessing_config(new_id)

    def add_samples_organization(
        self, config: SamplesOrganizationConfig
    ) -> SamplesOrganizationConfig:
        new_id = self._insert(
            (
                'INSERT INTO samples_organization_config (seed, ratio, "order", select_mode, '
                "n_train, n_val, query, train_idxs, val_idxs) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                config.seed,
                config.ratio,
                config.order,
                config.select_mode,
                config.n_train,
                config.n_val,
                config.query,
                config.train_idxs,
                config.val_idxs,
            ),
        )
        return self.get_samples_organization_config(new_id)

    def add_webdataset_beamformer_pack(
        self,
        beamformer_setup_id: int,
        data_size_config_id: int,
        data_preprocessing_config_id: int,
        samples_organization_config_id: int,
    ) -> int:
        return self._insert(
            (
                "INSERT INTO webdataset_beamformer_pack "
                "(beamformer_setup_id, data_size_config_id, data_preprocessing_config_id, samples_organization_config_id) "
                "VALUES (?, ?, ?, ?)"
            ),
            (
                beamformer_setup_id,
                data_size_config_id,
                data_preprocessing_config_id,
                samples_organization_config_id,
            ),
        )
