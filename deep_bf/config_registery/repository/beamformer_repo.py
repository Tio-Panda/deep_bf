from ..codecs import decode_json_dict, encode_json
from ..entities import (
    ApodConfig,
    BeamformerConfig,
    BeamformerSetup,
    CompoundingConfig,
    DataTypeConfig,
    ResamplerConfig,
)
from .base import BaseRepository


class BeamformerRepository(BaseRepository):
    def get_data_type_config(self, id: int) -> DataTypeConfig:
        row = self._fetch_one(
            "SELECT id, type, params_json FROM data_type_config WHERE id = ?",
            (id,),
            f"data_type_config id={id}",
        )
        return DataTypeConfig(
            id=int(row["id"]),
            type=str(row["type"]),
            params=decode_json_dict(str(row["params_json"])),
        )

    def get_beamformer_config(self, id: int) -> BeamformerConfig:
        row = self._fetch_one(
            "SELECT id, type, params_json FROM beamformer_config WHERE id = ?",
            (id,),
            f"beamformer_config id={id}",
        )
        return BeamformerConfig(
            id=int(row["id"]),
            type=str(row["type"]),
            params=decode_json_dict(str(row["params_json"])),
        )

    def get_resampler_config(self, id: int) -> ResamplerConfig:
        row = self._fetch_one(
            "SELECT id, type, params_json FROM resampler_config WHERE id = ?",
            (id,),
            f"resampler_config id={id}",
        )
        return ResamplerConfig(
            id=int(row["id"]),
            type=str(row["type"]),
            params=decode_json_dict(str(row["params_json"])),
        )

    def get_compounding_config(self, id: int) -> CompoundingConfig:
        row = self._fetch_one(
            "SELECT id, type, params_json FROM compounding_config WHERE id = ?",
            (id,),
            f"compounding_config id={id}",
        )
        return CompoundingConfig(
            id=int(row["id"]),
            type=str(row["type"]),
            params=decode_json_dict(str(row["params_json"])),
        )

    def get_apod_config(self, id: int) -> ApodConfig:
        row = self._fetch_one(
            "SELECT id, type, params_json FROM apod_config WHERE id = ?",
            (id,),
            f"apod_config id={id}",
        )
        return ApodConfig(
            id=int(row["id"]),
            type=str(row["type"]),
            params=decode_json_dict(str(row["params_json"])),
        )

    def get_beamformer_setup(self, id: int) -> BeamformerSetup:
        row = self._fetch_one(
            (
                "SELECT id, data_type_config_id, beamformer_config_id, resampler_config_id, "
                "compounding_config_id, apod_config_id FROM beamformer_setup WHERE id = ?"
            ),
            (id,),
            f"beamformer_setup id={id}",
        )
        return BeamformerSetup(
            id=int(row["id"]),
            data_type_config=self.get_data_type_config(int(row["data_type_config_id"])),
            beamformer_config=self.get_beamformer_config(int(row["beamformer_config_id"])),
            resampler_config=self.get_resampler_config(int(row["resampler_config_id"])),
            compounding_config=self.get_compounding_config(int(row["compounding_config_id"])),
            apod_config=self.get_apod_config(int(row["apod_config_id"])),
        )

    def add_data_type(self, config: DataTypeConfig) -> DataTypeConfig:
        new_id = self._insert(
            "INSERT INTO data_type_config (type, params_json) VALUES (?, ?)",
            (config.type, encode_json(config.params)),
        )
        return self.get_data_type_config(new_id)

    def add_beamformer(self, config: BeamformerConfig) -> BeamformerConfig:
        new_id = self._insert(
            "INSERT INTO beamformer_config (type, params_json) VALUES (?, ?)",
            (config.type, encode_json(config.params)),
        )
        return self.get_beamformer_config(new_id)

    def add_resampler(self, config: ResamplerConfig) -> ResamplerConfig:
        new_id = self._insert(
            "INSERT INTO resampler_config (type, params_json) VALUES (?, ?)",
            (config.type, encode_json(config.params)),
        )
        return self.get_resampler_config(new_id)

    def add_compounding(self, config: CompoundingConfig) -> CompoundingConfig:
        new_id = self._insert(
            "INSERT INTO compounding_config (type, params_json) VALUES (?, ?)",
            (config.type, encode_json(config.params)),
        )
        return self.get_compounding_config(new_id)

    def add_apod(self, config: ApodConfig) -> ApodConfig:
        new_id = self._insert(
            "INSERT INTO apod_config (type, params_json) VALUES (?, ?)",
            (config.type, encode_json(config.params)),
        )
        return self.get_apod_config(new_id)

    def add_beamformer_setup(
        self,
        data_type_config_id: int,
        beamformer_config_id: int,
        resampler_config_id: int,
        compounding_config_id: int,
        apod_config_id: int,
    ) -> BeamformerSetup:
        new_id = self._insert(
            (
                "INSERT INTO beamformer_setup "
                "(data_type_config_id, beamformer_config_id, resampler_config_id, compounding_config_id, apod_config_id) "
                "VALUES (?, ?, ?, ?, ?)"
            ),
            (
                data_type_config_id,
                beamformer_config_id,
                resampler_config_id,
                compounding_config_id,
                apod_config_id,
            ),
        )
        return self.get_beamformer_setup(new_id)
