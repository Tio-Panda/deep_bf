from ..codecs import decode_json_dict
from ..entities import ApodConfig, BeamformerConfig, CompoundingConfig, ResamplerConfig
from .base import BaseRepository


class BeamformerRepository(BaseRepository):
    def get_beamformer_config(self, id: int) -> BeamformerConfig:
        row = self._fetch_one(
            "SELECT id, type, resampler_id, params_json FROM beamformer WHERE id = ?",
            (id,),
            f"beamformer id={id}",
        )

        return BeamformerConfig(
            id=int(row["id"]),
            type=str(row["type"]),
            resampler_id=int(row["resampler_id"]),
            params=decode_json_dict(str(row["params_json"])),
        )

    def get_resampler_config(self, id: int) -> ResamplerConfig:
        row = self._fetch_one(
            "SELECT id, type, params_json FROM resampler WHERE id = ?",
            (id,),
            f"resampler id={id}",
        )

        return ResamplerConfig(
            id=int(row["id"]),
            type=str(row["type"]),
            params=decode_json_dict(str(row["params_json"])),
        )

    def get_apod_config(self, id: int) -> ApodConfig:
        row = self._fetch_one(
            "SELECT id, type, params_json FROM apod WHERE id = ?",
            (id,),
            f"apod id={id}",
        )

        return ApodConfig(
            id=int(row["id"]),
            type=str(row["type"]),
            params=decode_json_dict(str(row["params_json"])),
        )

    def get_compounding_config(self, id: int) -> CompoundingConfig:
        row = self._fetch_one(
            "SELECT id, type, params_json FROM compounding WHERE id = ?",
            (id,),
            f"compounding id={id}",
        )

        return CompoundingConfig(
            id=int(row["id"]),
            type=str(row["type"]),
            params=decode_json_dict(str(row["params_json"])),
        )
