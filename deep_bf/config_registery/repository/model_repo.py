from ..codecs import decode_bool, decode_json_dict, decode_kernel
from ..entities import (
    ActivationConfig,
    ArchitectureCnnBfConfig,
    Conv2dInitConfig,
    ModelConfig,
)
from .base import BaseRepository


class ModelRepository(BaseRepository):
    def get_model_config(self, id: int) -> ModelConfig:
        row = self._fetch_one(
            (
                "SELECT id, family, model_id, conv2d_init_id, activation_id, beamformer_id "
                "FROM model WHERE id = ?"
            ),
            (id,),
            f"model id={id}",
        )

        return ModelConfig(
            id=int(row["id"]),
            family=str(row["family"]),
            model_id=int(row["model_id"]),
            conv2d_init_id=int(row["conv2d_init_id"]),
            activation_id=int(row["activation_id"]),
            beamformer_id=int(row["beamformer_id"]),
        )

    def get_conv2d_init_config(self, id: int) -> Conv2dInitConfig:
        row = self._fetch_one(
            "SELECT id, init_weights, init_bias FROM conv2d_init WHERE id = ?",
            (id,),
            f"conv2d_init id={id}",
        )

        return Conv2dInitConfig(
            id=int(row["id"]),
            init_weights=str(row["init_weights"]),
            init_bias=str(row["init_bias"]),
        )

    def get_activation_config(self, id: int) -> ActivationConfig:
        row = self._fetch_one(
            "SELECT id, type, params_json FROM activation WHERE id = ?",
            (id,),
            f"activation id={id}",
        )

        return ActivationConfig(
            id=int(row["id"]),
            type=str(row["type"]),
            params=decode_json_dict(str(row["params_json"])),
        )

    def get_architecture_configs(
        self, family: str, model_id: int
    ) -> list[ArchitectureCnnBfConfig]:
        rows = self.conn.execute(
            (
                "SELECT model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias "
                "FROM architecture_cnn_bf WHERE family = ? AND model_id = ? ORDER BY pos ASC"
            ),
            (family, model_id),
        ).fetchall()

        return [
            ArchitectureCnnBfConfig(
                model_id=int(row["model_id"]),
                family=str(row["family"]),
                pos=int(row["pos"]),
                type=str(row["type"]),
                ch_in=int(row["ch_in"]),
                ch_out=int(row["ch_out"]),
                kernel=decode_kernel(str(row["kernel_json"])),
                padding=str(row["padding"]),
                bias=decode_bool(int(row["bias"])),
            )
            for row in rows
        ]
