from ..codecs import decode_bool, decode_json_dict, decode_kernel, encode_json
from ..entities import (
    ActivationConfig,
    ArchitectureCnnBfConfig,
    Conv2dInitConfig,
)
from .base import BaseRepository


class ModelRepository(BaseRepository):
    def get_conv2d_init_config(self, id: int) -> Conv2dInitConfig:
        row = self._fetch_one(
            "SELECT id, init_weights, init_bias FROM conv2d_init_config WHERE id = ?",
            (id,),
            f"conv2d_init_config id={id}",
        )
        return Conv2dInitConfig(
            id=int(row["id"]),
            init_weights=str(row["init_weights"]),
            init_bias=str(row["init_bias"]),
        )

    def get_activation_config(self, id: int) -> ActivationConfig:
        row = self._fetch_one(
            "SELECT id, type, params_json FROM activation_config WHERE id = ?",
            (id,),
            f"activation_config id={id}",
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
                "FROM architecture_cnn_bf_config WHERE family = ? AND model_id = ? ORDER BY pos ASC"
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

    def get_model_pack_row(self, id: int):
        return self._fetch_one(
            (
                "SELECT id, family, model_id, conv2d_init_config_id, activation_config_id, beamformer_setup_id "
                "FROM model_pack WHERE id = ?"
            ),
            (id,),
            f"model_pack id={id}",
        )

    def add_conv2d_init(self, config: Conv2dInitConfig) -> Conv2dInitConfig:
        new_id = self._insert(
            "INSERT INTO conv2d_init_config (init_weights, init_bias) VALUES (?, ?)",
            (config.init_weights, config.init_bias),
        )
        return self.get_conv2d_init_config(new_id)

    def add_activation(self, config: ActivationConfig) -> ActivationConfig:
        new_id = self._insert(
            "INSERT INTO activation_config (type, params_json) VALUES (?, ?)",
            (config.type, encode_json(config.params)),
        )
        return self.get_activation_config(new_id)

    def add_model_pack(
        self,
        family: str,
        model_id: int,
        conv2d_init_config_id: int,
        activation_config_id: int,
        beamformer_setup_id: int,
    ) -> int:
        return self._insert(
            (
                "INSERT INTO model_pack "
                "(family, model_id, conv2d_init_config_id, activation_config_id, beamformer_setup_id) "
                "VALUES (?, ?, ?, ?, ?)"
            ),
            (
                family,
                model_id,
                conv2d_init_config_id,
                activation_config_id,
                beamformer_setup_id,
            ),
        )

    def add_architecture(self, config: ArchitectureCnnBfConfig) -> ArchitectureCnnBfConfig:
        self.conn.execute(
            (
                "INSERT INTO architecture_cnn_bf_config "
                "(model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                config.model_id,
                config.family,
                config.pos,
                config.type,
                config.ch_in,
                config.ch_out,
                encode_json(list(config.kernel)),
                config.padding,
                int(config.bias),
            ),
        )
        self.conn.commit()
        rows = self.get_architecture_configs(config.family, config.model_id)
        for row in rows:
            if row.pos == config.pos:
                return row
        raise RuntimeError("Inserted architecture config was not found")
