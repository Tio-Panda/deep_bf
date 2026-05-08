from ..codecs import decode_json_dict
from ..entities import (
    DataSizeConfig,
    DataTypeConfig,
    ResizeGtConfig,
    SamplesOrganizationConfig,
    TransformDataConfig,
    WebDatasetBeamformerConfig,
)
from .base import BaseRepository


class WebDatasetBeamformerRepository(BaseRepository):
    def get_webdataset_beamformer_config(self, id: int) -> WebDatasetBeamformerConfig:
        row = self._fetch_one(
            (
                "SELECT id, gt_source, data_type_id, data_size_id, samples_organization_id, "
                "transform_data_id, resize_gt_id "
                "FROM webdataset_beamformer WHERE id = ?"
            ),
            (id,),
            f"webdataset_beamformer id={id}",
        )

        return WebDatasetBeamformerConfig(
            id=int(row["id"]),
            gt_source=str(row["gt_source"]),
            data_type_id=int(row["data_type_id"]),
            data_size_id=int(row["data_size_id"]),
            samples_organization_id=int(row["samples_organization_id"]),
            transform_data_id=int(row["transform_data_id"]),
            resize_gt_id=int(row["resize_gt_id"]),
        )

    def get_data_size_config(self, id: int) -> DataSizeConfig:
        row = self._fetch_one(
            "SELECT id, nz, nx, ns FROM data_size WHERE id = ?",
            (id,),
            f"data_size id={id}",
        )

        return DataSizeConfig(
            id=int(row["id"]),
            nz=int(row["nz"]),
            nx=int(row["nx"]),
            ns=int(row["ns"]),
        )

    def get_data_type_config(self, id: int) -> DataTypeConfig:
        row = self._fetch_one(
            "SELECT id, type, params_json FROM data_type WHERE id = ?",
            (id,),
            f"data_type id={id}",
        )

        return DataTypeConfig(
            id=int(row["id"]),
            type=str(row["type"]),
            params=decode_json_dict(str(row["params_json"])),
        )

    def get_samples_organization_config(self, id: int) -> SamplesOrganizationConfig:
        row = self._fetch_one(
            (
                'SELECT id, seed, ratio, "order", select_mode, n_train, n_val, query, '
                "train_idxs, val_idxs "
                "FROM samples_organization WHERE id = ?"
            ),
            (id,),
            f"samples_organization id={id}",
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

    def get_transform_data_config(self, id: int) -> TransformDataConfig:
        row = self._fetch_one(
            "SELECT id, type, params_json FROM transform_data WHERE id = ?",
            (id,),
            f"transform_data id={id}",
        )

        return TransformDataConfig(
            id=int(row["id"]),
            type=str(row["type"]),
            params=decode_json_dict(str(row["params_json"])),
        )

    def get_resize_gt_config(self, id: int) -> ResizeGtConfig:
        row = self._fetch_one(
            "SELECT id, type, params_json FROM resize_gt WHERE id = ?",
            (id,),
            f"resize_gt id={id}",
        )

        return ResizeGtConfig(
            id=int(row["id"]),
            type=str(row["type"]),
            params=decode_json_dict(str(row["params_json"])),
        )
