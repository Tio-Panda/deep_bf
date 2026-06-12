from typing import Any

import numpy as np

from ...config_registery.entities import (
    DataPreprocessingConfig,
    DataSizeConfig,
    Experiment,
    ModelPack,
)
from ...beamformers.utils.b_mode import get_bmode
from ...constants.bf import PWDataType


class ModelReconstruction:
    def __init__(
        self,
        pw: Any,
        data: Any,
        mode: PWDataType,
        times: Any,
        data_size_config: DataSizeConfig,
        data_preprocessing_config: DataPreprocessingConfig,
        model_pack: ModelPack,
        experiment: Experiment,
    ) -> None:
        self.zlims = np.array(pw.zlims) * 1e3
        self.xlims = np.array(pw.xlims) * 1e3
        self.data = data.cpu().numpy()
        self.mode = mode
        self.times = times
        self.name = pw.name

        self.data_size_config = data_size_config
        self.data_preprocessing_config = data_preprocessing_config
        self.model_pack = model_pack
        self.experiment = experiment

    def get_bmode(self, vmin=-60, vmax=0, eps=1e-10):
        b_mode = get_bmode(self.data, self.mode, vmin=vmin, vmax=vmax, eps=eps)

        return b_mode

    def get_total_time(self):
        total, _ = self._time_summary()
        return total

    def _safe_attr(self, obj, attr, default=None):
        if obj is None:
            return default
        return getattr(obj, attr, default)

    def _time_summary(self):
        times = self.times
        total = 0.0
        batches = None

        if isinstance(times, list):
            batches = len(times)
            for t in times:
                arr = np.asarray(t, dtype=float)
                if arr.size > 0:
                    total += float(np.sum(arr))
            return total, batches

        if hasattr(times, "__dict__"):
            values = vars(times)
        elif isinstance(times, dict):
            values = times
        else:
            return None, None

        for key in ("meshgrid", "d_rx", "apod"):
            v = values.get(key, None)
            if isinstance(v, (int, float)):
                total += float(v)

        batch_lengths = []
        for key in ("batch_gsi", "batch_resampler", "batch_bf"):
            v = values.get(key, None)
            if isinstance(v, (list, tuple, np.ndarray)):
                arr = np.asarray(v, dtype=float)
                if arr.size > 0:
                    total += float(np.sum(arr))
                batch_lengths.append(len(v))

        if batch_lengths:
            batches = max(batch_lengths)

        return total, batches

    def __str__(self):
        shape = tuple(self.data.shape)
        dtype = str(self.data.dtype)

        z0 = float(self.zlims[0]) if len(self.zlims) > 0 else float("nan")
        z1 = float(self.zlims[-1]) if len(self.zlims) > 0 else float("nan")
        x0 = float(self.xlims[0]) if len(self.xlims) > 0 else float("nan")
        x1 = float(self.xlims[-1]) if len(self.xlims) > 0 else float("nan")

        nz = self._safe_attr(self.data_size_config, "nz", None)
        nx = self._safe_attr(self.data_size_config, "nx", None)
        ns = self._safe_attr(self.data_size_config, "ns", None)

        preprocessing_id = self._safe_attr(self.data_preprocessing_config, "id", None)
        preprocessing_type = self._safe_attr(self.data_preprocessing_config, "type", "None")

        model_pack_id = self._safe_attr(self.model_pack, "id", None)
        model_family = self._safe_attr(self.model_pack, "family", "None")
        model_id = self._safe_attr(self.model_pack, "model_id", None)

        experiment_id = self._safe_attr(self.experiment, "id", None)
        experiment_description = self._safe_attr(self.experiment, "description", "None")

        total_time, batches = self._time_summary()
        if total_time is None:
            time_line = "times: total=unknown, batches=unknown"
        elif batches is None:
            time_line = f"times: total={total_time:.4f}s, batches=unknown"
        else:
            time_line = f"times: total={total_time:.4f}s, batches={batches}"

        return (
            f"name: {self.name}\n"
            f"mode: {self.mode}\n"
            f"shape: {shape}\n"
            f"dtype: {dtype}\n"
            f"limits_mm: z=[{z0:.3f}, {z1:.3f}], x=[{x0:.3f}, {x1:.3f}]\n"
            f"data_size: nz={nz}, nx={nx}, ns={ns}\n"
            f"preprocessing: id={preprocessing_id}, type={preprocessing_type}\n"
            f"model_pack: id={model_pack_id}, family={model_family}, model_id={model_id}\n"
            f"experiment: id={experiment_id}, description={experiment_description}\n"
            f"{time_line}"
        )

    __repr__ = __str__
