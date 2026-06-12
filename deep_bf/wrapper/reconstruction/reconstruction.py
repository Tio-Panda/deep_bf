from typing import Any

import numpy as np

from ...config_registery.entities import (
    ApodConfig,
    BeamformerConfig,
    CompoundingConfig,
    DataSizeConfig,
    ResamplerConfig,
)
from ...beamformers.utils.b_mode import get_bmode
from ...constants.bf import PWDataType


class Reconstruction:
    def __init__(
        self,
        pw: Any,
        data: Any,
        mode: PWDataType,
        times: Any,
        data_size_config: DataSizeConfig,
        beamformer_config: BeamformerConfig,
        resampler_config: ResamplerConfig,
        apod_config: ApodConfig,
        compounding_config: CompoundingConfig,
    ) -> None:
        self.zlims = np.array(pw.zlims) * 1e3
        self.xlims = np.array(pw.xlims) * 1e3
        self.data = data.cpu().numpy()
        self.mode = mode
        self.times = times
        self.name = pw.name

        self.data_size_config = data_size_config
        self.beamformer_config = beamformer_config
        self.resampler_config = resampler_config
        self.apod_config = apod_config
        self.compounding_config = compounding_config

    def get_bmode(self, vmin=-60, vmax=0, eps=1e-10):
        b_mode = get_bmode(self.data, self.mode, vmin=vmin, vmax=vmax, eps=eps)

        return b_mode

    def get_total_time(self):
        pass

    def _cfg_type(self, cfg):
        if cfg is None:
            return "None"
        return getattr(cfg, "type", "None")

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

        bf_type = self._cfg_type(self.beamformer_config)
        resampler_type = self._cfg_type(self.resampler_config)
        apod_type = self._cfg_type(self.apod_config)
        compounding_type = self._cfg_type(self.compounding_config)

        nz = self._safe_attr(self.data_size_config, "nz", None)
        nx = self._safe_attr(self.data_size_config, "nx", None)
        ns = self._safe_attr(self.data_size_config, "ns", None)

        total_time, batches = self._time_summary()
        if total_time is None:
            time_line = "  times: total=unknown, batches=unknown"
        elif batches is None:
            time_line = f"  times: total={total_time:.4f}s, batches=unknown"
        else:
            time_line = f"  times: total={total_time:.4f}s, batches={batches}"

        # return (
        #     f"Reconstruction(name={self.name}, mode={self.mode}, shape={shape}, dtype={dtype})\\n"
        #     f"  limits_mm: z=[{z0:.3f}, {z1:.3f}], x=[{x0:.3f}, {x1:.3f}]\\n"
        #     f"  pipeline: bf={bf_type} | resampler={resampler_type} | apod={apod_type} | compounding={compounding_type}\\n"
        #     f"  data_size: nz={nz}, nx={nx}, ns={ns}\\n"
        #     f"{time_line}"
        # )

        return (
            f"name: {self.name}\n"
            f"mode: {self.mode}\n"
            f"shape: {shape}\n"
            f"dtype: {dtype}\n"
            f"limits_mm: z=[{z0:.3f}, {z1:.3f}], x=[{x0:.3f}, {x1:.3f}]\n"
            f"pipeline: bf={bf_type} | resampler={resampler_type} | apod={apod_type} | compounding={compounding_type}\n"
            f"data_size: nz={nz}, nx={nx}, ns={ns}\n"
            f"{time_line}"
        )

    __repr__ = __str__
