from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

class ReconstructionMetrics:
    MAIN_METRICS_CONFIG = {
        "contrast_speckle_expe_dataset_rf": {
            "bbox1": [1780, 118, 130, 19],
            "bbox2": [1780, 90, 130, 19],
        },
        "contrast_speckle_simu_dataset_rf": {
            "bbox1": [1830, 114, 280, 28],
            "bbox2": [1830, 75, 280, 28],
        },
        "resolution_distorsion_expe_dataset_rf": {
            "bbox1": [1190, 44, 268, 27],
            "bbox2": [1190, 90, 268, 27],
        },
    }

    def __init__(self, clip_min: float = -60.0, clip_max: float = 0.0, bins: int = 256):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.bins = bins

    def cnr(self, reconstruction) -> float:
        roi1, roi2 = self._get_rois(reconstruction)

        denom = np.sqrt(np.var(roi1, ddof=0) + np.var(roi2, ddof=0))
        if denom < 1e-10:
            denom = 1e-10

        return float(np.abs(roi1.mean() - roi2.mean()) / denom)

    def gcnr(self, reconstruction) -> float:
        roi1, roi2 = self._get_rois(reconstruction)

        _, bins = np.histogram(np.concatenate((roi1, roi2)), bins=self.bins)
        f, _ = np.histogram(roi1, bins=bins, density=True)
        g, _ = np.histogram(roi2, bins=bins, density=True)

        f = f.astype(np.float64)
        g = g.astype(np.float64)

        f_sum = f.sum()
        g_sum = g.sum()
        if f_sum <= 0 or g_sum <= 0:
            raise ValueError("No se pudo normalizar el histograma para gCNR.")

        f /= f_sum
        g /= g_sum
        return float(1.0 - np.sum(np.minimum(f, g)))

    def ssim(self, reconstruction, ground_truth_reconstruction) -> float:
        self._validate_reconstruction(ground_truth_reconstruction, arg_name="ground_truth_reconstruction")

        pred_bmode = self._get_clipped_bmode(reconstruction)
        gt_bmode = self._get_clipped_bmode(ground_truth_reconstruction)

        if pred_bmode.shape != gt_bmode.shape:
            raise ValueError(
                "SSIM requiere que reconstruction y ground_truth_reconstruction tengan el mismo shape "
                f"(recibido {pred_bmode.shape} vs {gt_bmode.shape})"
            )

        data_range = float(self.clip_max - self.clip_min)
        if data_range <= 0:
            raise ValueError(
                f"Rango invalido para SSIM: clip_max ({self.clip_max}) debe ser mayor que clip_min ({self.clip_min})"
            )

        return float(structural_similarity(pred_bmode, gt_bmode, data_range=data_range))

    def contrast_metrics_markdown(self, reconstructions) -> str:
        if reconstructions is None:
            raise ValueError("reconstructions no puede ser None")

        rows = []
        for idx, reconstruction in enumerate(reconstructions):
            self._validate_reconstruction(reconstruction, arg_name=f"reconstructions[{idx}]")

            name = self._get_metrics_name(reconstruction, idx)
            rows.append(
                {
                    "name": name,
                    "cnr": self.cnr(reconstruction),
                    "gcnr": self.gcnr(reconstruction),
                }
            )

        if not rows:
            raise ValueError("reconstructions no puede estar vacio")

        df = pd.DataFrame(rows, columns=["name", "cnr", "gcnr"])
        return df.to_markdown(index=False, floatfmt=".6f")

    def ssim_metrics_markdown(self, reconstructions, ground_truth_reconstruction) -> str:
        if reconstructions is None:
            raise ValueError("reconstructions no puede ser None")

        self._validate_reconstruction(
            ground_truth_reconstruction,
            arg_name="ground_truth_reconstruction",
        )

        rows = []
        for idx, reconstruction in enumerate(reconstructions):
            self._validate_reconstruction(reconstruction, arg_name=f"reconstructions[{idx}]")

            name = self._get_metrics_name(reconstruction, idx)
            rows.append(
                {
                    "name": name,
                    "ssim": self.ssim(reconstruction, ground_truth_reconstruction),
                }
            )

        if not rows:
            raise ValueError("reconstructions no puede estar vacio")

        df = pd.DataFrame(rows, columns=["name", "ssim"])
        return df.to_markdown(index=False, floatfmt=".6f")

    def _get_rois(self, reconstruction):
        self._validate_reconstruction(reconstruction)
        name = reconstruction.name

        if name not in self.MAIN_METRICS_CONFIG:
            valid_names = ", ".join(sorted(self.MAIN_METRICS_CONFIG.keys()))
            raise ValueError(
                f"No hay bbox definidas para '{name}'. Nombres soportados: {valid_names}"
            )

        bmode = self._get_clipped_bmode(reconstruction)

        cfg = self.MAIN_METRICS_CONFIG[name]
        z1, x1, h1, w1 = cfg["bbox1"]
        z2, x2, h2, w2 = cfg["bbox2"]

        roi1 = bmode[z1 : z1 + h1, x1 : x1 + w1]
        roi2 = bmode[z2 : z2 + h2, x2 : x2 + w2]

        if roi1.size == 0 or roi2.size == 0:
            raise ValueError(
                f"ROI vacia para '{name}'. Verifica bbox y dimensiones de la reconstruccion."
            )

        return roi1, roi2

    def _get_clipped_bmode(self, reconstruction):
        self._validate_reconstruction(reconstruction)
        bmode = reconstruction.get_bmode()
        return np.clip(np.asarray(bmode), self.clip_min, self.clip_max)

    def _get_metrics_name(self, reconstruction, idx) -> str:
        experiment = getattr(reconstruction, "experiment", None)
        description = getattr(experiment, "description", None)
        if description is not None:
            description_text = str(description).strip()
            if description_text:
                return description_text

        beamformer_config = getattr(reconstruction, "beamformer_config", None)
        beamformer_type = getattr(beamformer_config, "type", None)
        if beamformer_type is not None:
            beamformer_text = str(beamformer_type).strip()
            if beamformer_text:
                return beamformer_text

        raise ValueError(
            "No se pudo construir la columna 'name' para "
            f"reconstructions[{idx}]: falta experiment.description o beamformer_config.type"
        )

    def _validate_reconstruction(self, reconstruction, arg_name="reconstruction"):
        if reconstruction is None:
            raise ValueError(f"{arg_name} no puede ser None")

        if not hasattr(reconstruction, "name"):
            raise ValueError(f"{arg_name} debe tener atributo 'name'")

        if not hasattr(reconstruction, "get_bmode") or not callable(reconstruction.get_bmode):
            raise ValueError(f"{arg_name} debe implementar el metodo 'get_bmode()'")


ContrastMetrics = ReconstructionMetrics
