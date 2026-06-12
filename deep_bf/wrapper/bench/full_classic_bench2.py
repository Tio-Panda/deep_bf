from __future__ import annotations

from typing import List

import torch.nn as nn

from ...config_registery import (
    ApodConfig,
    BeamformerConfig,
    CompoundingConfig,
    DataSizeConfig,
    PathCenter,
    ResamplerConfig,
)
from ...data_handler import DataLoader
from ..reconstruction import ReconstructionCatalog
from .classic_bench2 import ClassicBench2


class FullClassicBench2(nn.Module):
    def __init__(
        self,
        names: List[str],
        data_type,
        data_sizes: List[DataSizeConfig],
        bfs: List[BeamformerConfig],
        resamplers: List[ResamplerConfig],
        apods: List[ApodConfig],
        compoundings: List[CompoundingConfig],
        angles: List[int] = [-1],
        angle_batch_size=5,
        device="cuda",
        location="local",
    ):
        super().__init__()
        self.names = names
        self.data_type = data_type

        with PathCenter(location) as pc:
            self.dl = DataLoader(pc.dl)

        self.data_sizes = data_sizes
        self.bfs = bfs
        self.apods = apods
        self.resamplers = resamplers
        self.compoundings = compoundings
        self.angles = angles
        self.angle_batch_size = angle_batch_size
        self.device = device

    def forward(self):
        catalog = ReconstructionCatalog()

        for name in self.names:
            pw = self.dl.get_defined_pwdata(name, self.data_type)
            bench = ClassicBench2(
                pw,
                self.data_sizes,
                self.bfs,
                self.resamplers,
                self.apods,
                self.compoundings,
                angles=self.angles,
                angle_batch_size=self.angle_batch_size,
                device=self.device,
            )

            for reconstruction in bench().all():
                catalog.add(reconstruction)

        return catalog
