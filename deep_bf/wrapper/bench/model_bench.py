from __future__ import annotations
import torch
import torch.nn as nn

from typing import List

from ...config_registery import (
    Experiment,
    DataPreprocessingConfig,
)

from ...config_registery import PathCenter
from ...data_handler import DataLoader

from ...constants.bf import PWDataType
from ..reconstruction.model_reconstruction import ModelReconstruction
from ..reconstruction.model_reconstruction_catalog import ModelReconstructionCatalog

from ...models.model_loader import get_experiment_best_model

from ...webdataset.quick_data_by_name import get_quick_input


# NOTE: Este necesita que exista un samples_idx ya creado
class ModelBench(nn.Module):
    def __init__(
        self,
        names,
        experiments: List[Experiment],
        data_preprocessing_configs: List[DataPreprocessingConfig],
    ):
        super().__init__()
        with PathCenter(location="local") as pc:
            dl_path = pc.dl
            self.DL = DataLoader(dl_path)

        self.names = names
        self.experiments = experiments
        self.dpCs = data_preprocessing_configs
        # self.wdbP = experiments[0].webdataset_beamformer_pack
        # self.wdbP.samples_organization_config.query = '(RF == 1) and (nc == 128) and (name.str.slice(0, 3) != "JHU")'

    
    def forward(self):
        catalog = ModelReconstructionCatalog()
        
        for name in self.names:
            pw = self.DL.get_defined_pwdata(name, PWDataType.RF)

            for exp in self.experiments:
                wdbP = exp.webdataset_beamformer_pack
                wdbP.samples_organization_config.query = '(RF == 1) and (nc == 128) and (name.str.slice(0, 3) != "JHU")'
                for dpC in self.dpCs:

                    wdbP.data_preprocessing_config = dpC 
                    data, sii, angle = get_quick_input(name=pw.name, config=wdbP, location="local")

                    model = get_experiment_best_model(exp, location="local")
                    with torch.no_grad():
                        output = model(data, sii, angle)

                    reconstruction = ModelReconstruction(
                        pw,
                        output.squeeze(),
                        pw.type,
                        None,
                        data_size_config=wdbP.data_size_config,
                        data_preprocessing_config=dpC,
                        model_pack=exp.model_pack,
                        experiment=exp,
                    )
                    catalog.add(reconstruction)

        return catalog
