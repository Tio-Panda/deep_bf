import torch.nn as nn

from .beamformers.das import DAS
from .utils_layers.extractor import SamplesIdxExtractor
from .utils_layers.tofc import ToFCModels

from ...config_registery import BeamformerSetup
from ...constants.bf import BeamformerType
# from ...constants.model

class BFLayer(nn.Module):
    def __init__(self, extractor, tofc, bf):
        super().__init__()
        self.extractor = extractor
        self.tofc = tofc
        self.bf = bf

    def forward(self, batch_data, ids, angles):
        batch_samples_idx = self.extractor(ids, angles)
        batch_tofc_data = self.tofc(batch_data, batch_samples_idx)
        batch_bf_data = self.bf(batch_tofc_data)

        return batch_bf_data

def get_beamformer_layer(beamformer_setup: BeamformerSetup, batch_size=1, location="local", filter_chunk_size=None):
    extractor = SamplesIdxExtractor(location) 
    tofc = ToFCModels(batch_size=batch_size)

    bf_type = beamformer_setup.beamformer_config.type
    bf_type = BeamformerType.DAS

    if bf_type == BeamformerType.DAS:
        bf = DAS(batch_size=batch_size, filter_chunk_size=filter_chunk_size)
    else:
        bf = DAS(batch_size=batch_size, filter_chunk_size=filter_chunk_size)

    bf_layer = BFLayer(extractor, tofc, bf)

    return bf_layer
