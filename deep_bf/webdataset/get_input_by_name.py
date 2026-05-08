import torch
import webdataset as wds
import numpy as np
import shutil

from ..config_registery import WebDatasetBeamformerPacking
from .utils.transform_data import get_transformed_data_for_webdataset
from .utils.ground_truth import get_transformed_ground_truth
from ..webdataset.gsi.gsi_for_training import GlobalSamplesIdxForTraining

def get_input_by_name(name, config: WebDatasetBeamformerPacking, location="local", device="cuda", dtype=torch.float32):
    gsi = GlobalSamplesIdxForTraining(config, cache_limit=5, location=location, reset=False)
    pack = get_transformed_data_for_webdataset(name, gsi, config, location)

    data_list = [p["data"] for p in pack]
    sii_list = [p["sii"] for p in pack]
    angles_list = [p["angle"] for p in pack]
    
    batch_data = torch.from_numpy(np.stack(data_list)).to(device=device, dtype=dtype)
    batch_sii = torch.tensor(sii_list, device=device, dtype=torch.long)
    batch_angles = torch.tensor(angles_list, device=device, dtype=torch.long)

    return batch_data, batch_sii, batch_angles
