import torch
import numpy as np

from deep_bf.config_registery import WebDatasetBeamformerPack

from .utils.preprosessing.preprosessing import get_preprosessed_input, get_preprosessed_gt

def get_quick_input(name, config:WebDatasetBeamformerPack, location="local", device="cuda", dtype=torch.float32):
    aug_inputs = get_preprosessed_input(name, config, location=location)

    data_list = [p["data"] for p in aug_inputs]
    sii_list = [p["sii"] for p in aug_inputs]
    angles_list = [p["angle"] for p in aug_inputs]

    batch_data = torch.from_numpy(np.stack(data_list)).to(device=device, dtype=dtype)
    batch_sii = torch.tensor(sii_list, device=device, dtype=torch.long)
    batch_angles = torch.tensor(angles_list, device=device, dtype=torch.long)

    return batch_data, batch_sii, batch_angles
    
def get_quick_gt(name, config:WebDatasetBeamformerPack, location="local", device="cuda", dtype=torch.float32):
    gt = get_preprosessed_gt(name, config, location=location)
    gt = torch.from_numpy(gt).to(dtype=dtype, device=device)

    return gt
