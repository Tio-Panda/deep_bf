import numpy as np
from ....data_handler import DataLoader
from ....config_registery import WebDatasetBeamformerPack, PathCenter
from ....constants.bf import PWDataType
from ..hdf5.load import get_ground_truth_hdf5

from .modules.normalize.normalize import normalize
from .modules.padding import padding_input

from ....models.bf.utils_layers.extractor import get_name2id_dic

def get_preprosessed_input(name, config: WebDatasetBeamformerPack, location="local"):
    soC = config.samples_organization_config
    with PathCenter(location=location) as pc:
        raw_path = str(pc.dataset_paths.raw)
        dl = DataLoader(raw_path)

    data_type = config.beamformer_setup.data_type_config.type
    pw = dl.get_defined_pwdata(name, data_type)

    order = soC.order
    C = order.find("C")
    W = order.find("W") - 1
    H = order.find("H") - 1
    
    # NOTE: O aqui iria data augmentation
    augmentation_mode = "center"
    if augmentation_mode == "center":
        aug_angle_idx = [pw.na//2]
        aug_data = [pw.data[aug_angle_idx]]
    
    name2id = get_name2id_dic(soC.query)
    aug_packs = []
    for data, angle_idx in zip(aug_data, aug_angle_idx):
        # TODO: manejar esto mejor
        data = data.squeeze(0)

        ppC = config.data_preprocessing_config
        data = normalize(data, ppC.type, ppC.params)

        new_ns = config.data_size_config.ns 
        data = padding_input(data, new_ns, data_type)
        
        if data_type == PWDataType.RF:
            data = data.transpose(W, H)
        else:
            data = data.transpose(W, H, 2)
        data = np.expand_dims(data, axis=C)

        pack = {
            "sii": name2id[name],
            "angle": angle_idx,
            "data": data
        }

        aug_packs.append(pack)

    return aug_packs

def get_preprosessed_gt(name, config: WebDatasetBeamformerPack, location="local"):
    bfS = config.beamformer_setup
    data_type = bfS.data_type_config.type
    bfC = bfS.beamformer_config
    rC = bfS.resampler_config
    aC = bfS.apod_config
    cC = bfS.compounding_config
    dC = config.data_size_config

    ppC = config.data_preprocessing_config

    gt = get_ground_truth_hdf5(name, data_type, bfC, rC, aC, cC, dC, location=location)

    gt = normalize(gt, ppC.type, ppC.params)

    return gt
