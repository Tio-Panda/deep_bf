import numpy as np
from ...data_handler import DataLoader
from ...config_registery import WebDatasetBeamformerPacking, PathCenter

def padding_data(data, new_ns, mode):
    ns = data.shape[1]
    pad_width = ((0,0), (0, new_ns - ns)) if mode == "RF" else ((0,0), (0, new_ns - ns), (0, 0))
    if ns <= new_ns:
        data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
    else:
        data = data[:, :new_ns]
    
    return data

def sharifzadeh_transform(data, eps=1e-8):
    if data.ndim == 2:
        axes = (0, 1) # (nc, ns)
    elif data.ndim == 3:
        axes = (1, 2) # (na, nc, ns) -> por cada 'a'
    elif data.ndim == 4 and data.shape[-1] == 2:
        axes = (1, 2) # (na, nc, ns, 2) -> por cada (a, canal IQ)

    max_abs = np.max(np.abs(data), axis=axes, keepdims=True)
    data_norm = data / max_abs
    sigma = np.std(data_norm, axis=axes, keepdims=True) + eps
    return data_norm / sigma

TRANSFORM_NONE = "none"
TRANSFORM_SHARIFZADEH = "sharifzadeh"
def get_transormed_pw(name, config: WebDatasetBeamformerPacking, mode="full", location="local"):
    with PathCenter(location=location) as pc:
        raw_path = str(pc.dataset_paths.raw)
        dl = DataLoader(raw_path)

    data_type = config.data_type_config.type
    pw = dl.get_defined_pwdata(name, data_type)

    if mode == "center":
        idx = pw.na//2
        pw.data = pw.data[idx]

    tdC = config.transform_data_config
    transform_mode = tdC.type
    if transform_mode == TRANSFORM_SHARIFZADEH:
        pw.data = sharifzadeh_transform(pw.data, **tdC.params)

    return pw


def get_transformed_data_for_webdataset(name, gsi, config: WebDatasetBeamformerPacking, location="local"):
    soC = config.samples_organization_config
    with PathCenter(location=location) as pc:
        raw_path = str(pc.dataset_paths.raw)
        dl = DataLoader(raw_path)

    data_type = config.data_type_config.type
    pw = dl.get_defined_pwdata(name, data_type)

    order = soC.order
    C = order.find("C")
    W = order.find("W") - 1
    H = order.find("H") - 1
    
    mode = config.data_type_config.type
    data = get_transormed_pw(name, config, mode="center", location=location).data

    new_ns = config.data_size_config.ns 
    data = padding_data(data, new_ns, data_type)

    if mode == "RF":
        data = data.transpose(W, H)
    else:
        data = data.transpose(W, H, 2)
    data = np.expand_dims(data, axis=C)

    # TODO: Aqui iria data augmentation
    packs = [{
        "sii": gsi[name],
        "angle": pw.na//2,
        "data": data
    }]

    return packs
