import torch

from ..config import PathCenter, ExperimentPacking
from ...dataset.gsi import GlobalSamplesIdx, DL

def get_rfs_and_ids_by_names(names, mode, config:ExperimentPacking):
    pc = PathCenter(is_server=False)
    paths = pc.get_dataset_paths()

    query = config.webdataset.query_filter
    query += f" and name in {names}"

    gsi = GlobalSamplesIdx(paths.samples_idx, query, is_train=True)

    # Hacerle padding a las rfs para colocarlas en rfs = empty(B, nc, ns)
    # Obtener new_ns desde config
    
    B = len(names)
    new_ns = 2048
    rfs = torch.empty(B, 1, 128, new_ns)
    ids = torch.empty(B)

    for i, name in enumerate(names):
        pw = DL.get_defined_pwdata(name, mode)
        angle = pw.n_angles // 2

        rfs[i] = pw.data[angle]
        ids[i] = gsi[name]

    rfs = rfs.to(device="cuda", dtype=torch.float32)
    ids = ids.to(device="cuda", dtype=torch.float32)

    return rfs, ids
