import webdataset as wds
import h5py
import hdf5plugin
import numpy as np
import shutil

from ..config_registery import WebDatasetBeamformerPacking
from .utils import padding_rf, resize_gt, sharifzadeh_transform


RESIZE_GT_BIGGEST_NZ = 2048
RESIZE_GT_ORIGINAL = "original"
RESIZE_GT_RESIZE = "resize"


def shard_writer(
    gsi,
    path,
    files,
    webdataset_beamformer_packing: WebDatasetBeamformerPacking,
):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

    wbC = webdataset_beamformer_packing.webdataset_beamformer_config
    soC = webdataset_beamformer_packing.samples_organization_config
    dsC = webdataset_beamformer_packing.data_size_config
    tdC = webdataset_beamformer_packing.transform_data_config
    gtC = webdataset_beamformer_packing.resize_gt_config

    order = soC.order
    C = order.find("C")
    W = order.find("W") - 1
    H = order.find("H") - 1

    nz = dsC.nz
    nx = dsC.nx
    ns = dsC.ns
    gt_source, gt_mode = wbC.gt_source.split("_")

    rf_group = tdC.type

    with wds.ShardWriter(f"{path}/dataset-%03d.tar", maxcount=100) as sink:
        for file in files:
            name = file.stem
            with h5py.File(file, "r") as f:
                g = f.require_group(rf_group)

                rf: np.ndarray = g["rf"][:]  # type: ignore
                rf = padding_rf(rf, ns)
                rf = rf.transpose(W, H)
                rf = np.expand_dims(rf, axis=C)

                if gtC.type == RESIZE_GT_ORIGINAL:
                    g2 = g.require_group(f"{nz}/{gt_source}")
                else:
                    g2 = g.require_group(f"{RESIZE_GT_BIGGEST_NZ}/{gt_source}")

                    gt: np.ndarray = g2[f"ground_truth_{gt_mode}"][:]  # type: ignore

                if gtC.type == RESIZE_GT_RESIZE:
                    # TODO: Agrear los params y ordenarlos mejor
                    gt = resize_gt(gt, nz, nx)

                gt = gt.transpose(W, H)
                samples_idx_id = str(gsi[name])

                sink.write(
                    {
                        "__key__": name,
                        "rf.npy": rf,
                        "gt.npy": gt,
                        "sii.txt": samples_idx_id,
                    }
                )
