import webdataset as wds
import numpy as np
import shutil

from ...config_registery import WebDatasetBeamformerPacking
from .transform_data import get_transformed_data_for_webdataset
from .ground_truth import get_transformed_ground_truth

def beamformer_shard_writer(gsi, path, names, config:WebDatasetBeamformerPacking, location="local"):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

    with wds.ShardWriter(f"{path}/dataset-%03d.tar", maxcount=100) as sink:
        for name in names:
            gt = get_transformed_ground_truth(name, config, location)
            packs = get_transformed_data_for_webdataset(name, gsi, config, location)

            for pack in packs:
                sii, angle, data = pack.values()
                sink.write(
                    {
                        "__key__": name,
                        "data.npy": data,
                        "gt.npy": gt,
                        "angle.txt": str(angle),
                        "sii.txt": str(sii)
                    }
                )
