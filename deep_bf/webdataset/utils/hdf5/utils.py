import h5py
from ....config_registery import PathCenter, WebDatasetBeamformerPack

def inspect_ground_truth_hdf5(name: str, location: str = "local", show_datasets: bool = True):
    with PathCenter(location=location) as pc:
        path = f"{pc.dataset_paths.ground_truth}/{name}.hdf5"

    group_paths = []
    dataset_info = []

    with h5py.File(path, "r") as f:
        group_paths.append("/")
        def visitor(obj_path, obj):
            if isinstance(obj, h5py.Group):
                group_paths.append(f"/{obj_path}")
            elif show_datasets and isinstance(obj, h5py.Dataset):
                dataset_info.append(
                    {
                        "path": f"/{obj_path}",
                        "shape": tuple(obj.shape),
                        "dtype": str(obj.dtype),
                    }
                )
        f.visititems(visitor)

    print(f"\nHDF5: {path}")
    print("\nGrupos encontrados:")
    for g in group_paths:
        print(f"  {g}")

    if show_datasets:
        print("\nDatasets encontrados:")
        for d in dataset_info:
            print(f"  {d['path']} | shape={d['shape']} | dtype={d['dtype']}")

    return group_paths, dataset_info

def get_expected_ground_truth_group(pack: WebDatasetBeamformerPack):
    bfS = pack.beamformer_setup
    dC = pack.data_size_config

    data_type = bfS.data_type_config.type
    bf_type = bfS.beamformer_config.type
    r_type = bfS.resampler_config.type
    a_type = bfS.apod_config.type
    c_type = bfS.compounding_config.type
    
    group = f"/{data_type}/{bf_type}/{r_type}/{a_type}/{c_type}/{dC.nz}x{dC.nx}"
    print(group)

    return group
