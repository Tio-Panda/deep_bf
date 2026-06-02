import argparse
import torch
from pathlib import Path

from deep_bf.train.train_loop import train_loop, set_seed
from deep_bf.config_registery import ConfigRegisteryCenter, PathCenter

if __name__ == "__main__":
    print("==================================================================")
    parser = argparse.ArgumentParser()
    parser.add_argument("-location", "--location", type=str, default="server")
    parser.add_argument("-e_id", "--experiment_id", type=int, default=0)
    parser.add_argument("-db_mode", "--dataset_mode", type=str, default="no-general")
    args = parser.parse_args()

    LOCATION = args.location
    EXPERIMENT_ID = args.experiment_id
    DATASET_MODE = args.dataset_mode
    print(f"Executing experiment id={EXPERIMENT_ID}")

    with ConfigRegisteryCenter() as cc:
        e = cc.get_experiment(id=EXPERIMENT_ID)
        print(e)
    
    seed = e.trainloop_setup.hyperparameters_config.seed
    set_seed(seed)
    
    if LOCATION != "server":
        e.trainloop_setup.hyperparameters_config.batch_size = 1

    # NOTE: Ahora mismo, todos los webdataset comparten el mismo samples_idx.
    # NOTE: Hay que asegurarse que los samples_organization sean siempre el mismo.

    # TODO: Reconocer e implementar lo necesario para impedir la ejecucion de un experimento si el dataset general no es COMPATIBLE
    # Probablemente se pueda hacer con el webdataset_beamformer_pack
    with PathCenter(location=LOCATION) as pc:
        webdataset_path = Path(pc.dataset_paths.webdataset_beamformer)
        p = pc.get_webdataset_beamformer_paths(e.webdataset_beamformer_pack,  DATASET_MODE, create=False)
        webdataset_path = p.base_path
        train_path = p.train_path
        val_path = p.val_path

    if DATASET_MODE != "general":
        if not webdataset_path.is_dir():
            raise FileNotFoundError(f"No existe carpeta de experimento: {webdataset_path}")
        if not train_path.is_dir():
            raise FileNotFoundError(f"No existe carpeta train: {train_path}")
        if not val_path.is_dir():
            raise FileNotFoundError(f"No existe carpeta val: {val_path}")

        n_train = len(list(train_path.glob("*.tar")))
        n_val = len(list(val_path.glob("*.tar")))

        if n_train == 0 or n_val == 0:
            raise FileNotFoundError(f"Dataset incompleto en {webdataset_path}: train_tars={n_train}, val_tars={n_val}")

    INTERVAL_EPOCH_SAVE = 5
    NUM_WORKERS = 1
    PIN_MEMORY = False

    DEVICE = "cuda"
    DTYPE = torch.float32
    
    train_loop(e, DATASET_MODE, INTERVAL_EPOCH_SAVE, NUM_WORKERS, PIN_MEMORY, LOCATION, DEVICE, DTYPE)

    print("==================================================================")
