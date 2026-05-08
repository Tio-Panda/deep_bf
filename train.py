import argparse
import torch

from deep_bf.train.train_loop import train_loop, set_seed
from deep_bf.config_registery import ConfigRegisteryCenter

if __name__ == "__main__":
    print("==================================================================")
    parser = argparse.ArgumentParser()
    parser.add_argument("-location", "--location", type=str, default="server")
    parser.add_argument("-e_id", "--experiment_id", type=int, default=0)
    args = parser.parse_args()

    LOCATION = args.location
    EXPERIMENT_ID = args.experiment_id
    print(f"Executing experiment id={EXPERIMENT_ID}")

    with ConfigRegisteryCenter() as cc:
        e = cc.get_experiment_packing(id=EXPERIMENT_ID)
        print(e)
    
    seed = e.trainloop.hyperparameters_config.seed
    set_seed(seed)

    e.trainloop.hyperparameters_config.batch_size = 1

    INTERVAL_EPOCH_SAVE = 5
    NUM_WORKERS = 1
    PIN_MEMORY = False

    DEVICE = "cuda"
    DTYPE = torch.float32

    train_loop(e, INTERVAL_EPOCH_SAVE, NUM_WORKERS, PIN_MEMORY, LOCATION, DEVICE, DTYPE)

    print("==================================================================")
