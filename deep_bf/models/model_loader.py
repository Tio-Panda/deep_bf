import torch
from ..config_registery import PathCenter, Experiment
from .model_builder import model_builder

def get_experiment_best_model(experiment: Experiment, location="local"):
    with PathCenter(location=location) as pc:
        mP = pc.get_model_paths(experiment, test_mode=True)
        best_path = mP.best

    model = model_builder(experiment.model_pack, batch_size=1).to("cuda")
    ckpt = torch.load(f"{best_path}/best.pt", map_location="cuda", weights_only=True)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    return model

def get_experiment_epoch_models(experiment: Experiment, location="local"):
    with PathCenter(location=location) as pc:
        mP = pc.get_model_paths(experiment, test_mode=True)
        epochs_path = mP.epochs

    epoch_models = {}
    for ckpt_path in sorted(epochs_path.glob("*.pt")):
        epoch = int(ckpt_path.stem.split("-")[-1])
        model = model_builder( experiment.model_pack, batch_size=1, location=location).to("cuda")

        ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=True)
        model.load_state_dict(ckpt["model_state"], strict=False)
        model.eval()

        epoch_models[epoch] = model

    return epoch_models
