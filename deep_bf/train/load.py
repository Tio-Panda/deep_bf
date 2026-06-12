import torch
from pathlib import Path

def load_backup(path: Path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location="cpu") 
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    rng_state = ckpt.get("rng_state", None)
    if rng_state is not None:
        rng_state = torch.as_tensor(rng_state, dtype=torch.uint8, device="cpu").contiguous()
        torch.set_rng_state(rng_state)

    cuda_rng_state = ckpt.get("cuda_rng_state", None)
    if torch.cuda.is_available() and cuda_rng_state is not None:
        fixed = [
            torch.as_tensor(s, dtype=torch.uint8, device="cpu").contiguous()
            for s in cuda_rng_state
        ]
        torch.cuda.set_rng_state_all(fixed)

    model.to(device)

    epoch = int(ckpt.get("epoch", -1))
    global_step = int(ckpt.get("global_step", 0))
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

    return epoch, global_step, best_val_loss


# def load_ckpt():
