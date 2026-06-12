import os
import torch
from pathlib import Path

def make_backup(model, optimizer, scheduler, epoch, global_step, best_val_loss):
    backup = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "rng_state": torch.get_rng_state().to(dtype=torch.uint8, device="cpu"),
        "cuda_rng_state": (
            [s.to(dtype=torch.uint8, device="cpu") for s in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }
    return backup

def make_checkpoint(model):
    ckpt = { "model_state": model.state_dict() }
    return ckpt

def save_model(obj, path: Path, suffix):
    file_path = path / suffix
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, file_path)
