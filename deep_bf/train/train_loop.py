import os
import random
import numpy as np
import torch
import json
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
from dataclasses import asdict, astuple

from deep_bf.webdataset.loader import get_datasets
from .save import make_backup, make_checkpoint, save_model
from .load import load_backup
from .logs import append_epoch_loss, append_epoch_weights
from .build import set_train_strategy

from ..config_registery import PathCenter, ExperimentPacking
from ..webdataset.gsi.gsi_for_training import GlobalSamplesIdxForTraining
from ..models.model_builder import model_builder

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_seed_worker(seed, id):
    seed = seed + id
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# TODO: Agregar mas mensajes de log, para saver que se guardo un mejor modelo, si bajo lr, etc.

def train_loop(
    config: ExperimentPacking, 
    interval_epoch_save, 
    num_workers, 
    pin_memory, 
    location="local",
    device="cuda",
    dtype=torch.float32
):
    # TODO: implementar logica para soportar un scheduler=None o distintos tipos de scheduler
    hC = config.trainloop.hyperparameters_config

    with PathCenter(location=location) as pc:
        mp = pc.get_model_paths(config)
        
        BACKUP_PATH = mp.backup
        BEST_CKPT_PATH = mp.best
        EPOCH_CKPT_PATH = mp.epochs
        LOGS_PATH = mp.logs
    
    gsi = GlobalSamplesIdxForTraining(config.webdataset_beamformer, cache_limit=15, location=location, reset=False)
    data_type = config.webdataset_beamformer.data_type_config.type
    BATCH_SIZE = hC.batch_size
    model = model_builder(data_type, config.model, gsi, BATCH_SIZE)
    model = model.to(device=device, dtype=dtype)

    print("Saving model metadata")
    metadata = asdict(config)
    with open(f"{LOGS_PATH}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    N_EPOCHS = hC.n_epoch
    train_loader, val_loader = get_datasets(hC, num_workers, pin_memory, location)
    criterion, optimizer, scheduler = set_train_strategy(model, config.trainloop)
    criterion_name = config.trainloop.criterion_config.type

    global_step = 0
    start_epoch = -1
    best_val_loss = float("inf")

    # === Load ===
    backup_file = BACKUP_PATH / "backup.pt"
    if backup_file.exists():
        start_epoch, global_step, best_val_loss = load_backup(
            backup_file, model, optimizer, scheduler, device
        )

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch + 1, N_EPOCHS):
        # === Train ===
        train_loss_weighted = 0.0
        train_numel = 0
        grad_norm_sum = 0.0
        grad_norm_steps = 0

        model.train()
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{N_EPOCHS} [train]", leave=False
        )
        for sample in train_pbar:
            rfs = sample[0].to(device, non_blocking=True)
            ids = sample[1].to(device, non_blocking=True)
            angles = sample[2].to(device, non_blocking=True)
            targets = sample[3].to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(rfs, ids, angles)

            loss = criterion(outputs, targets)
            loss.backward()

            grad_norm_l2 = float(
                clip_grad_norm_(model.parameters(), max_norm=float("inf"))
            )
            grad_norm_sum += grad_norm_l2
            grad_norm_steps += 1

            optimizer.step()

            batch_numel = targets.numel()
            train_loss_weighted += loss.item() * batch_numel
            train_numel += batch_numel

            global_step += 1

            train_pbar.set_postfix(
                loss=(train_loss_weighted / train_numel) if train_numel else float("nan")
            )

        avg_train = train_loss_weighted / train_numel
        avg_grad_norm_l2 = (
            grad_norm_sum / grad_norm_steps if grad_norm_steps else float("nan")
        )

        # === Validation ===
        val_loss_weighted = 0.0
        val_numel = 0

        model.eval()
        with torch.no_grad():
            val_pbar = tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{N_EPOCHS} [val]", leave=False
            )
            for sample in val_pbar:
                rfs = sample[0].to(device, non_blocking=True)
                ids = sample[1].to(device, non_blocking=True)
                angles = sample[2].to(device, non_blocking=True)
                targets = sample[3].to(device, non_blocking=True).unsqueeze(1)

                outputs = model(rfs, ids, angles)
                loss = criterion(outputs, targets)

                batch_numel = targets.numel()
                val_loss_weighted += loss.item() * batch_numel
                val_numel += batch_numel

                val_pbar.set_postfix(
                    loss=(val_loss_weighted / val_numel) if val_numel else float("nan")
                )

        avg_val = val_loss_weighted / val_numel
        tqdm.write(
            f"Epoch {epoch + 1}/{N_EPOCHS} -> Train {criterion_name}: {avg_train:.6f} | Val {criterion_name}: {avg_val:.6f}"
        )

        prev_lr = optimizer.param_groups[0]["lr"]

        if scheduler is not None:
            scheduler.step(avg_val)

        next_lr = optimizer.param_groups[0]["lr"]

        append_epoch_weights(LOGS_PATH, epoch, prev_lr, next_lr, avg_grad_norm_l2)

        # === Save best model ===
        if avg_val < best_val_loss:
            print("New best model")
            best_val_loss = avg_val
            best_ckpt = make_checkpoint(model)
            save_model(best_ckpt, BEST_CKPT_PATH, "best.pt")

        append_epoch_loss(LOGS_PATH, epoch, avg_train, avg_val, best_val_loss)

        # === Save model freq epoch ===
        if (epoch % interval_epoch_save == 0) or (epoch == N_EPOCHS - 1):
            interval_ckpt = make_checkpoint(model)
            save_model(interval_ckpt, EPOCH_CKPT_PATH, f"epoch-{epoch:03}.pt")

        # === Save backup ===
        backup = make_backup(
            model, optimizer, scheduler, epoch, global_step, best_val_loss
        )
        save_model(backup, BACKUP_PATH, "backup.pt")

    # Borrar backup
    if backup_file.exists():
        backup_file.unlink()
