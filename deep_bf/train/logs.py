import csv
from pathlib import Path


def append_epoch_log(file_path: Path, header, data):
    write_header = (not file_path.exists()) or (file_path.stat().st_size == 0)
    
    with file_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)

def append_epoch_loss(path: Path, epoch, avg_train, avg_val, best_val_loss):
    loss_file = path / "loss.csv"
    header = ["epoch", "train_loss", "val_loss", "gap", "best_val_loss"] 
    data = [epoch, avg_train, avg_val, avg_val - avg_train, best_val_loss]

    append_epoch_log(loss_file, header, data)

def append_epoch_weights(path: Path, epoch, prev_lr, next_lr, avg_grad_norm_l2):
    weights_file = path / "weights.csv"
    header = ["epoch", "prev_lr", "next_lr", "avg_grad_norm_l2"]
    data = [epoch, prev_lr, next_lr, avg_grad_norm_l2]

    append_epoch_log(weights_file, header, data)
