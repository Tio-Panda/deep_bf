import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import json
from tqdm import tqdm

from deep_bf.dataset import GlobalSamplesIdx, get_datasets
from deep_bf.models import DAS


def def_conv2d(in_ch, out_ch, kernel_size, padding):
    m = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=True)
    nn.init.xavier_uniform_(m.weight)
    nn.init.zeros_(m.bias)

    return m

class Toy(nn.Module):
    def __init__(self, gsi, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.conv1 = def_conv2d(1, 16, (5, 3), padding="same")
        self.conv2 = def_conv2d(16, 8, (5, 3), padding="same")
        self.conv3 = def_conv2d(8, 8, (5, 3), padding="same")
        self.conv4 = def_conv2d(8, 4, (5, 3), padding="same")
        self.conv5 = def_conv2d(4, 2, (5, 3), padding="same")
        self.conv6 = def_conv2d(2, 1, (7, 5), padding="same")
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.das = DAS(gsi, device=device, dtype=dtype)

    def forward(self, rfs, ids):
        x = rfs.to(device=self.device, dtype=self.dtype)
       
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.das(x, ids)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.activation(x)
        x = self.conv6(x)
        x = self.activation(x)
        return x

if __name__ == "__main__":
    base_url = "/home/panda/rf_data/dataset/webdataset"
    # base_url = "/mnt/workspace/sgutierrezm/deep_bf/dataset/webdataset"
    seed = 42

    n_epoch = 100
    batch_size = 2
    num_workers = 1
    pin_memory = True
    device = "cuda"

    # with open(f"{base_url}/metadata.json", "r", encoding="utf-8") as f:
    #     metadata = json.load(f)
    #
    # n_train = metadata["n_train"]
    # n_val = metadata["n_val"]
    #
    # print(n_train, n_val)

    gsi = GlobalSamplesIdx()
    train_loader, val_loader = get_datasets(
        base_url, seed, batch_size, num_workers, pin_memory
    )

    model = Toy(gsi).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-7, weight_decay=0.0, amsgrad=False)
    # criterion = nn.MSELoss(reduction="sum")
    criterion = nn.MSELoss(reduction="mean")

    best_val_loss = float("inf")

    for epoch in range(n_epoch):
        model.train()
        # train_sse = 0.0
        train_loss_weighted = 0.0
        train_numel = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epoch} [train]", leave=False)
        for sample in train_pbar:
            rfs, ids, targets = sample[0].to(device), sample[1].to(device), sample[2].to(device)

            optimizer.zero_grad()
            # outputs = model(rfs, ids).squeeze(1)
            outputs = model(rfs, ids)
            targets = targets.unsqueeze(1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # train_sse += loss.item()
            # train_numel += targets.numel()
            batch_numel = targets.numel()
            train_loss_weighted += loss.item() * batch_numel
            train_numel += batch_numel

            train_pbar.set_postfix(mse=(train_loss_weighted / train_numel) if train_numel else float("nan"))

        avg_train = train_loss_weighted / train_numel

        model.eval()
        # val_sse = 0.0
        val_loss_weighted = 0.0
        val_numel = 0

        with torch.no_grad():

            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epoch} [val]", leave=False)
            for sample in val_pbar:
                rfs, ids, targets = sample[0].to(device), sample[1].to(device), sample[2].to(device)

                # outputs = model(rfs, ids).squeeze(1)
                outputs = model(rfs, ids)
                targets = targets.unsqueeze(1)

                loss = criterion(outputs, targets)

                # val_sse += loss.item()
                # val_numel += targets.numel()
                batch_numel = targets.numel()
                val_loss_weighted += loss.item() * batch_numel
                val_numel += batch_numel

                val_pbar.set_postfix(mse=(val_loss_weighted / val_numel) if val_numel else float("nan"))

        avg_val = val_loss_weighted / val_numel
        tqdm.write(f"Epoch {epoch+1}/{n_epoch} -> Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_val_loss,
                },
                "./best_model.pth",
            )
            tqdm.write("¡Modelo guardado!")
