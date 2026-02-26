import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

DATALOADER_PATH = "/home/panda/rf_data/"

LOCAL_BASE_URL = "/home/panda/rf_data/dataset/webdataset"
SERVER_BASE_URL = "/mnt/workspace/sgutierrezm/deep_bf/dataset/webdataset"

LOCAL_SAMPLES_IDX_PATH = "/home/panda/rf_data/dataset/samples_idx"
SEVER_SAMPLES_IDX_PATH = "/mnt/workspace/sgutierrezm/deep_bf/dataset/samples_idx"

RAW_PATH = "/home/panda/rf_data/dataset/raw"
IMG_PATH = "/home/panda/rf_data/dataset/img"
WEBDATASET_PATH = "/home/panda/rf_data/dataset/webdataset"
NC = 128
NS = 2300

NZ = 2048
NX = 256

DEVICE = "cuda"
TORCH_DTYPE = torch.float32

SEED = 42
BATCH_SIZE = 1
