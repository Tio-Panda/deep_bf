import os
import sys
from collections import defaultdict
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_log_error

from modules import get_bmode, get_bmode_to_ax, load_data

DATA_PATH = "/home/panda/rf_data/dl_data/"

def generate_all_weight_paths(condor=True):
    if condor:
        BASE_PATH = Path("./models/condor")
    else:
        BASE_PATH = Path("./models/local")

    checkpoints = {}
    CHECKPOINT_PATH = BASE_PATH / "checkpoints"
    models = [name.name for name in CHECKPOINT_PATH.iterdir() if name.is_dir()]

    for aux in models:
        family, name = aux.split("-")

        if family not in checkpoints:
            checkpoints[family] = {}
        
        checkpoints[family][name] = {}

        model_versions = [
            version.name
            for version in (CHECKPOINT_PATH / f"{family}-{name}").iterdir()
            if version.is_dir()
        ]

        for version in model_versions:
            checkpoints[family][name][version] = {}
            _path = CHECKPOINT_PATH / f"{family}-{name}" / version

            best_path = _path / "best.keras"
            checkpoints[family][name][version]["best"] = best_path

            n_epochs = sum(1 for file in _path.iterdir() if file.is_file()) - 1
            epoch_range = range(5, (n_epochs * 5) + 1, 5)

            epoch_paths = []
            for epoch in epoch_range:
                epoch_paths.append(_path / f"epoch_{epoch:03d}.keras")

            checkpoints[family][name][version]["epochs"] = epoch_paths

    logs = {}
    LOGS_PATH = BASE_PATH / "logs"
    models = [name.name for name in LOGS_PATH.iterdir() if name.is_dir()]

    for aux in models:
        family, name = aux.split("-")

        if family not in logs:
            logs[family] = {}

        logs[family][name] = {}
        model_versions = [
            version.stem
            for version in (LOGS_PATH / f"{family}-{name}").iterdir()
            if version.is_file()
        ]
        for version in model_versions:
            _path = LOGS_PATH / f"{family}-{name}" / f"{version}.csv"
            logs[family][name][version] = _path

    output = {"checkpoints": checkpoints, "logs": logs}
    return output


ALL_PATHS = generate_all_weight_paths(condor=True)


def get_checkpoint_path(model_family, model_name, model_version):
    checkpoints = ALL_PATHS["checkpoints"][model_family][model_name][f"v-{model_version}"]
    return checkpoints["best"], checkpoints["epochs"]


def get_logs_path(model_family, model_name, model_version):
    return ALL_PATHS["logs"][model_family][model_name][f"v-{model_version}"]

def load_model_handler(path, old_mode=True):
    if old_mode == True:
        obj = { "LeakyReLU": keras.layers.LeakyReLU }
        model = keras.saving.load_model(path, custom_objects=obj)
    else:
        model = keras.saving.load_model(path)

    return model


class ModelHandler:
    def __init__(self, family, name, version):
        self.family = family
        self.name = name
        self.version = version

        self.best_path, self.epoch_paths = get_checkpoint_path(self.family, self.name, self.version)
        self.best = load_model_handler(self.best_path)

        self.epochs = []
        for epoch_path in self.epoch_paths:
            self.epochs.append(load_model_handler(epoch_path))

        self.logs = pd.read_csv(get_logs_path(self.family, self.name, self.version))

    def predict(self, inputs, use_best=True):
        if use_best:
            return self.best.predict(inputs)

        results = []
        for model in self.epochs:
            results.append(model.predict(inputs))
        return results 


class ModelLoader:
    def __init__(self, model_families, model_names, model_versions):
        self.models = {}
        for family in model_families:
            self.models[family] = {}
            for name in model_names:
                self.models[family][name] = {}
                for version in model_versions:
                    self.models[family][name][version] = ModelHandler(family, name, version)

    def __getitem__(self, key):
        return self.models[key]


class ModelPred:
    def __init__(self, model, input_name):
        self.model = model
        self.input_name = input_name

        self.inputs, self.ground_truth = load_data(input_name, DATA_PATH)

        self.grid = tf.squeeze(self.inputs["grid"])
        self.ground_truth_bmode = get_bmode(self.ground_truth)

        self.best_bmode = None
        self.epoch_bmodes = None
        self.cnr = None
        self.gcnr = None
        self.msle = None

    def calculate_best_bmode(self):
        # if self.best_bmode is not None: return
        self.best_pred = self.model.predict(self.inputs)
        self.best_bmode = get_bmode(self.best_pred[0, ..., 0])

    def calculate_epoch_bmodes(self):
        # if self.epoch_bmodes is not None: return
        bmodes = []
        for pred in self.model.predict(self.inputs, use_best=False):
            bmodes.append(get_bmode(pred[0, ..., 0]))
        self.epoch_bmodes = bmodes

    def calculate_contrast_metrics(self):
        if self.input_name not in MAIN_METRICS_CONFIG:
            return

        if self.cnr is not None and self.gcnr is not None:
            return

        if self.best_bmode is None:
            return

        inputs = MAIN_METRICS_CONFIG[self.input_name]
        z1, x1, h1, w1 = inputs["bbox1"]
        z2, x2, h2, w2 = inputs["bbox2"]

        bmode = np.clip(self.best_bmode, -45, 0)

        roi1 = bmode[z1 : z1 + h1, x1 : x1 + w1]
        roi2 = bmode[z2 : z2 + h2, x2 : x2 + w2]

        self.cnr = cnr(roi1, roi2)
        self.gcnr = gcnr(roi1, roi2)

    def calculate_msle(self):
        if self.best_bmode is None:
            return

        img1 = self.best_pred
        denom1 = np.max(img1) - np.min(img1)
        img1 = (img1 - np.min(img1)) / denom1 if denom1 != 0 else img1

        img2 = self.ground_truth
        denom2 = np.max(img2) - np.min(img2)
        img2 = (img2 - np.min(img2)) / denom2 if denom2 != 0 else img2

        self.msle = mean_squared_log_error(img2.ravel(), img1.ravel())


def cnr(img1, img2):
    d = np.sqrt(np.var(img1, ddof=0) + np.var(img2, ddof=0))
    if d < 1e-10:
        d = 1e-10

    return np.abs(img1.mean() - img2.mean()) / d


def gcnr(img1, img2):
    _, bins = np.histogram(np.concatenate((img1, img2)), bins=256)
    f, _ = np.histogram(img1, bins=bins, density=True)
    g, _ = np.histogram(img2, bins=bins, density=True)
    f /= f.sum()
    g /= g.sum()
    return 1 - np.sum(np.minimum(f, g))


MAIN_METRICS_CONFIG = {
    "contrast_speckle_expe_dataset_rf": {
        "bbox1": [1780, 118, 130, 19],
        "bbox2": [1780, 90, 130, 19],
    },
    "contrast_speckle_simu_dataset_rf": {
        "bbox1": [1830, 114, 280, 28],
        "bbox2": [1830, 75, 280, 28],
    },
    "resolution_distorsion_expe_dataset_rf": {
        "bbox1": [1190, 44, 268, 27],
        "bbox2": [1190, 90, 268, 27],
    },
}


import pandas as pd
from collections import defaultdict

import pandas as pd
from collections import defaultdict

class PredStorage:
    def __init__(
        self, models, input_names, best=True, epochs=True, metrics=True, msle=True
    ):
        self._data = {}
        self._indices = {
            "input_name": defaultdict(set),
            "model_family": defaultdict(set),
            "model_name": defaultdict(set),
            "model_version": defaultdict(set),
        }

        data = []
        # Accedemos a models.models que es el diccionario con la estructura jerárquica
        for family in models.models.keys():
            for name in models.models[family].keys():
                for version in models.models[family][name].keys():
                    # Aquí obtenemos el ModelHandler
                    model_handler = models.models[family][name][version]

                    for input_name in input_names:
                        pred = ModelPred(model_handler, input_name)
                        
                        # Ya no hace falta asignar la familia manualmente porque 
                        # ModelHandler ya la tiene como atributo (self.family)

                        print(f"Getting model-{family}-{name}-v{version}: {input_name}")

                        if best:
                            pred.calculate_best_bmode()
                        if epochs:
                            pred.calculate_epoch_bmodes()
                        if metrics:
                            pred.calculate_contrast_metrics()
                        if msle:
                            pred.calculate_msle()

                        self.add(pred)

                        data.append(
                            {
                                "input": input_name,
                                "family": family,
                                "model": name,
                                "version": version,
                                "full_id": f"{family}-{name}-v{version}",
                                "cnr": pred.cnr,
                                "gcnr": pred.gcnr,
                                "msle": pred.msle,
                            }
                        )

        self.df = pd.DataFrame(data)

    def add(self, item: ModelPred):
        # Usamos los atributos directamente del ModelHandler (item.model)
        primary_key = (
            item.input_name, 
            item.model.family, 
            item.model.name, 
            item.model.version
        )

        if primary_key in self._data:
            return

        self._data[primary_key] = item

        self._indices["input_name"][item.input_name].add(primary_key)
        self._indices["model_family"][item.model.family].add(primary_key)
        self._indices["model_name"][item.model.name].add(primary_key)
        self._indices["model_version"][item.model.version].add(primary_key)

    def find(self, input_name=None, model_family=None, model_name=None, model_version=None):
        filters = {}
        if input_name is not None:
            filters["input_name"] = input_name
        if model_family is not None:
            filters["model_family"] = model_family
        if model_name is not None:
            filters["model_name"] = model_name
        if model_version is not None:
            filters["model_version"] = model_version

        if not filters:
            return list(self._data.values())

        candidate_keys = None

        sorted_filters = sorted(
            filters.items(), key=lambda item: len(self._indices[item[0]][item[1]])
        )

        for attr, value in sorted_filters:
            found_keys = self._indices[attr].get(value, set())

            if candidate_keys is None:
                candidate_keys = found_keys
            else:
                candidate_keys = candidate_keys.intersection(found_keys)

            if not candidate_keys:
                return []

        return [self._data[k] for k in candidate_keys]


if __name__ == "__main__":
    model_names = ["1", "2"]
    model_versions = ["1", "2"]

    PICMUS_NAMES = [
        "contrast_speckle_expe_dataset_rf",
        "contrast_speckle_simu_dataset_rf",
        "resolution_distorsion_expe_dataset_rf",
        "resolution_distorsion_simu_dataset_rf",
        "carotid_long_expe_dataset_rf",
        "carotid_cross_expe_dataset_rf",
    ]

    model = ModelLoader(model_names, model_versions)

    pred = model["1"]["2"]
