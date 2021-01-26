# Imports: standard library
from typing import Dict

# Imports: third party
import numpy as np
import pandas as pd
from data import get_downstream_datasets, get_pretraining_datasets
from hyperoptimize import build_downstream_model, build_pretraining_model
from sklearn.metrics import r2_score, roc_auc_score
from train_downstream import _get_configs_and_models


def get_metrics(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    metrics = {}
    for name, yp in pred.items():
        if np.isnan(yp).any():
            print(f"{np.isnan(yp).sum()} NaNs found in predictions for {name}")
            continue
        if yp.shape[-1] == 2:
            metrics[name + "_rocauc"] = roc_auc_score(
                np.argmax(true[name], axis=-1),
                np.argmax(yp, axis=-1),
            )
        else:
            metrics[name + "_R2"] = r2_score(true[name], yp)
    return metrics


def infer_upstream_models(models_folder: str, csv_folder: str):
    results = []
    for name, config, model_file in _get_configs_and_models(models_folder):
        config["model_file"] = model_file
        (_, valid, test), stats, cleanups = get_pretraining_datasets(
            ecg_length=config["ecg_length"],
            augmentation_strengths={},
            num_augmentations=0,
            hd5_folder=config["hd5_folder"],
            num_workers=4,
            batch_size=1000,  # hacky way to get all the data at once
            csv_folder=csv_folder,
        )
        model = build_pretraining_model(**config)
        model.load_weights(model_file)
        x, y = next(valid.as_numpy_iterator())
        for cleanup in cleanups:
            cleanup()
        pred = model.predict(x)
        metrics = get_metrics(y, pred)
        description = config["description"]
        metrics["pretraining_loss"] = float(
            description.split(" ")[-1][:-1],
        )  # hacky way to get pretraining performance
        metrics["description"] = description
        print(description, metrics)
        results.append(metrics)
        for cleanup in cleanups:
            cleanup()
    return pd.DataFrame(results)


def infer_downstream_models(models_folder: str, csv_folder: str):
    configs = _get_configs_and_models(models_folder)
    results = []
    for name, config, model_file in configs:
        description = config["description"]
        if "Frozen" in description:
            continue
        print(f"Beginning inference on {description}")
        config["model_file"] = model_file
        (_, valid, test), stats, cleanups = get_downstream_datasets(
            downstream_tmap_name=config["downstream_tmap_name"],
            downstream_size=config["downstream_size"],
            ecg_length=config["ecg_length"],
            augmentation_strengths={},
            num_augmentations=0,
            hd5_folder=config["hd5_folder"],
            num_workers=4,
            batch_size=1000,  # hacky way to get all the data at once
            csv_folder=csv_folder,
        )
        model = build_downstream_model(**config)
        x, y = next(valid.as_numpy_iterator())
        for cleanup in cleanups:
            cleanup()
        pred = model.predict(x)
        metrics = get_metrics(y, pred)
        print(description, metrics)
        metrics["description"] = description
        metrics["task"] = config["downstream_tmap_name"]
        metrics["downstream_size"] = config["downstream_size"]
        metrics["pretraining_loss"] = float(
            description.split(" ")[-1][:-1],
        )  # hacky way to get pretraining performance
        metrics["pretraining_method"] = description.split(" ")[0]
        results.append(metrics)
    return pd.DataFrame(results)
