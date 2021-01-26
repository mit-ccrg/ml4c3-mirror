# Imports: standard library
import os
from typing import Dict

# Imports: third party
import ray
import numpy as np
import pandas as pd
from data import get_downstream_datasets, get_pretraining_datasets
from hyperoptimize import build_downstream_model, build_pretraining_model
from sklearn.metrics import r2_score, roc_auc_score
from train_downstream import _get_configs_and_models


N_INFER = 10000000000  # hacky way to get all the data at once


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


@ray.remote(num_gpus=.5, max_calls=1)
def infer_model(config, upstream: bool, output_folder: str):
    if upstream:
        (_, valid, test), stats, cleanups = get_pretraining_datasets(
            ecg_length=config["ecg_length"],
            augmentation_strengths={},
            num_augmentations=0,
            hd5_folder=config["hd5_folder"],
            num_workers=4,
            batch_size=N_INFER,
            csv_folder=config["csv_folder"],
        )
        model = build_pretraining_model(**config)
        model.load_weights(config['model_file'])
    else:
        (_, valid, test), stats, cleanups = get_downstream_datasets(
            downstream_tmap_name=config["downstream_tmap_name"],
            downstream_size=config["downstream_size"],
            ecg_length=config["ecg_length"],
            augmentation_strengths={},
            num_augmentations=0,
            hd5_folder=config["hd5_folder"],
            num_workers=4,
            batch_size=N_INFER,  # hacky way to get all the data at once
            csv_folder=config['csv_folder'],
        )
        model = build_downstream_model(**config)
    description = config["description"]
    print(f'Running inference for {description}')
    x, y = next(valid.as_numpy_iterator())
    output_path = os.path.join(output_folder, f'{description}.csv')
    print(f'Saving {output_path}')
    pd.DataFrame({**model.predict(x), **y}).to_csv(output_path, index=False)


def infer_models(models_folder: str, output_folder: str, upstream: bool):
    configs = _get_configs_and_models(models_folder)
    for _, config, model_file in configs:
        config['model_file'] = model_file
    ray.get([
        infer_model.remote(config=config, output_folder=output_folder, upstream=upstream)
        for name, config, model_file in configs
    ])
