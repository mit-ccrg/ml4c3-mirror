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

N_INFER = 10000  # hacky way to get all the data at once


def get_metrics(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    metrics = {}
    for name, yp in pred.items():
        if not np.isfinite(yp).all():
            print(
                f"{(~np.isfinite(yp)).sum()} non-finite found in predictions for {name}",
            )
            continue
        try:
            if yp.shape[-1] == 2:
                metrics[name + "_rocauc"] = roc_auc_score(
                    np.argmax(true[name], axis=-1),
                    np.argmax(yp, axis=-1),
                )
            else:
                metrics[name + "_R2"] = r2_score(true[name], yp)
        except ValueError:
            print(f"Got value error calculating metric for {name}")
            continue
    return metrics


@ray.remote(num_gpus=0.5, max_calls=1)
def infer_model(config, upstream: bool, output_folder: str):
    # Imports: third party
    import tensorflow as tf  # necessary for ray tune

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(
            gpu,
            True,
        )  # do not allocate all memory right away

    os.makedirs(output_folder, exist_ok=True)
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
        model.load_weights(config["model_file"])
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
            csv_folder=config["csv_folder"],
        )
        model = build_downstream_model(**config)
    description = config["description"]
    print(f"Running inference for {description}")
    x, y = next(valid.as_numpy_iterator())
    pred = model.predict(x)
    print(description, get_metrics(y, pred))
    output_path = os.path.join(output_folder, f"{description}.csv")
    print(f"Saving {output_path}")
    df_dict = {}
    for k, v in pred.items():
        if v.shape[-1] >= 2:  # classifier case
            v = v.argmax(axis=-1)
        df_dict[f"{k}_pred"] = list(v.flatten())
    for k, v in y.items():
        if v.shape[-1] >= 2:  # classifier case
            v = v.argmax(axis=-1)
        df_dict[f"{k}_actual"] = list(v.flatten())
    pd.DataFrame(df_dict).to_csv(output_path, index=False)
    for cleanup in cleanups:
        cleanup()


def infer_models(models_folder: str, output_folder: str, upstream: bool):
    configs = _get_configs_and_models(models_folder)
    for _, config, model_file in configs:
        config["model_file"] = model_file
    ray.get(
        [
            infer_model.remote(
                config=config,
                output_folder=output_folder,
                upstream=upstream,
            )
            for name, config, model_file in configs
        ],
    )
