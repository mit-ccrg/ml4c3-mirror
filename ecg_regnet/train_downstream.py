# Imports: standard library
import os
import glob
import shutil
from typing import Dict, List, Tuple
from itertools import product

# Imports: third party
import pandas as pd
from data import DOWNSTREAM_SIZES, get_downstream_tmaps
from ray.tune import Analysis, run
from hyperoptimize import STEPS_PER_EPOCH, EarlyStopping, RegNetTrainable
from ray.tune.experiment import Experiment


def _get_configs_and_models(folder: str) -> List[Tuple[str, Dict, str]]:
    """returns name, config, model file path"""
    analysis = Analysis(folder)
    configs = analysis.get_all_configs()
    out = []
    for trial, config in configs.items():
        try:
            df = pd.read_csv(os.path.join(trial, "progress.csv"))
        except pd.errors.EmptyDataError:
            print(f"Skipping config with description {config['description']}")
            continue
        best_checkpoint_idx = df["val_loss"].argmin() + 1
        model_file = os.path.join(
            os.getcwd(),
            trial,
            f"checkpoint_{best_checkpoint_idx}",
            "pretraining_model.h5",
        )
        if not os.path.exists(model_file):
            print(
                f'Skipping {config["description"]} because it has no checkpoint saved',
            )
            continue
        name = os.path.basename(trial)
        out.append((name, config, model_file))
    return out


def _scratch_model_experiment_from_config(tmap, size, config):
    config = config.copy()
    config["model_file"] = None
    config["downstream_tmap_name"] = tmap.name
    config["downstream_size"] = size
    config["num_augmentations"] = 0
    config[
        "description"
    ] = f"Not pretrained {tmap.name} model. Downstream labels {size}. {config['description']}."
    print(f"Set up config with description: {config['description']}")
    return config


def _pretrained_model_experiment_from_config(tmap, size, config, model_file):
    config = config.copy()
    config["model_file"] = model_file
    config["downstream_tmap_name"] = tmap.name
    config["downstream_size"] = size
    config["num_augmentations"] = 0
    config[
        "description"
    ] = f"Pretrained {tmap.name} model. Downstream labels {size}. {config['description']}."
    print(f"Set up config with description: {config['description']}")
    return config


def _frozen_model_experiment_from_config(tmap, size, config, model_file):
    config = config.copy()
    config["model_file"] = model_file
    config["downstream_tmap_name"] = tmap.name
    config["downstream_size"] = size
    config["num_augmentations"] = 0
    config["freeze_weights"] = True
    config[
        "description"
    ] = f"Frozen {tmap.name} model. Downstream labels {size}. {config['description']}."
    print(f"Set up config with description: {config['description']}")
    return config


def _run_experiments(
    epochs: int,
    pretrained_folder: str,
    output_folder: str,
    patience: int,
    cpus_per_model: int,
    gpus_per_model: float,
):
    configs_models = _get_configs_and_models(pretrained_folder)
    stopper = EarlyStopping(patience=patience, max_epochs=epochs)
    experiments = []
    for name, original_config, model_file in configs_models:
        for size, tmap in product(DOWNSTREAM_SIZES, get_downstream_tmaps()):
            for modified_config in [
                _scratch_model_experiment_from_config(tmap, size, original_config),
                _pretrained_model_experiment_from_config(
                    tmap,
                    size,
                    original_config,
                    model_file,
                ),
                _frozen_model_experiment_from_config(
                    tmap,
                    size,
                    original_config,
                    model_file,
                ),
            ]:
                experiments.append(
                    Experiment(
                        name=modified_config["description"],
                        run=RegNetTrainable,
                        config=modified_config,
                        keep_checkpoints_num=1,
                        checkpoint_score_attr="min-val_loss",
                        checkpoint_freq=1,
                        stop=stopper,
                        resources_per_trial={
                            "cpu": cpus_per_model,
                            "gpu": gpus_per_model,
                        },
                        local_dir=output_folder,
                    ),
                )
    print(f"Training {len(experiments)} models")
    run(
        experiments,
    )
