# Imports: standard library
import os
import glob
import shutil
from typing import Dict, List, Tuple

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
        assert os.path.exists(model_file)
        name = os.path.basename(trial)
        out.append((name, config, model_file))
    return out


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
        for size in DOWNSTREAM_SIZES:
            for tmap in get_downstream_tmaps():
                original_config["model_file"] = None
                original_config["decay_steps"] = epochs * STEPS_PER_EPOCH
                original_config["downstream_tmap_name"] = tmap.name
                original_config["downstream_size"] = size
                original_config["num_augmentations"] = 0
                config = original_config.copy()
                config[
                    "description"
                ] = f"Not pretrained {tmap.name} model. Downstream labels {size}. {original_config['description']}."
                print(f"Set up config with description: {config['description']}")
                experiments.append(
                    Experiment(
                        name=f"{name}_{size}_not_pretrained",
                        run=RegNetTrainable,
                        config=config,
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
                original_config["model_file"] = model_file
                config = original_config.copy()
                config[
                    "description"
                ] = f"Pretrained {tmap.name} model. Downstream labels {size}. {original_config['description']}."
                print(f"Set up config with description: {config['description']}")
                experiments.append(
                    Experiment(
                        name=f"{name}_{size}_pretrained",
                        run=RegNetTrainable,
                        config=config,
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
                # config = original_config.copy()
                # config["description"] = f"Frozen pretrained {tmap.name} model. Downstream labels {size}. {original_config['description']}."
                # config["freeze_weights"] = True
                # print(f"Set up config with description: {config['description']}")
                # experiments.append(Experiment(
                #     name=f"{name}_{size}_frozen",
                #     run=RegNetTrainable,
                #     config=config,
                #     keep_checkpoints_num=1,
                #     checkpoint_score_attr="min-val_loss",
                #     checkpoint_freq=1,
                #     stop=stopper,
                #     resources_per_trial={
                #         "cpu": cpus_per_model,
                #         "gpu": gpus_per_model,
                #     },
                #     local_dir=output_folder,
                # ))
    print(f"Training {len(experiments)} models")
    run(
        experiments,
    )
