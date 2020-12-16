# Imports: standard library
import os
import glob
import shutil
from typing import Dict, List, Tuple

# Imports: third party
from ray.tune import Analysis, run
from ecg_regnet.hyperoptimize import EarlyStopping, RegNetTrainable
from ray.tune.experiment import Experiment
from data import (
    DOWNSTREAM_SIZES,
    get_downstream_tmaps,
)


def _get_configs_and_models(folder: str) -> List[Tuple[str, Dict, str]]:
    """returns name, config, model file path"""
    analysis = Analysis(folder)
    configs = analysis.get_all_configs()
    out = []
    for trial, config in configs:
        model_files = glob.glob(f"{trial}/*/*.h5")
        assert len(model_files) == 1
        name = os.path.basename(trial)
        out.append((name, config, model_files[0]))
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
    for name, config, model_file in configs_models:
        for size in DOWNSTREAM_SIZES:
            for tmap in get_downstream_tmaps():
                config["downstream_tmap_name"] = tmap.name
                config["downstream_size"] = size
                config["num_augmentations"] = 0
                experiments.append(Experiment(
                    name=f"{name}_not_pretrained",
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
                ))
                config["model_file"] = model_file
                experiments.append(Experiment(
                    name=f"{name}_pretrained",
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
                ))
    run(
        experiments,
    )
