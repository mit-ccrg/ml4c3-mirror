# Imports: standard library
import os
import json
import shutil
from typing import Dict, List, Tuple

# Imports: third party
import numpy as np
import pandas as pd
from ray.tune import Analysis, run, stopper
from hyperoptimize import EarlyStopping, PretrainingTrainable, build_pretraining_model
from ray.tune.experiment import Experiment


def _get_training_dfs(analysis_df: pd.DataFrame) -> List[pd.DataFrame]:
    return [
        pd.read_csv(os.path.join(folder, "progress.csv"))
        for folder in analysis_df["logdir"]
    ]


def _get_configs(folder: str, num_configs: int) -> Tuple[List[str], List[Dict]]:
    """
    gets num_configs config dictionaries from ray output folder
    The configs will be well distributed in terms of validation loss
    """
    df = Analysis(folder).dataframe()
    training_dfs = _get_training_dfs(df)
    best_val_loss_col = "best_val_loss"
    df[best_val_loss_col] = [tdf["val_loss"].min() for tdf in training_dfs]
    df = df[df[best_val_loss_col] < 7]
    if len(df) < num_configs:
        raise ValueError(
            f"Cannot select {num_configs} trials with only {len(df)} well trained models.",
        )

    df = df.sort_values(by=best_val_loss_col)
    indices = np.linspace(
        0,
        len(df) - 1,
        num_configs,
        dtype=int,
    )  # select models with evenly spaced quantiles
    config_cols = [col for col in df.columns if "config" in col]
    config_df = df[config_cols].rename(
        columns={col: col.replace("config/", "") for col in config_cols},
    )
    configs = config_df.iloc[indices].to_dict("records")
    names = [f"{i}_best_val_loss_{df[best_val_loss_col].iloc[i]}" for i in indices]
    return names, configs


def _run_experiments(
    epochs: int,
    ray_output_folder: str,
    output_folder: str,
    num_experiments: int,
    patience: int,
    cpus_per_model: int,
    gpus_per_model: float,
):
    names, configs = _get_configs(ray_output_folder, num_experiments)
    stopper = EarlyStopping(patience=patience, max_epochs=epochs)
    experiments = [
        Experiment(
            name,
            run=PretrainingTrainable,
            config=config,
            keep_checkpoints_num=1,
            checkpoint_score_attr="val_loss",
            checkpoint_freq=1,
            stop=stopper,
            resources_per_trial={
                "cpu": cpus_per_model,
                "gpu": gpus_per_model,
            },
            local_dir=output_folder,
        )
        for name, config in zip(names, configs)
    ]
    result = run(
        experiments,
    )
    os.makedirs(output_folder, exist_ok=True)
    for trial in result.trials:
        checkpoint = trial.checkpoint
        path = checkpoint.value
        name = next(n for n in path.split("/") if "best_val_loss" in n)
        with open(os.path.join(output_folder, f"{name}.json"), "w") as f:
            json.dump(checkpoint.result, f, default=str)
        shutil.copy(
            os.path.join(path, "pretraining_model.h5"),
            os.path.join(output_folder, f"{name}.h5"),
        )
