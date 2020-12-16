# Imports: standard library
import os
import argparse
from typing import Dict, List, Callable
from functools import partial

# Imports: third party
import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Imports: first party
from ml4c3.datasets import train_valid_test_datasets
from ml4c3.normalizer import Standardize, ZeroMeanStd1
from ml4c3.validators import RangeValidator, validator_voltage_no_zero_padding
from ml4c3.explorations import _tensors_to_df
from ml4c3.tensormap.ecg import (
    ECG_PREFIX,
    ECG_REST_LEADS_ALL,
    tmaps,
    make_voltage_tff,
    get_ecg_age_from_hd5,
)
from ml4c3.tensormap.ecg_labels import tmaps as ecg_label_tmaps
from ml4c3.tensormap.TensorMap import TensorMap


PRETRAIN_NAMES = "pretrain_train", "pretrain_valid", "pretrain_test"
DOWNSTREAM_SIZES = 500, 1000, 10000
DOWNSTREAM_NAMES = "downstream_train", "downstream_valid", "downstream_test"


def most_recent_ecg_date(hd5: h5py.File) -> List[str]:
    return [max(hd5[ECG_PREFIX])]


# ECG augmentations
# All strengths have a reasonable range between 0 and 1
def warp(ecg, strength):
    strength *= 0.2  # reasonable range between -> [0, 1]

    i = np.linspace(0, 1, len(ecg))
    envelope = strength * (0.5 - np.abs(0.5 - i))
    warped = i + envelope * (
        np.sin(np.random.rand() * 5 * i + np.random.randn() * 5)
        + np.cos(np.random.rand() * 5 * i + np.random.randn() * 5)
    )
    warped_ecg = np.zeros_like(ecg)
    for j in range(ecg.shape[1]):
        warped_ecg[:, j] = np.interp(i, warped, ecg[:, j])
    return warped_ecg


def crop(ecg, strength):
    strength *= 1  # reasonable range between -> [0, 1]

    cropped_ecg = ecg.copy()
    crop_len = int(np.random.randint(len(ecg)) * strength)
    crop_start = max(0, np.random.randint(-crop_len, len(ecg)))
    cropped_ecg[crop_start : crop_start + crop_len] = 0
    return cropped_ecg


def gaussian_noise(ecg, strength):
    strength *= 2  # reasonable range between -> [0, 1]

    noise_frac = np.random.rand() * strength
    return ecg + noise_frac * np.random.randn(*ecg.shape)


def roll(ecg, strength):
    if strength == 0:
        return ecg
    amount = np.random.randint(ecg.shape[0])
    return np.roll(ecg, amount, axis=0)


def baseline_drift(ecg, strength):
    strength *= 2  # reasonable range between -> [0, 1]

    frequency = (
        (np.random.rand() * 20 + 10) * 10 / 60
    )  # typical breaths per second for an adult
    phase = np.random.rand() * 2 * np.pi
    drift = strength * np.sin(np.linspace(0, 1, len(ecg)) * frequency + phase)
    return ecg + drift[:, np.newaxis]


def augmentation_dict():
    return {
        func.__name__: func
        for func in [warp, crop, gaussian_noise, roll, baseline_drift]
    }


def chain_apply(input_value, functions: List[Callable]):
    out = input_value
    for func in functions:
        out = func(input_value)
    return out


def apply_augmentation_strengths(
    augmentation_strengths: Dict[str, float],
    num_augmentations: int,
):
    aug_dict = augmentation_dict()
    if num_augmentations == 0:
        return []
    augmentations = [
        partial(aug_dict[func_name], strength=strength)
        for func_name, strength in augmentation_strengths.items()
    ]
    return [
        lambda ecg: chain_apply(
            ecg,
            np.random.choice(augmentations, num_augmentations),
        ),
    ]


# ECG tmaps
def get_ecg_tmap(length: int, augmentations: List) -> TensorMap:
    return TensorMap(
        name="ecg",
        shape=(length, len(ECG_REST_LEADS_ALL)),
        path_prefix=ECG_PREFIX,
        tensor_from_file=make_voltage_tff(exact_length=False),
        normalizers=ZeroMeanStd1(),  # TODO: build clip normalizer? Need to pick a physioligical range
        channel_map=ECG_REST_LEADS_ALL,
        time_series_limit=0,
        time_series_filter=most_recent_ecg_date,
        validators=validator_voltage_no_zero_padding,  # TODO: physiolical range check?
        augmenters=augmentations,
    )


# Pretraining tmaps
def get_axis_tmaps() -> List[TensorMap]:
    keys = "ecg_paxis_std_md", "ecg_raxis_std_md", "ecg_taxis_std_md"
    out = []
    for key in keys:
        tmap = tmaps[key]
        tmap.loss = "mse"
        tmap.time_series_filter = most_recent_ecg_date
        out.append(tmap)
    return out


def get_interval_tmaps() -> List[TensorMap]:
    keys = "ecg_rate_std_md", "ecg_pr_std_md", "ecg_qrs_std_md", "ecg_qt_std_md"
    out = []
    for key in keys:
        tmap = tmaps[key]
        tmap.loss = "mse"
        tmap.time_series_filter = most_recent_ecg_date
        out.append(tmap)
    return out


def get_pretraining_tasks() -> List[TensorMap]:
    return get_interval_tmaps() + get_axis_tmaps()


# Downstream tmaps
# Secondary tasks
def get_age_tmap() -> TensorMap:
    return TensorMap(
        name="age",
        path_prefix=ECG_PREFIX,
        loss="mse",
        tensor_from_file=get_ecg_age_from_hd5,
        shape=(1,),
        time_series_limit=0,
        time_series_filter=most_recent_ecg_date,
        validators=RangeValidator(20, 90),
        normalizers=Standardize(54, 21),
    )


def get_sex_tmap() -> TensorMap:
    sex = tmaps["ecg_sex"]
    sex.time_series_filter = most_recent_ecg_date
    return sex


# Primary tasks
def get_afib_tmap() -> TensorMap:
    afib = ecg_label_tmaps["atrial_fibrillation"]
    afib.time_series_filter = most_recent_ecg_date
    return afib


def get_lvh_tmap() -> TensorMap:
    lvh = ecg_label_tmaps["lvh"]
    lvh.time_series_filter = most_recent_ecg_date
    return lvh


def get_downstream_tmaps() -> List[TensorMap]:
    return [
        get_age_tmap(),
        get_sex_tmap(),
        get_afib_tmap(),
        get_lvh_tmap(),
    ]


# Data generators
def get_pretraining_datasets(
    ecg_length: int,
    augmentation_strengths: Dict[str, float],
    num_augmentations: int,
    hd5_folder: str,
    num_workers: int,
    batch_size: int,
    csv_folder: str,
):
    augmentations = apply_augmentation_strengths(
        augmentation_strengths,
        num_augmentations,
    )
    ecg_tmap = get_ecg_tmap(ecg_length, augmentations)
    csvs = get_pretrain_csv_paths(csv_folder)
    return train_valid_test_datasets(
        [ecg_tmap],
        get_pretraining_tasks(),
        tensors=hd5_folder,
        batch_size=batch_size,
        num_workers=num_workers,
        train_csv=csvs[0],
        valid_csv=csvs[1],
        test_csv=csvs[2],
    )


def downstream_tmap_from_name(name: str) -> TensorMap:
    return next(
        tmap for tmap in get_downstream_tmaps()
        if tmap.name == name
    )


def get_downstream_datasets(
    downstream_tmap_name: str,
    downstream_size: int,
    ecg_length: int,
    augmentation_strengths: Dict[str, float],
    num_augmentations: int,
    hd5_folder: str,
    num_workers: int,
    batch_size: int,
    csv_folder: str,
):
    augmentations = apply_augmentation_strengths(
        augmentation_strengths,
        num_augmentations,
    )
    ecg_tmap = get_ecg_tmap(ecg_length, augmentations)
    downstream_tmap = downstream_tmap_from_name(downstream_tmap_name)
    csvs = get_downstream_csv_paths(csv_folder, downstream_tmap, downstream_size)
    return train_valid_test_datasets(
        [ecg_tmap],
        [downstream_tmap],
        tensors=hd5_folder,
        batch_size=batch_size,
        num_workers=num_workers,
        train_csv=csvs[0],
        valid_csv=csvs[1],
        test_csv=csvs[2],
    )


# plot augmentations
def demo_augmentation(
    ecg: np.ndarray,
    augmentation: Callable[[np.ndarray, float], np.ndarray],
    axes: List[plt.Axes],
):
    for axis, strength in zip(axes, np.linspace(0, 1, len(axes))):
        axis.axis("off")  # hide x and y ticks
        axis.plot(augmentation(ecg, strength)[:, 0], c="k")  # plot only lead I


def demo_augmentations(
    hd5_folder: str,
    csv_folder: str,
    output_folder: str,
    **kwargs,
):
    augmentations = augmentation_dict()
    ecgs_per_augmentation = 3

    (train_dataset, _, _), _, cleanups = get_pretraining_datasets(
        ecg_length=2500,
        augmentation_strengths={name: 0.0 for name in augmentations},
        num_augmentations=0,
        hd5_folder=hd5_folder,
        num_workers=1,
        batch_size=ecgs_per_augmentation,
        csv_folder=csv_folder,
    )
    ecgs = next(train_dataset.take(1).as_numpy_iterator())[0]["input_ecg_continuous"]
    for name, augmentation in augmentations.items():
        fig, axes = plt.subplots(
            10,
            ecgs_per_augmentation,
            figsize=(10 * ecgs_per_augmentation, 10),
        )
        for i, ecg in enumerate(ecgs):
            np.random.seed(i)
            demo_augmentation(ecg, augmentation, axes[:, i])
        plt.savefig(os.path.join(output_folder, f"{name}.png"))

    for cleanup in cleanups:
        cleanup()


def _get_pretrain_csv_path(folder: str, name: str) -> str:
    return os.path.join(folder, f"{name}.csv")


def get_pretrain_csv_paths(folder: str) -> List[str]:
    return [_get_pretrain_csv_path(folder, name) for name in PRETRAIN_NAMES]


def _get_downstream_csv_path(folder: str, name: str, tmap: TensorMap, size: int) -> str:
    return os.path.join(folder, f"{name}_{tmap.name}_{size}.csv")


def get_downstream_csv_paths(folder: str, tmap: TensorMap, size: int) -> List[str]:
    return [_get_downstream_csv_path(folder, name, tmap, size) for name in DOWNSTREAM_NAMES]


def _get_null_check_col(tmap: TensorMap) -> str:
    if tmap.channel_map is None:
        return tmap.name
    return f"{tmap.name}_{list(tmap.channel_map)[0]}"


def explore_all_data(
    hd5_folder: str,
    output_folder: str,
    sample_csv: str,
    **kwargs,
):
    """Builds all of the sample CSVs for pretraining and downstream tasks"""
    run_id = "pretraining_explore"
    os.makedirs(os.path.join(args.output_folder, run_id), exist_ok=True)
    sample_freq_tmap = tmaps["ecg_sampling_frequency_pc_continuous"]
    sample_freq_tmap.time_series_filter = most_recent_ecg_date

    ecg_tmaps = [get_ecg_tmap(2500, []), sample_freq_tmap]
    pretraining_tmaps = get_downstream_tmaps()
    downstream_tmaps = get_downstream_tmaps()

    explore_tmaps = ecg_tmaps + pretraining_tmaps + downstream_tmaps
    df = _tensors_to_df(
        tensor_maps_in=explore_tmaps,
        tensors=hd5_folder,
        num_workers=10,
        output_folder=output_folder,
        run_id=run_id,
        valid_ratio=0,
        test_ratio=0,
        export_error=False,
        export_fpath=True,
        export_generator=False,
        sample_csv=sample_csv,
    )
    sample_ids = [
        int(os.path.splitext(os.path.basename(path))[0]) for path in df["fpath"]
    ]
    df["sample_id"] = sample_ids
    df.to_csv(
        os.path.join(output_folder, "pretraining_explore.tsv"),
        index=False,
        sep="\t",
    )

    shuffled_df = df.sample(frac=1, random_state=3005).dropna(
        subset=list(map(_get_null_check_col, ecg_tmaps))  # make sure ECG is always present
    )
    sizes = [
        # pretrain train, pretrain valid, downstream train, downstream valid, test
        int(frac * len(shuffled_df)) for frac in np.cumsum([0.0, 0.7, 0.1, 0.1, 0.05, 0.05])
    ]
    split_dfs = [shuffled_df.iloc[sizes[i]: sizes[i + 1]] for i in range(len(sizes) - 1)]

    # pretraining data
    pretraining_cols = list(map(_get_null_check_col, pretraining_tmaps))  # exclude errors in these cols
    for name, split_df in zip(
            PRETRAIN_NAMES,
            [split_dfs[0], split_dfs[1], split_dfs[-1]]
    ):
        path = _get_pretrain_csv_path(output_folder, name)
        print(f"Writing {len(split_df)} {name} ids to {path}.")
        split_df["sample_id"].dropna(subset=[pretraining_cols]).to_csv(
            path, index=False,
        )
    # finetuning data
    for name, split_df in zip(DOWNSTREAM_NAMES, split_dfs[2:]):
        for tmap in downstream_tmaps:
            not_null_df = split_df.dropna([_get_null_check_col(tmap)])
            for size in DOWNSTREAM_SIZES:
                path = _get_downstream_csv_path(name=name, size=size, folder=output_folder, tmap=tmap)
                print(f"Writing {size} {name} ids to {path}.")
                not_null_df["sample_id"].iloc[:size].to_csv()


MODES: Dict[str, Callable] = {
    "augmentation_demo": demo_augmentations,
    "explore": explore_all_data,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        help=f"Which data functionality to use. Choose between {list(MODES)}.",
        choices=list(MODES),
    )
    parser.add_argument(
        "--sample_csv",
        help=("Path to CSV with all Sample IDs to use."),
    )
    parser.add_argument(
        "--csv_folder",
        help="Folder where sample CSVs are.",
    )
    parser.add_argument(
        "--hd5_folder",
        help="Path to folder containing hd5s.",
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        default="./recipes-output",
        help="Path to output folder for recipes.py runs.",
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    MODES[args.mode](**args.__dict__)
