# Imports: standard library
import os
import argparse
from typing import Dict, List, Callable
from functools import partial

# Imports: third party
import h5py
import numpy as np
from matplotlib import pyplot as plt

# Imports: first party
from ml4c3.datasets import train_valid_test_datasets
from ml4c3.normalizer import Standardize, ZeroMeanStd1
from ml4c3.validators import RangeValidator, validator_voltage_no_zero_padding
from ml4c3.tensormap.ecg import (
    ECG_PREFIX,
    ECG_REST_LEADS_ALL,
    tmaps,
    make_voltage_tff,
    get_ecg_age_from_hd5,
)
from ml4c3.tensormap.TensorMap import TensorMap


def most_recent_ecg_date(hd5: h5py.File) -> List[str]:
    return sorted(list(hd5[ECG_PREFIX]), reverse=True)


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
    keys = "ecg_paxis_md", "ecg_raxis_md", "ecg_taxis_md"
    out = []
    for key in keys:
        tmap = tmaps[key]
        tmap.loss = "mse"
        tmap.time_series_filter = most_recent_ecg_date
        out.append(tmap)
    return out


def get_interval_tmaps() -> List[TensorMap]:
    keys = "ecg_rate_md", "ecg_pr_md", "ecg_qrs_md", "ecg_qt_md"
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


# Data generators
def get_pretraining_datasets(
    ecg_length: int,
    augmentation_strengths: Dict[str, float],
    num_augmentations: int,
    hd5_folder: str,
    num_workers: int,
    batch_size: int,
    train_csv: str,
    valid_csv: str,
    test_csv: str,
):
    augmentations = apply_augmentation_strengths(
        augmentation_strengths,
        num_augmentations,
    )
    ecg_tmap = get_ecg_tmap(ecg_length, augmentations)
    return train_valid_test_datasets(
        [ecg_tmap],
        get_pretraining_tasks(),
        tensors=hd5_folder,
        batch_size=batch_size,
        num_workers=num_workers,
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_csv=test_csv,
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
    train_csv: str,
    valid_csv: str,
    test_csv: str,
    output_folder: str,
    **kwargs,
):
    os.makedirs(output_folder, exist_ok=True)
    augmentations = augmentation_dict()
    ecgs_per_augmentation = 3

    (train_dataset, _, _), _, cleanups = get_pretraining_datasets(
        ecg_length=2500,
        augmentation_strengths={name: 0.0 for name in augmentations},
        num_augmentations=0,
        hd5_folder=hd5_folder,
        num_workers=1,
        batch_size=ecgs_per_augmentation,
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_csv=test_csv,
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


MODES = {
    "augmentation_demo": demo_augmentations,
    # TODO: add explore
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        help=f"Which data functionality to use. Choose between {list(MODES)}.",
        choices=list(MODES),
    )
    parser.add_argument(
        "--train_csv",
        help="Path to CSV with Sample IDs to reserve for training.",
        required=True,
    )
    parser.add_argument(
        "--valid_csv",
        help=("Path to CSV with Sample IDs to reserve for validation"),
        required=True,
    )
    parser.add_argument(
        "--test_csv",
        help=("Path to CSV with Sample IDs to reserve for testing."),
        required=True,
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
    demo_augmentations(**args.__dict__)
