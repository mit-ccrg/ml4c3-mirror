# Imports: standard library
import pickle
import logging
from typing import List

# Imports: third party
import numpy as np


def multiply_signals_by_factor(bundle, factor: float):
    logging.info(f"Adding 2 artificial cluster multiplying by {factor}")
    patients = list(bundle.patients.values())
    for patient in patients:
        patient.outcome["outcome"] = "Alive"

    half_patients = patients[: int(len(patients) / 2)]
    for patient in half_patients:
        patient.outcome["outcome"] = "Dead"
        for signal in patient.signals.values():
            signal.values = signal.values * factor
    return bundle


def keep_only_signal(bundle, signal_name: str):
    logging.info(f"Removing all signals apart from {signal_name}")
    signals = bundle.any_patient().signals
    signals_to_delete = [sig for sig in signals if sig != signal_name]

    patients = list(bundle.patients.values())
    for patient in patients:
        for signal in signals_to_delete:
            del patient.signals[signal]

    remaining_signals = bundle.any_patient().signals
    print("Left with signals: ", remaining_signals)
    return bundle


def exclude_signals(bundle, signal_list: List[str]):
    logging.info(f"Removing signals: {exclude_signals}")
    for patient in bundle.patients.values():
        for signal2exclude in signal_list:
            del patient.signals[signal2exclude]
    return bundle


def reduce_to_cores(bundle, cores_file):
    logging.info("Using only risk cores...")

    with open(cores_file, "rb") as f:
        cores = pickle.load(f)

    allowed_patients = cores["minimum"] + cores["moderate"] + cores["maximum"]
    all_patients = len(bundle.patients)
    patients_to_delete = []
    for p in bundle.patients.keys():
        if p not in allowed_patients:
            patients_to_delete.append(p)
    for p in patients_to_delete:
        del bundle.patients[p]

    logging.info(f"Patient list reduced from {all_patients} to {len(bundle.patients)}")
    return bundle


def crop_time(bundle, max_len: int):
    logging.info(f"Cropping signals to {max_len}h")
    for patient in bundle.patients.values():
        for signal in patient.signals.values():
            start_time = signal.time[0]
            end_time = signal.time[0] + max_len * 3600
            idx = np.where((signal.time >= start_time) & (signal.time <= end_time))[0]
            signal.time = signal.time[idx]
            signal.values = signal.values[idx]

    return bundle
