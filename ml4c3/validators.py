# Imports: standard library
import logging

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from definitions.ecg import ECG_ZERO_PADDING_THRESHOLD
from ml4c3.tensormap.TensorMap import TensorMap, PatientData


def validator_clean_mrn(tm: TensorMap, tensor: np.ndarray, data: PatientData):
    int(tensor)


def validator_not_all_zero(tm: TensorMap, tensor: np.ndarray, data: PatientData):
    if np.count_nonzero(tensor) == 0:
        error_message = f"TensorMap {tm.name} failed all-zero check"
        logging.debug(f"{error_message} on sample {data.id}")
        raise ValueError(error_message)


def validator_no_empty(tm: TensorMap, tensor: np.ndarray, data: PatientData):
    if any(tensor == ""):
        error_message = f"TensorMap {tm.name} failed empty string check"
        logging.debug(f"{error_message} on sample {data.id}")
        raise ValueError(error_message)


def validator_no_nans(tm: TensorMap, tensor: np.ndarray, data: PatientData):
    if pd.isnull(tensor).any():
        error_message = f"TensorMap {tm.name} failed no nans check"
        logging.debug(f"{error_message} on sample {data.id}")
        raise ValueError(error_message)


def validator_no_negative(tm: TensorMap, tensor: np.ndarray, data: PatientData):
    if any(tensor < 0):
        error_message = f"TensorMap {tm.name} failed non-negative check"
        logging.debug(f"{error_message} on sample {data.id}")
        raise ValueError(error_message)


def validator_voltage_no_zero_padding(
    tm: TensorMap,
    tensor: np.ndarray,
    data: PatientData,
):
    for cm, idx in tm.channel_map.items():
        lead_length = tm.shape[-1]
        lead = tensor[..., tm.channel_map[cm]]
        num_zero = lead_length - np.count_nonzero(lead)
        if num_zero > ECG_ZERO_PADDING_THRESHOLD * lead_length:
            error_message = f"Lead {cm} is zero-padded"
            logging.debug(f"{error_message} on sample {data.id}")
            raise ValueError(error_message)


class RangeValidator:
    def __init__(self, minimum: float, maximum: float):
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, tm: TensorMap, tensor: np.ndarray, data: PatientData):
        if not ((tensor > self.minimum).all() and (tensor < self.maximum).all()):
            error_message = f"TensorMap {tm.name} failed range check"
            logging.info(f"{error_message} on sample {data.id}")
            raise ValueError(error_message)

    def __str__(self):
        return f"Range Validator (min, max) = ({self.minimum}, {self.maximum})"

    def __repr__(self):
        return self.__str__()
