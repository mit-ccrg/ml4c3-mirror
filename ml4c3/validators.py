# Imports: third party
import numpy as np

# Imports: first party
from ml4c3.definitions.ecg import ECG_ZERO_PADDING_THRESHOLD
from ml4c3.tensormap.TensorMap import TensorMap, PatientData


def validator_clean_mrn(tm: TensorMap, tensor: np.ndarray, data: PatientData):
    int(tensor)


def validator_not_all_zero(tm: TensorMap, tensor: np.ndarray, data: PatientData):
    if np.count_nonzero(tensor) == 0:
        raise ValueError(
            f"TensorMap {tm.name} failed all-zero check on sample {data.id}",
        )


def validator_no_empty(tm: TensorMap, tensor: np.ndarray, data: PatientData):
    if any(tensor == ""):
        raise ValueError(
            f"TensorMap {tm.name} failed empty string check on sample {data.id}",
        )


def validator_no_nans(tm: TensorMap, tensor: np.ndarray, data: PatientData):
    if np.isnan(tensor).any():
        raise ValueError(
            f"TensorMap {tm.name} failed no nans check on sample {data.id}.",
        )


def validator_no_negative(tm: TensorMap, tensor: np.ndarray, data: PatientData):
    if any(tensor < 0):
        raise ValueError(
            f"TensorMap {tm.name} failed non-negative check on sample {data.id}",
        )


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
            raise ValueError(f"Lead {cm} is zero-padded for ECG in {data.id}")


class RangeValidator:
    def __init__(self, minimum: float, maximum: float):
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, tm: TensorMap, tensor: np.ndarray, data: PatientData):
        if not ((tensor > self.minimum).all() and (tensor < self.maximum).all()):
            raise ValueError(f"TensorMap {tm.name} failed range check.")

    def __str__(self):
        return f"Range Validator (min, max) = ({self.minimum}, {self.maximum})"

    def __repr__(self):
        return self.__str__()
