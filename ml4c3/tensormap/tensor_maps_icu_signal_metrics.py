# Imports: standard library
from typing import List

# Imports: third party
import h5py
import numpy as np

# Imports: first party
from ml4c3.definitions.icu import ICU_TMAPS_METADATA
from ml4c3.tensormap.TensorMap import TensorMap


def compute_feature(
    tm: TensorMap,
    hd5: h5py.File,
    visit: str,
    indices: List[int],
    feature: str,
    imputation_type: str = None,
    **kwargs,
):
    if len(indices) == 0:
        tensor = np.array([])
    elif feature == "raw":
        if tm.name.endswith("_timeseries"):
            tensor = tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][:, indices]
        else:
            tensor = tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices]
    elif feature in ("last", "first"):
        k = -1 if feature == "last" else 0
        if tm.name.endswith("_timeseries"):
            values = tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][0, indices]
        else:
            values = tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices]
        try:
            tensor = values[~np.isnan(values)][k]
        except IndexError:
            tensor = np.nan
    elif feature == "min":
        if tm.name.endswith("_timeseries"):
            tensor = np.nanmin(
                tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][0, indices],
            )
        else:
            tensor = np.nanmin(
                tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices],
            )
    elif feature == "max":
        if tm.name.endswith("_timeseries"):
            tensor = np.nanmax(
                tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][0, indices],
            )
        else:
            tensor = np.nanmax(
                tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices],
            )
    elif feature == "median":
        if tm.name.endswith("_timeseries"):
            tensor = np.nanmedian(
                tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][0, indices],
            )
        else:
            tensor = np.nanmedian(
                tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices],
            )
    elif feature == "mean":
        if tm.name.endswith("_timeseries"):
            raise KeyError(
                "To compute the mean use signal_value, not signal_timeseries.",
            )
        tensor = np.nanmean(
            tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices],
        )
    elif feature == "std":
        if tm.name.endswith("_timeseries"):
            raise KeyError(
                "To compute the std use signal_value, not signal_timeseries.",
            )
        tensor = np.nanstd(
            tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices],
        )
    elif feature == "count":
        if tm.name.endswith("_timeseries"):
            raise KeyError(
                "To compute the number of counts use signal_value, "
                "not signal_timeseries.",
            )
        values = tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices]
        tensor = values[~np.isnan(values)].size
    else:
        raise KeyError("Unable to compute feature {feature}.")

    tensor = missing_imputation(tm.name, tensor, imputation_type)

    if tm.name.endswith("_timeseries") and feature != "raw":
        sample_time = np.where(
            tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][0, indices]
            == tensor,
        )[0]
        if len(sample_time) == 0:
            raise KeyError("Unable to compute feature {feature}.")
        time = tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][1, indices][
            sample_time[-1]
        ]
        tensor = np.array([tensor, time])

    return tensor if isinstance(tensor, np.ndarray) else np.array([tensor])


def missing_imputation(
    tm_name: str,
    tensor: np.ndarray,
    imputation_type: str = None,
):
    if imputation_type:
        name = tm_name.replace(f"_{imputation_type}", "")
        imputation = ICU_TMAPS_METADATA[name][imputation_type]
        tensor = np.nan_to_num(tensor, nan=imputation)
        if len(tensor) == 0:
            tensor = np.array([imputation])
    return tensor
