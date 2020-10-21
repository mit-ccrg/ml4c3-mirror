# Imports: standard library
from typing import List

# Imports: third party
import h5py
import numpy as np

# Imports: first party
from ml4c3.tensormap.TensorMap import TensorMap


def compute_feature(
    tm: TensorMap,
    hd5: h5py.File,
    visit: str,
    indices: List[int],
    feature: str,
    **kwargs
):
    if feature == "raw":
        tensor = tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices]
    elif feature == "last":
        values = tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices]
        try:
            tensor = values[~np.isnan(values)][-1]
        except IndexError:
            tensor = np.nan
    elif feature == "first":
        values = tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices]
        try:
            tensor = values[~np.isnan(values)][0]
        except IndexError:
            tensor = np.nan
    elif feature == "min":
        tensor = np.nanmin(
            tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices],
        )
    elif feature == "max":
        tensor = np.nanmax(
            tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices],
        )
    elif feature == "mean":
        tensor = np.nanmean(
            tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices],
        )
    elif feature == "median":
        tensor = np.nanmedian(
            tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices],
        )
    elif feature == "std":
        tensor = np.nanstd(
            tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices],
        )
    else:
        raise KeyError("Unable to compute feature {feature}.")
    return tensor if isinstance(tensor, np.ndarray) else np.array([tensor])
