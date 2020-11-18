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
    period: str = None,
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
                f"To compute {feature} use signal_value, not signal_timeseries.",
            )
        tensor = np.nanmean(
            tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices],
        )
    elif feature == "std":
        if tm.name.endswith("_timeseries"):
            raise KeyError(
                f"To compute {feature} use signal_value, not signal_timeseries.",
            )
        tensor = np.nanstd(
            tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices],
        )
    elif feature == "count":
        if tm.name.endswith("_timeseries"):
            raise KeyError(
                f"To compute {feature} use signal_value, not signal_timeseries.",
            )
        values = tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices]
        tensor = values[~np.isnan(values)].size
    elif feature == "mean_crossing_rate":
        if tm.name.endswith("_timeseries"):
            raise KeyError(
                f"To compute {feature} use signal_value, not signal_timeseries.",
            )
        tensor = tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices]
        mean = np.nanmean(tensor)
        tensor = np.sign(tensor - mean)
        tensor = np.where(tensor[1:] - tensor[:-1])[0].size

    elif feature == "mean_slope":
        if not tm.name.endswith("_timeseries"):
            raise KeyError(
                f"To compute {feature} use signal_timeseries, not signal_value.",
            )
        tensor = tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][:, indices]
        tensor = np.nanmean(
            (tensor[0, 1:] - tensor[0, :-1]) / (tensor[1, 1:] - tensor[1, :-1]),
        )
    else:
        raise KeyError("Unable to compute feature {feature}.")

    tensor = missing_imputation(tm.name, tensor, imputation_type)

    if tm.name.endswith("_timeseries") and feature in [
        "min",
        "max",
        "median",
        "first",
        "last",
    ]:
        # Obtain time indice where the feature is found
        if feature in ("last", "first"):
            sample_time = -1 if feature == "last" else 0
        else:  # min, max, median
            # We obtain the argmin of the absolute value of the difference, that is
            # the index of the sample that has the closest value to the feature
            # If there are more than two values with the feature value,
            # this approach will return the first one. If the period is pre event,
            # we want the last one (closest to the event) so the array is reversed
            if period == "pre":
                sample_time = abs(
                    np.flip(
                        tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][
                            0,
                            indices,
                        ]
                        - tensor,
                    ),
                ).argmin()
                # As we reversed the array, we recompute the original indice
                sample_time = len(indices) - sample_time - 1
            else:
                sample_time = abs(
                    tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][0, indices]
                    - tensor,
                ).argmin()

        time = tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][1, indices][
            sample_time
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
