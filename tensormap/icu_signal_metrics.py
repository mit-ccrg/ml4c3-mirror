# Imports: standard library
from typing import List

# Imports: third party
import h5py
import numpy as np

# Imports: first party
from definitions.icu import ICU_TMAPS_METADATA
from tensormap.TensorMap import TensorMap


def compute_feature(
    tm: TensorMap,
    hd5: h5py.File,
    visit: str,
    indices: List[int],
    feature: str,
    period: str,
    imputation_type: str = None,
    **kwargs,
):
    if tm.name.endswith("_timeseries") and feature in [
        "mean",
        "std",
        "count",
        "mean_crossing_rate",
    ]:
        raise KeyError(
            f"To compute {feature} use signal_value, not signal_timeseries.",
        )
    if not tm.name.endswith("_timeseries") and feature in ["mean_slope"]:
        raise KeyError(
            f"To compute {feature} use signal_timeseries, not signal_value.",
        )

    if len(indices) == 0:
        tensor = np.array([np.nan])
    elif feature == "raw":
        if tm.name.endswith("_timeseries"):
            tensor = tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][:, indices]
        else:
            tensor = tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices]
    elif feature == "mean_slope":
        values = tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][:, indices]
        values = np.delete(values, np.where(np.isnan(values))[1], 1)
        if values.size <= 1:
            tensor = np.array([np.nan])
        else:
            tensor = np.nanmean(
                (values[0, 1:] - values[0, :-1]) / (values[1, 1:] - values[1, :-1]),
            )
    else:
        if tm.name.endswith("_timeseries"):
            values = tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][0, indices]
        else:
            values = tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][indices]

        values = values[~np.isnan(values)]
        if feature == "count":
            tensor = values.size
        elif values.size == 0:
            tensor = np.array([np.nan])
        elif feature in ("last", "first"):
            tensor = values[-1] if feature == "last" else values[0]
        elif feature == "min":
            tensor = np.min(values)
        elif feature == "max":
            tensor = np.max(values)
        elif feature == "median":
            tensor = np.median(values)
        elif feature == "mean":
            tensor = np.mean(values)
        elif feature == "std":
            tensor = np.std(values)
        elif feature == "mean_crossing_rate":
            mean = np.mean(values)
            values = np.sign(values - mean)
            tensor = np.where(values[1:] - values[:-1])[0].size
        else:
            raise KeyError("Unable to compute feature {feature}.")

    tensor = missing_imputation(
        tm=tm,
        hd5=hd5,
        visit=visit,
        indices=indices,
        period=period,
        tensor=tensor,
        imputation_type=imputation_type,
        **kwargs,
    )

    if tm.name.endswith("_timeseries") and feature in [
        "min",
        "max",
        "median",
        "first",
        "last",
    ]:
        # Obtain time indice where the feature is found
        if np.isnan(tensor).all():
            tensor = np.array([np.nan, np.nan])
        elif feature in ("last", "first"):
            sample_time = -1 if feature == "last" else 0
        else:
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
    tm: TensorMap,
    hd5: h5py.File,
    visit: str,
    indices: List[int],
    period: str,
    tensor: np.ndarray,
    imputation_type: str = None,
    **kwargs,
):
    if imputation_type == "sample_and_hold":
        if len(tensor) == 0 or np.isnan(tensor).all():
            if period == "pre":
                values = tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][
                    : indices[-1]
                ]
                indice = -1
            else:
                values = tm.tensor_from_file(tm, hd5, visits=visit, **kwargs)[0][
                    indices[0] :
                ]
                indice = 0
            imputation = values[~np.isnan(values)]
            if imputation.size == 0:
                imputation = np.array([np.nan])
            tensor = np.array([imputation[indice]])
    elif imputation_type:
        name = tm.name.replace(f"_{imputation_type}", "")
        imputation = ICU_TMAPS_METADATA[name][imputation_type]
        tensor = np.nan_to_num(tensor, nan=imputation)
        if len(tensor) == 0:
            tensor = np.array([imputation])
    return tensor
