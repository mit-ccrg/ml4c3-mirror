# Imports: standard library
import logging
import datetime
from typing import Dict, List, Union, Callable

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from ml4c3.normalizer import MinMax, RobustScaler
from ml4c3.validators import validator_no_nans, validator_not_all_zero
from ml4c3.definitions.echo import ECHO_PREFIX, ECHO_MRN_COLUMN, ECHO_DATETIME_COLUMN
from ml4c3.tensormap.TensorMap import (
    Dates,
    TensorMap,
    ChannelMap,
    PatientData,
    Interpretation,
    is_dynamic_shape,
    outcome_channels,
)

tmaps: Dict[str, TensorMap] = {}

# Define name and statistical properties of continuous properties to enable standardization. These values are calculated from the entire dataset.
echo_measures_continuous: Dict[str, Dict[str, Union[int, float]]] = {
    "av_area": {"median": 1, "iqr": 2, "min": 0, "max": 3},
    "av_peak_gradient": {"median": 1, "iqr": 2, "min": 0, "max": 3},
    "av_mean_gradient": {"median": 1, "iqr": 2, "min": 0, "max": 3},
    "av_peak_velocity": {"median": 1, "iqr": 2, "min": 0, "max": 3},
}


def get_echo_dates(data: PatientData) -> Dates:
    return data[ECHO_PREFIX][ECHO_DATETIME_COLUMN]


def _make_echo_tff_continuous(key: str) -> Callable:
    def tensor_from_file(tm: TensorMap, data: PatientData) -> np.ndarray:
        echo_dates = tm.time_series_filter(data)
        tensor = data[ECHO_PREFIX].loc[echo_dates.index, key].to_numpy()
        if is_dynamic_shape(tm):
            tensor = tensor[:, None]
        return tensor

    return tensor_from_file


# Create dictionary: key is tmap name, value is tmap key in the source CSV
continuous_tmap_names_and_keys = {
    "av_area": "AV Area",
    "av_mean_gradient": "AV Mean Gradient",
    "av_peak_gradient": "AV Peak Gradient",
    "av_peak_velocity": "AV Peak Velocity",
}

# Iterate over tensor map names and CSV keys
for tmap_name, tmap_key in continuous_tmap_names_and_keys.items():

    # Make tmaps for both raw and scaled data
    for standardize in ["", "_scaled"]:
        normalizer = None
        if standardize == "_scaled":
            normalizer = RobustScaler(
                median=echo_measures_continuous[tmap_name]["median"],
                iqr=echo_measures_continuous[tmap_name]["iqr"],
            )
        else:
            normalizer = None

        tmaps[tmap_name + standardize] = TensorMap(
            name=tmap_name + standardize,
            shape=(1,),
            interpretation=Interpretation.CONTINUOUS,
            path_prefix=ECHO_PREFIX,
            tensor_from_file=_make_echo_tff_continuous(key=tmap_key),
            validators=validator_no_nans,
            normalizers=normalizer,
            time_series_limit=0,
            time_series_filter=get_echo_dates,
        )


tmap_name = "echo_datetime"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    shape=(1,),
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECHO_PREFIX,
    tensor_from_file=_make_echo_tff_continuous(key=ECHO_DATETIME_COLUMN),
    validators=validator_no_nans,
    time_series_limit=0,
    time_series_filter=get_echo_dates,
)
