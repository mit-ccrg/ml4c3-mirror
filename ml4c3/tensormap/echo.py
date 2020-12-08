# Imports: standard library
from typing import Dict, Union, Callable

# Imports: third party
import numpy as np

# Imports: first party
from definitions.echo import ECHO_PREFIX, ECHO_DATETIME_COLUMN
from ml4c3.normalizer import RobustScaler
from ml4c3.validators import RangeValidator, validator_no_nans, validator_not_all_zero
from ml4c3.tensormap.TensorMap import (
    Dates,
    TensorMap,
    PatientData,
    Interpretation,
    is_dynamic_shape,
)

tmaps: Dict[str, TensorMap] = {}

# Define name and statistical properties of continuous properties to enable
# standardization. These values are calculated from the entire dataset.
echo_measures_continuous: Dict[str, Dict[str, Union[int, float]]] = {
    "av_area": {"median": 1.39, "iqr": 1.03, "min": 0, "max": 8},
    "av_mean_gradient": {"median": 12, "iqr": 14, "min": 0, "max": 100},
    "av_peak_velocity": {"median": 241.07, "iqr": 239, "min": 30, "max": 750},
}

# Create dictionary: key is tmap name, value is tmap key in the source CSV
continuous_tmap_names_and_keys = {
    "av_area": "AV Area",
    "av_mean_gradient": "AV Mean Gradient",
    "av_peak_velocity": "AV Peak Velocity",
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
            validators=RangeValidator(
                minimum=echo_measures_continuous[tmap_name]["min"],
                maximum=echo_measures_continuous[tmap_name]["max"],
            ),
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


def tensor_from_file_aortic_stenosis_category(
    tm: TensorMap,
    data: PatientData,
) -> np.ndarray:
    """Categorizes aortic stenosis as none, mild, moderate, or severe
    from aortic valve mean gradient"""
    echo_dates = tm.time_series_filter(data)
    av_mean_gradient_key = continuous_tmap_names_and_keys["av_mean_gradient"]
    av_mean_gradients = data[ECHO_PREFIX].loc[echo_dates.index, av_mean_gradient_key]

    # Initialize tensor array of zeros where each row is the channel map
    tensor = np.zeros((len(av_mean_gradients), 4))

    # Iterate through the peak velocities and mean gradients from all echos
    for idx, av_mean_gradient in enumerate(av_mean_gradients):
        if av_mean_gradient < 10:
            category = "none"
        elif 10 <= av_mean_gradient < 20:
            category = "mild"
        elif 20 <= av_mean_gradient < 40:
            category = "moderate"
        elif av_mean_gradient >= 40:
            category = "severe"
        else:
            continue
        tensor[idx, tm.channel_map[category]] = 1
    return tensor


tmap_name = "as"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    channel_map={"none": 0, "mild": 1, "moderate": 2, "severe": 3},
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECHO_PREFIX,
    tensor_from_file=tensor_from_file_aortic_stenosis_category,
    validators=validator_not_all_zero,
    time_series_limit=0,
    time_series_filter=get_echo_dates,
)


def make_aortic_stenosis_binary(threshold: float) -> Callable:
    def tensor_from_file(
        tm: TensorMap,
        data: PatientData,
    ) -> np.ndarray:
        echo_dates = tm.time_series_filter(data)
        mean_gradient_key = continuous_tmap_names_and_keys["av_mean_gradient"]
        mean_gradients = data[ECHO_PREFIX].loc[echo_dates.index, mean_gradient_key]
        tensor = np.zeros((len(echo_dates), 2))
        for idx, mean_gradient in enumerate(mean_gradients):
            if np.isnan(mean_gradient):
                continue
            tensor[idx, 1 if mean_gradient >= threshold else 0] = 1
        return tensor

    return tensor_from_file


tmap_name = "as_any"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    channel_map={"no_as": 0, "as": 1},
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECHO_PREFIX,
    tensor_from_file=make_aortic_stenosis_binary(threshold=10),
    validators=validator_not_all_zero,
    time_series_limit=0,
    time_series_filter=get_echo_dates,
)


def make_aortic_stenosis_single_category(severity: str) -> Callable:
    if severity == "mild":
        low = 10
        high = 20
    elif severity == "moderate":
        low = 20
        high = 40
    elif severity == "severe":
        low = 40
        high = 10000
    else:
        raise ValueError(f"Unknown Aortic Stenosis severity: {severity}")

    def tensor_from_file(tm: TensorMap, data: PatientData) -> np.ndarray:
        echo_dates = tm.time_series_filter(data)
        mean_gradient_key = continuous_tmap_names_and_keys["av_mean_gradient"]
        mean_gradients = data[ECHO_PREFIX].loc[echo_dates.index, mean_gradient_key]
        tensor = np.zeros((len(echo_dates), 2))
        for idx, mean_gradient in enumerate(mean_gradients):
            if np.isnan(mean_gradient):
                continue
            tensor[idx, 1 if low <= mean_gradient < high else 0] = 1
        return tensor

    return tensor_from_file


for severity in ["mild", "moderate", "severe"]:
    tmap_name = f"as_{severity}"
    tmaps[tmap_name] = TensorMap(
        name=tmap_name,
        channel_map={f"not_{severity}": 0, severity: 1},
        interpretation=Interpretation.CATEGORICAL,
        path_prefix=ECHO_PREFIX,
        tensor_from_file=make_aortic_stenosis_single_category(severity),
        validators=validator_not_all_zero,
        time_series_limit=0,
        time_series_filter=get_echo_dates,
    )
