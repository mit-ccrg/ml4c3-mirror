# pylint: disable=redefined-outer-name
# Imports: standard library
import logging
from typing import Dict, List, Union, Callable

# Imports: third party
import numpy as np

# Imports: first party
from definitions.sts import STS_PREFIX, STS_SURGERY_DATE_COLUMN
from ml4c3.normalizer import MinMax, RobustScalePopulation
from ml4c3.validators import validator_no_nans, validator_not_all_zero
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

sts_features_preoperative = {
    "age",
    "carshock",
    "chf",
    "chrlungd",
    "classnyh",
    "creatlst",
    "cva",
    "cvd",
    "cvdpcarsurg",
    "cvdtia",
    "diabetes",
    "dialysis",
    "ethnicity",
    "gender",
    "hct",
    "hdef",
    "heightcm",
    "hypertn",
    "immsupp",
    "incidenc",
    "infendty",
    "medadp5days",
    "medgp",
    "medinotr",
    "medster",
    "numdisv",
    "platelets",
    "pocpci",
    "pocpciin",
    "prcab",
    "prcvint",
    "prvalve",
    "pvd",
    "raceasian",
    "raceblack",
    "racecaucasian",
    "racenativeam",
    "raceothernativepacific",
    "resusc",
    "status",
    "vdinsufa",
    "vdinsufm",
    "vdinsuft",
    "vdstena",
    "vdstenm",
    "wbc",
    "weightkg",
}

sts_features_categorical = {
    "classnyh": [1, 2, 3, 4],
    "incidenc": [1, 2, 3],
    "numdisv": [0, 1, 2, 3],
    "infendty": [1, 2, 3],
    "status": [1, 2, 3, 4],
}

# Define the name, median, and IQR of continuous features to enable standardization
# These values are calculated from the entire STS MGH cohort using a Jupyter Notebook
# fmt: off
sts_features_continuous: Dict[str, Dict[str, Union[int, float]]] = {
    "age":       {"median": 67,    "iqr": 18,    "min": 21,   "max": 110},
    "creatlst":  {"median": 1,     "iqr": 0.36,  "min": 0.17, "max": 17.89},
    "hct":       {"median": 39,    "iqr": 8,     "min": 2.3,  "max": 76.766},
    "hdef":      {"median": 60,    "iqr": 16,    "min": 5,    "max": 102.381},
    "heightcm":  {"median": 173,   "iqr": 15,    "min": 52.7, "max": 208.3},
    "platelets": {"median": 20700, "iqr": 90000, "min": 2000, "max": 777277.179},
    "wbc":       {"median": 7.3,   "iqr": 3,     "min": 0.2,  "max": 99.99},
    "weightkg":  {"median": 82,    "iqr": 24,    "min": 26.4, "max": 204},
    "perfustm":  {"median": 123,   "iqr": 72,    "min": 0,    "max": 960},
    "xclamptm":  {"median": 90,    "iqr": 65,    "min": 0,    "max": 867.168},
    "predmort":  {"median": 0,     "iqr": 1,     "min": 0,    "max": 1},
}
# fmt: on

# Binary features are all pre-op features minus categorical and continuous features,
# plus cabg and valve procedures (binary)
sts_features_binary = (
    set(sts_features_preoperative)
    - set(sts_features_categorical)
    - set(sts_features_continuous)
)

# fmt: off
sts_outcomes = {
    "sts_death":                 "mtopd",
    "sts_stroke":                "cnstrokp",
    "sts_renal_failure":         "crenfail",
    "sts_prolonged_ventilation": "cpvntlng",
    "sts_dsw_infection":         "deepsterninf",
    "sts_reoperation":           "reop",
    "sts_long_stay":             "llos",
}

# Get keys of outcomes in STS features CSV
sts_outcome_keys = [key for outcome, key in sts_outcomes.items()]

sts_operative_types = {
    "opcab":                     "opcab",
    "opvalve":                   "opvalve",
    "opother":                   "opother",
}
# fmt: on


def binarize(
    key: str,
    value: int,
    negative_value: int = 0,
    positive_value: int = 1,
) -> List[int]:
    if value == negative_value:
        return [1, 0]
    if value == positive_value:
        return [0, 1]
    logging.debug(
        f"STS {key} has value that is not {negative_value} or {positive_value}",
    )
    return [0, 0]


def one_hot(channel_map: Dict[str, int], value: int) -> List[int]:
    t = [0] * len(channel_map)
    prefix = next(iter(channel_map)).split("_")[0]
    value_str = f"{prefix}_{value}"
    if value_str in channel_map:
        t[channel_map[value_str]] = 1
    return t


def get_sts_surgery_dates(data: PatientData) -> Dates:
    # STS data is stored in data as a pandas DataFrame
    return data[STS_PREFIX][STS_SURGERY_DATE_COLUMN]


def _make_sts_tff_continuous(key: str) -> Callable:
    def tensor_from_file(tm: TensorMap, data: PatientData) -> np.ndarray:
        surgery_dates = tm.time_series_filter(data)
        tensor = data[STS_PREFIX].loc[surgery_dates.index, key].to_numpy()
        if is_dynamic_shape(tm):
            tensor = tensor[:, None]
        return tensor

    return tensor_from_file


def _make_sts_tff_binary(
    key: str,
    negative_value: int,
    positive_value: int,
) -> Callable:
    def tensor_from_file(tm: TensorMap, data: PatientData) -> np.ndarray:
        surgery_dates = tm.time_series_filter(data)
        tensor = data[STS_PREFIX].loc[surgery_dates.index, key].to_numpy()
        tensor = tensor[:, None]
        tensor = np.apply_along_axis(
            func1d=lambda x: binarize(key, x, negative_value, positive_value),
            axis=1,
            arr=tensor,
        )
        if not is_dynamic_shape(tm):
            tensor = tensor[0]
        return tensor

    return tensor_from_file


def _make_sts_tff_categorical(key: str) -> Callable:
    def tensor_from_file(tm: TensorMap, data: PatientData) -> np.ndarray:
        surgery_dates = tm.time_series_filter(data)
        tensor = data[STS_PREFIX].loc[surgery_dates.index, key].to_numpy()
        tensor = tensor[:, None]
        if tm.channel_map is None:
            raise ValueError(f"{tm.name} channel map is None")
        tensor = np.apply_along_axis(lambda x: one_hot(tm.channel_map, x), 1, tensor)
        if not is_dynamic_shape(tm):
            tensor = tensor[0]
        return tensor

    return tensor_from_file


def _make_sts_categorical_channel_map(feature: str) -> ChannelMap:
    """Create channel map for categorical STS feature;
    e.g. turns {"classnyh": [1, 2, 3, 4]} into
               {"classnyh_1": 0, "classnyh_2": 1, "classnyh_3": 2, "classnyh_4": 3}"""
    values = sts_features_categorical[feature]
    channel_map = dict()
    for idx, value in enumerate(values):
        channel_map[f"{feature}_{value}"] = idx
    return channel_map


# Categorical (non-binary)
for tmap_name in sts_features_categorical:
    tff = _make_sts_tff_categorical(key=tmap_name)
    channel_map = _make_sts_categorical_channel_map(feature=tmap_name)

    tmaps[tmap_name] = TensorMap(
        name=tmap_name,
        interpretation=Interpretation.CATEGORICAL,
        path_prefix=STS_PREFIX,
        tensor_from_file=tff,
        channel_map=channel_map,
        validators=validator_not_all_zero,
        time_series_limit=0,
        time_series_filter=get_sts_surgery_dates,
    )

# Binary
for tmap_name in sts_features_binary:
    tff = _make_sts_tff_binary(
        key=tmap_name,
        negative_value=2,
        positive_value=1,
    )
    channel_map = outcome_channels(tmap_name)

    tmaps[tmap_name] = TensorMap(
        name=tmap_name,
        interpretation=Interpretation.CATEGORICAL,
        path_prefix=STS_PREFIX,
        tensor_from_file=tff,
        channel_map=channel_map,
        validators=validator_not_all_zero,
        time_series_limit=0,
        time_series_filter=get_sts_surgery_dates,
    )

# Continuous
for tmap_name in sts_features_continuous:
    tff = _make_sts_tff_continuous(key=tmap_name)

    # Make tmaps for both raw and scaled data
    for standardize in ["", "_scaled", "_minmaxed"]:
        normalizer = None
        if standardize == "_scaled":
            normalizer = RobustScalePopulation(
                median=sts_features_continuous[tmap_name]["median"],
                iqr=sts_features_continuous[tmap_name]["iqr"],
            )
        elif standardize == "_minmaxed":
            normalizer = MinMax(
                min=sts_features_continuous[tmap_name]["min"],
                max=sts_features_continuous[tmap_name]["max"],
            )
        tmaps[tmap_name + standardize] = TensorMap(
            name=tmap_name + standardize,
            shape=(1,),
            interpretation=Interpretation.CONTINUOUS,
            path_prefix=STS_PREFIX,
            tensor_from_file=tff,
            validators=validator_no_nans,
            normalizers=normalizer,
            time_series_limit=0,
            time_series_filter=get_sts_surgery_dates,
        )

# Outcomes + Operative Types
for tmap_name, key in {**sts_outcomes, **sts_operative_types}.items():
    tff = _make_sts_tff_binary(
        key=key,
        negative_value=0,
        positive_value=1,
    )

    tmaps[tmap_name] = TensorMap(
        name=tmap_name,
        interpretation=Interpretation.CATEGORICAL,
        path_prefix=STS_PREFIX,
        tensor_from_file=tff,
        channel_map=outcome_channels(tmap_name),
        validators=validator_not_all_zero,
        time_series_limit=0,
        time_series_filter=get_sts_surgery_dates,
    )


# Composite tensor map for any outcome
def tff_any(tm: TensorMap, data: PatientData) -> np.ndarray:
    surgery_dates = tm.time_series_filter(data)
    tensor = data[STS_PREFIX].loc[surgery_dates.index, sts_outcome_keys].to_numpy()
    tensor = tensor.any(axis=1).astype(int)
    tensor = tensor[:, None]
    tensor = np.apply_along_axis(lambda x: binarize("any", x), 1, tensor)
    if not is_dynamic_shape(tm):
        tensor = tensor[0]
    return tensor


tmap_name = "sts_any"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=STS_PREFIX,
    tensor_from_file=tff_any,
    channel_map=outcome_channels(tmap_name),
    validators=validator_not_all_zero,
    time_series_limit=0,
    time_series_filter=get_sts_surgery_dates,
)
