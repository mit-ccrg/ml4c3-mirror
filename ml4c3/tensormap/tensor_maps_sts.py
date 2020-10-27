# Imports: standard library
import logging
import datetime
from typing import Dict, Callable

# Imports: third party
import h5py
import numpy as np

# Imports: first party
from ml4c3.normalizer import MinMax, RobustScaler
from ml4c3.validators import validator_no_nans, validator_not_all_zero
from ml4c3.definitions.sts import STS_PREFIX, STS_DATE_FORMAT
from ml4c3.definitions.types import ChannelMap
from ml4c3.tensormap.TensorMap import (
    TensorMap,
    Interpretation,
    make_hd5_path,
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
sts_features_continuous = {
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
sts_outcomes_raw_keys = [key for outcome, key in sts_outcomes.items()]

sts_operative_types = {
    "opcab":                     "opcab",
    "opvalve":                   "opvalve",
    "opother":                   "opother",
}
# fmt: on


def _make_sts_tff_continuous(key: str) -> Callable:
    def tensor_from_file(tm: TensorMap, hd5: h5py.File) -> np.ndarray:
        surgery_dates = tm.time_series_filter(hd5)
        dynamic, shape = is_dynamic_shape(tm, len(surgery_dates))
        tensor = np.zeros(shape, dtype=np.float32)
        for i, surgery_date in enumerate(surgery_dates):
            for _, idx in tm.channel_map.items():
                try:
                    path = make_hd5_path(tm, surgery_date, key)
                    feature_value = hd5[path][()]
                    slices = (i, idx) if dynamic else (idx,)
                    tensor[slices] = feature_value
                except (KeyError, ValueError):
                    logging.debug(f"Could not get STS {key} for hd5 {hd5.filename}")
        return tensor

    return tensor_from_file


def _make_sts_tff_binary(
    key: str,
    negative_value: int = 2,
    positive_value: int = 1,
) -> Callable:
    def tensor_from_file(tm: TensorMap, hd5: h5py.File) -> np.ndarray:
        """
        Parses MRN from the HD5 file name, look up the feature in a dict, and return
        the feature. Note the default encoding of +/- features in STS is 2/1,
        e.g. yes == 1, no == 2, but outcomes are encoded using the usual format of 0/1.
        """
        surgery_dates = tm.time_series_filter(hd5)
        dynamic, shape = is_dynamic_shape(tm, len(surgery_dates))
        tensor = np.zeros(shape, dtype=np.float32)
        for i, surgery_date in enumerate(surgery_dates):
            try:
                path = make_hd5_path(tm, surgery_date, key)
                feature_value = hd5[path][()]
                if feature_value == positive_value:
                    idx = 1
                elif feature_value == negative_value:
                    idx = 0
                else:
                    raise ValueError(
                        f"TMap {tm.name} has value {feature_value} that is not a "
                        f"positive value ({positive_value}), or negative value "
                        f"({negative_value})",
                    )
                slices = (i, idx) if dynamic else (idx,)
                tensor[slices] = 1
            except (KeyError, ValueError):
                logging.debug(f"Could not get STS {key} for hd5 {hd5.filename}")
        return tensor

    return tensor_from_file


def _make_sts_tff_categorical(key: str) -> Callable:
    def tensor_from_file(tm: TensorMap, hd5: h5py.File) -> np.ndarray:
        surgery_dates = tm.time_series_filter(hd5)
        dynamic, shape = is_dynamic_shape(tm, len(surgery_dates))
        tensor = np.zeros(shape, dtype=np.float32)
        for i, surgery_date in enumerate(surgery_dates):
            try:
                path = make_hd5_path(tm, surgery_date, key)
                feature_value = hd5[path][()]
                for cm in tm.channel_map:
                    original_value = cm.split("_")[1]
                    if str(feature_value) == str(original_value):
                        slices = (
                            (i, tm.channel_map[cm])
                            if dynamic
                            else (tm.channel_map[cm],)
                        )
                        tensor[slices] = 1
                        break
            except (KeyError, ValueError):
                logging.debug(f"Could not get STS {key} for hd5 {hd5.filename}")
        return tensor

    return tensor_from_file


def _make_sts_categorical_channel_map(feature: str) -> ChannelMap:
    """Create channel map for categorical STS feature;
    e.g. turns {"classnyh": [1, 2, 3, 4]} into
               {classnyh_1: 0, classnyh_2: 1, classnyh_3: 2, classnyh_4: 3}"""
    values = sts_features_categorical[feature]
    channel_map = dict()
    for idx, value in enumerate(values):
        channel_map[f"{feature}_{value}"] = idx
    return channel_map


def str2datetime(
    input_date: str,
    date_format: str = STS_DATE_FORMAT,
) -> datetime.datetime:
    return datetime.datetime.strptime(input_date, date_format)


# Get keys of outcomes in STS features CSV
outcome_keys = [key for outcome, key in sts_outcomes.items()]


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
    )

# Continuous
for tmap_name in sts_features_continuous:
    tff = _make_sts_tff_continuous(key=tmap_name)

    # Make tmaps for both raw and scaled data
    for standardize in ["", "_scaled", "_minmaxed"]:
        channel_map = {tmap_name + standardize: 0}
        normalizer = None
        if standardize == "_scaled":
            normalizer = RobustScaler(
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
            interpretation=Interpretation.CONTINUOUS,
            path_prefix=STS_PREFIX,
            tensor_from_file=tff,
            channel_map=channel_map,
            validators=validator_no_nans,
            normalizers=normalizer,
            time_series_limit=0,
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
    )


# Composite tensor map for any outcome
def tff_any(tm: TensorMap, hd5: h5py.File) -> np.ndarray:
    surgery_dates = tm.time_series_filter(hd5)
    dynamic, shape = is_dynamic_shape(tm, len(surgery_dates))
    tensor = np.zeros(shape, dtype=np.float32)
    for i, surgery_date in enumerate(surgery_dates):
        try:
            idx = 0
            for key in sts_outcomes_raw_keys:
                path = make_hd5_path(tm, surgery_date, key)
                feature_value = hd5[path][()]
                if feature_value == 1:
                    idx = 1
                    break
            slices = (i, idx) if dynamic else (idx,)
            tensor[slices] = 1
            break
        except (KeyError, ValueError):
            logging.debug(f"Could not get STS any outcome for hd5 {hd5.filename}")
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
)
