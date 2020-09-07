# Imports: standard library
import logging
import datetime
from typing import Dict, Union, Callable
from collections import defaultdict

# Imports: third party
import h5py
import numpy as np
import pandas as pd

# Imports: first party
from ml4cvd.TensorMap import (
    TensorMap,
    Interpretation,
    id_from_filename,
    outcome_channels,
)
from ml4cvd.normalizer import RobustScaler
from ml4cvd.validators import validator_no_nans, validator_not_all_zero
from ml4cvd.definitions import (
    ECG_PREFIX,
    STS_DATA_CSV,
    STS_DATE_FORMAT,
    ECG_DATETIME_FORMAT,
    ChannelMap,
    SampleIntervalData,
)
from ml4cvd.tensor_maps_ecg import get_ecg_dates

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
    "age":       {"median": 67, "iqr": 18},
    "creatlst":  {"median": 1, "iqr": 0.36},
    "hct":       {"median": 39, "iqr": 8},
    "hdef":      {"median": 60, "iqr": 16},
    "heightcm":  {"median": 173, "iqr": 15},
    "platelets": {"median": 20700, "iqr": 90000},
    "wbc":       {"median": 7.3, "iqr": 3},
    "weightkg":  {"median": 82, "iqr": 24},
    "perfustm":  {"median": 123, "iqr": 72},
    "xclamptm":  {"median": 90, "iqr": 65},
}
# fmt: on

# Binary features are all pre-op features minus categorical and continuous features,
# plus cabg and valve procedures (binary)
sts_features_binary = (
    set(sts_features_preoperative)
    - set(sts_features_categorical)
    - set(sts_features_continuous)
)
sts_features_binary.add("opcab")
sts_features_binary.add("opvalve")

# fmt: off
sts_outcomes = {
    "sts_death":                 "mtopd",
    "sts_stroke":                "cnstrokp",
    "sts_renal_failure":         "crenfail",
    "sts_prolonged_ventilation": "cpvntlng",
    "sts_dsw_infection":         "deepsterninf",
    "sts_reoperation":           "reop",
    "sts_any_morbidity":         "anymorbidity",
    "sts_long_stay":             "llos",
}
# fmt: on


def _get_sts_data(
    data_file: str = STS_DATA_CSV,
    patient_column: str = "medrecn",
    start_column: str = "surgdt",
    start_offset: int = -30,
    end_column: str = "surgdt",
    end_offset: int = 0,
) -> SampleIntervalData:
    """
    Load and organize STS data from CSV file into a nested dict keyed by MRN -> preop window (tuple) → surgical data
    For example, returns:
    {
        123: {
            (6/15/2008, 7/15/2008): {
                "medrecn": 123,
                "surgdt": 7/15/2008,
                "mtopd": 0,
            },
            (4/3/2019, 5/3/2019): {
                "medrecn": 123,
                "surgdt": 5/3/2019,
                "mtopd": 1,
            },
        },
    }

    :param data_file: path to STS data file
    :param patient_column: name of column in data to patient identifier
    :param start_column: name of column in data to preop window start value
    :param start_offset: offset in days to add to preop window start value
    :param end_column: name of column in data to preop window end value
    :param end_offset: offset in days to add to preop window end value
    :return: dictionary of MRN to dictionary of intervals to dictionary of surgical features
    """
    sts_data = defaultdict(dict)
    df = pd.read_csv(data_file, low_memory=False)
    for surgery_data in df.to_dict(orient="records"):
        mrn = surgery_data[patient_column]
        start = (
            str2datetime(
                input_date=surgery_data[start_column], date_format=STS_DATE_FORMAT,
            )
            + datetime.timedelta(days=start_offset)
        ).strftime(ECG_DATETIME_FORMAT)
        end = (
            str2datetime(
                input_date=surgery_data[end_column], date_format=STS_DATE_FORMAT,
            )
            + datetime.timedelta(days=end_offset)
        ).strftime(ECG_DATETIME_FORMAT)
        sts_data[mrn][(start, end)] = surgery_data
    return sts_data


def _get_sts_data_for_newest_surgery_with_preop_ecg(
    sts_data: SampleIntervalData, tm: TensorMap, hd5: h5py.File,
) -> Dict[str, Union[str, int, float]]:
    """
    Given a patient, get surgical features and outcomes for the newest surgery for which a patient has a preop ECG
    For example, returns:
    {
        "medrecn": 123,
        "surgdt": 5/3/2019,
        "mtopd": 1,
    }

    :param sts_data: dictionary of MRN to dictionary of intervals to dictionary of surgical features
    :param tm: TensorMap with time series selection parameters
    :param hd5: hd5 file containing patient data
    :return: dictionary of surgical features
    """
    mrn = id_from_filename(hd5.filename)
    ecg_dates = get_ecg_dates(tm, hd5)
    ecg_dates.sort()
    preop_windows = list(sts_data[mrn].keys())
    preop_windows.sort(key=lambda x: x[1])
    newest_surgery_data = None
    for ecg_date in ecg_dates:
        for start, end in preop_windows:
            if start < ecg_date < end:
                newest_surgery_data = sts_data[mrn][(start, end)]
    if newest_surgery_data is None:
        raise ValueError(f"No surgery found for patient {mrn} with ECGs {ecg_dates}")
    return newest_surgery_data


def _make_sts_tff_continuous(sts_data: SampleIntervalData, key: str = "") -> Callable:
    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}):
        newest_surgery_data = _get_sts_data_for_newest_surgery_with_preop_ecg(
            sts_data=sts_data, tm=tm, hd5=hd5,
        )
        tensor = np.zeros(tm.shape, dtype=np.float32)
        for feature, idx in tm.channel_map.items():
            try:
                if key == "":
                    feature_value = newest_surgery_data[tm.name]
                else:
                    feature_value = newest_surgery_data[key]
                tensor[idx] = feature_value
            except:
                logging.debug(
                    f"Could not get continuous tensor using TMap {tm.name} from {hd5.filename}",
                )
        return tensor

    return tensor_from_file


def _make_sts_tff_binary(
    sts_data: SampleIntervalData,
    key: str = "",
    negative_value: int = 2,
    positive_value: int = 1,
) -> Callable:
    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}):
        """Parses MRN from the HD5 file name, look up the feature in a dict, and
        return the feature. Note the default encoding of +/- features in STS is 2/1,
        # e.g. yes == 1, no == 2, but outcomes are encoded using the usual format of 0/1.
        """
        newest_surgery_data = _get_sts_data_for_newest_surgery_with_preop_ecg(
            sts_data=sts_data, tm=tm, hd5=hd5,
        )
        tensor = np.zeros(tm.shape, dtype=np.float32)
        if key == "":
            feature_value = newest_surgery_data[tm.name]
        else:
            feature_value = newest_surgery_data[key]
        if feature_value == positive_value:
            idx = 1
        elif feature_value == negative_value:
            idx = 0
        else:
            raise ValueError(
                f"TMap {tm.name} has value {feature_value} that is not a positive value ({positive_value}), or negative value ({negative_value})",
            )
        tensor[idx] = 1

        return tensor

    return tensor_from_file


def _make_sts_tff_categorical(sts_data: SampleIntervalData, key: str = "") -> Callable:
    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}):
        newest_surgery_data = _get_sts_data_for_newest_surgery_with_preop_ecg(
            sts_data=sts_data, tm=tm, hd5=hd5,
        )
        tensor = np.zeros(tm.shape, dtype=np.float32)
        if key == "":
            feature_value = newest_surgery_data[tm.name]
        else:
            feature_value = newest_surgery_data[key]
        for cm in tm.channel_map:
            original_value = cm.split("_")[1]
            if str(feature_value) == str(original_value):
                tensor[tm.channel_map[cm]] = 1
                break
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
    input_date: str, date_format: str = STS_DATE_FORMAT,
) -> datetime.datetime:
    return datetime.datetime.strptime(input_date, date_format)


# Get keys of outcomes in STS features CSV
outcome_keys = [key for outcome, key in sts_outcomes.items()]


# Get STS features from CSV as dict
sts_data = _get_sts_data()
date_interval_lookup = {mrn: list(sts_data[mrn]) for mrn in sts_data}


# Categorical (non-binary)
for tmap_name in sts_features_categorical:
    interpretation = Interpretation.CATEGORICAL
    tff = _make_sts_tff_categorical(sts_data=sts_data)
    channel_map = _make_sts_categorical_channel_map(feature=tmap_name)
    validator = validator_no_nans

    tmaps[tmap_name] = TensorMap(
        name=tmap_name,
        interpretation=interpretation,
        path_prefix=ECG_PREFIX,
        tensor_from_file=tff,
        channel_map=channel_map,
        validator=validator,
        time_series_lookup=date_interval_lookup,
    )

# Binary
for tmap_name in sts_features_binary:
    interpretation = Interpretation.CATEGORICAL
    tff = _make_sts_tff_binary(
        sts_data=sts_data, key=tmap_name, negative_value=2, positive_value=1,
    )
    channel_map = outcome_channels(tmap_name)
    validator = validator_no_nans

    tmaps[tmap_name] = TensorMap(
        name=tmap_name,
        interpretation=interpretation,
        path_prefix=ECG_PREFIX,
        tensor_from_file=tff,
        channel_map=channel_map,
        validator=validator,
        time_series_lookup=date_interval_lookup,
    )

# Continuous
for tmap_name in sts_features_continuous:
    interpretation = Interpretation.CONTINUOUS

    # Note the need to set the key; otherwise, tff will use the tmap name
    # "foo_scaled" for the key, instead of "foo"
    tff = _make_sts_tff_continuous(sts_data, key=tmap_name)
    validator = validator_no_nans

    # Make tmaps for both raw and scaled data
    for standardize in ["", "_scaled"]:
        channel_map = {tmap_name + standardize: 0}
        normalizer = (
            RobustScaler(
                median=sts_features_continuous[tmap_name]["median"],
                iqr=sts_features_continuous[tmap_name]["iqr"],
            )
            if standardize == "_scaled"
            else None
        )
        tmaps[tmap_name + standardize] = TensorMap(
            name=tmap_name + standardize,
            interpretation=interpretation,
            path_prefix=ECG_PREFIX,
            tensor_from_file=tff,
            channel_map=channel_map,
            validator=validator,
            normalization=normalizer,
            time_series_lookup=date_interval_lookup,
        )

# Outcomes
for tmap_name in sts_outcomes:
    interpretation = Interpretation.CATEGORICAL
    tff = _make_sts_tff_binary(
        sts_data=sts_data,
        key=sts_outcomes[tmap_name],
        negative_value=0,
        positive_value=1,
    )
    channel_map = outcome_channels(tmap_name)
    validator = validator_not_all_zero

    tmaps[tmap_name] = TensorMap(
        name=tmap_name,
        interpretation=interpretation,
        path_prefix=ECG_PREFIX,
        tensor_from_file=tff,
        channel_map=channel_map,
        validator=validator,
        time_series_lookup=date_interval_lookup,
    )
