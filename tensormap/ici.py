# Imports: standard library
import re
from typing import Dict, Callable, Iterable

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from definitions.ici import ICI_PREFIX, ICI_CASE_COLUMN, ICI_DATE_COLUMN
from ml4c3.normalizer import (
    ZScore,
    RobustScale,
    ZScorePopulation,
    RobustScalePopulation,
)
from ml4c3.validators import validator_not_all_zero
from tensormap.TensorMap import TensorMap, PatientData, Interpretation

continuous_field_map = {
    "ici_age_zscore_pop": ("Age", ZScorePopulation(mean=64.13, std=13.25)),
    "ici_age_robustscale_pop": ("Age", RobustScalePopulation(median=65, iqr=18)),
    "ici_age_zscore": ("Age", ZScore()),
    "ici_age_robustscale": ("Age", RobustScale()),
    "ici_age": ("Age", None),
    "ici_prior_bmi_zscore_pop": ("bmi_before", ZScorePopulation(mean=27.26, std=5.73)),
    "ici_prior_bmi_robustscale_pop": (
        "bmi_before",
        RobustScalePopulation(median=26.43, iqr=7.14),
    ),
    "ici_prior_bmi_zscore": ("bmi_before", ZScore()),
    "ici_prior_bmi_robustscale": ("bmi_before", RobustScale()),
    "ici_prior_bmi": ("bmi_before", None),
    "ici_prior_bp_systolic_zscore_pop": (
        "bp_syst_before",
        ZScorePopulation(mean=127.58, std=17.76),
    ),
    "ici_prior_bp_systolic_robustscale_pop": (
        "bp_syst_before",
        RobustScalePopulation(median=126, iqr=22),
    ),
    "ici_prior_bp_systolic_zscore": ("bp_syst_before", ZScore()),
    "ici_prior_bp_systolic_robustscale": ("bp_syst_before", RobustScale()),
    "ici_prior_bp_systolic": ("bp_syst_before", None),
}

# Tensor map name -> tuple of data column and channel map
binary_field_map = {
    "ici_cv_event": (
        "event_after_noTIA",
        {"no_cv_event": 0, "cv_event": 1},
    ),
    "ici_sex": (
        "Male",
        {"female": 0, "male": 1},
    ),
    "ici_prior_ckd": (
        "anytpici_ckd",
        {"no_prior_ckd": 0, "prior_ckd": 1},
    ),
    "ici_prior_diabetes": (
        "anytpici_diabetes",
        {"no_prior_diabetes": 0, "prior_diabetes": 1},
    ),
    "ici_prior_hypertension": (
        "anytpici_hypertension",
        {"no_prior_hypertension": 0, "prior_hypertension": 1},
    ),
    "ici_prior_dyslipidemia": (
        "anytpici_dyslipidemia",
        {"no_prior_dyslipidemia": 0, "prior_dyslipidemia": 1},
    ),
    "ici_prior_cv_event": (
        "anyCVprior",
        {"no_prior_cv_event": 0, "prior_cv_event": 1},
    ),
    "ici_prior_radiation": (
        "prior_radiation_once",
        {"no_prior_radiation": 0, "prior_radiation": 1},
    ),
    "ici_case": (
        "case",
        {"control": 0, "case": 1},
    ),
}

categorical_field_map = {
    "ici_cancer_type": (
        "Cancer_Type",
        {
            "breast": 0,
            "brain": 1,
            "gi": 2,
            "headneck": 3,
            "hepatobiliary": 4,
            "hodgkin": 5,
            "lung": 6,
            "lymphoma": 7,
            "melanoma": 8,
            "obgyn": 9,
            "pancreatic": 10,
            "renal": 11,
            "other": 12,
        },
    ),
    "ici_type": (
        "ici_type",
        {
            "none": 0,
            "PD1": 1,
            "PDL1": 2,
            "CTLA4": 3,
            "CTLA4 AND PD1": 4,
            "CTLA4 OR PD1": 5,
        },
    ),
}


def make_get_ici_start_dates(
    case_only: bool = False,
) -> Callable[[PatientData], pd.Timestamp]:
    def get_ici_start_dates(data: PatientData) -> pd.Timestamp:
        ici = data[ICI_PREFIX]
        if case_only:
            ici = ici[ici[ICI_CASE_COLUMN] == 1]
        return ici[ICI_DATE_COLUMN]

    return get_ici_start_dates


def make_continuous_tensor_from_file(
    key: str,
) -> Callable[[TensorMap, PatientData], np.ndarray]:
    def tensor_from_file(tm: TensorMap, data: PatientData) -> np.ndarray:
        ici_dates = tm.time_series_filter(data)
        tensor = data[ICI_PREFIX].loc[ici_dates.index, key].to_numpy()[:, None]
        return tensor

    return tensor_from_file


def make_binary_tensor_from_file(
    key: str,
) -> Callable[[TensorMap, PatientData], np.ndarray]:
    def tensor_from_file(tm: TensorMap, data: PatientData) -> np.ndarray:
        ici_dates = tm.time_series_filter(data)
        positive_values = data[ICI_PREFIX].loc[ici_dates.index, key].to_numpy()
        negative_values = 1 - positive_values
        tensor = np.array([negative_values, positive_values]).T
        return tensor

    return tensor_from_file


def make_categorical_tensor_from_file(
    key: str,
) -> Callable[[TensorMap, PatientData], np.ndarray]:
    def tensor_from_file(tm: TensorMap, data: PatientData) -> np.ndarray:
        ici_dates = tm.time_series_filter(data)
        values = data[ICI_PREFIX].loc[ici_dates.index, key].to_numpy()
        tensor = np.zeros((len(values), len(tm.channel_map)))
        for i, value in enumerate(values):
            tensor[i, tm.channel_map[value]] = 1
        return tensor

    return tensor_from_file


def _make_regex(keys: Iterable[str]) -> str:
    return fr"^({'|'.join(keys)})(_case)?"


def make_ici_tmap(tmap_name: str, tmaps: Dict[str, TensorMap]) -> Dict[str, TensorMap]:
    new_tmap_name = None
    tensor_from_file = None
    time_series_filter = None
    shape = None
    channel_map = None
    validators = None
    normalizers = None
    interpretation = None

    # Try to match defined ICI continuous tensor maps
    pattern = _make_regex(continuous_field_map)
    match = re.match(pattern, tmap_name)
    if match is not None:
        # fmt: off
        # ici_age_zscore_case
        new_tmap_name  = match[0]       # ici_age_zscore_case
        base_tmap_name = match[1]       # ici_age_zscore
        case_only      = bool(match[2]) # _case -> True
        # fmt: on

        shape = (1,)
        interpretation = Interpretation.CONTINUOUS
        key, normalizers = continuous_field_map[base_tmap_name]
        tensor_from_file = make_continuous_tensor_from_file(key=key)
        time_series_filter = make_get_ici_start_dates(case_only=case_only)

    # Try to match ICI binary tensor maps
    pattern = _make_regex(binary_field_map)
    match = re.match(pattern, tmap_name)
    if match is not None:
        new_tmap_name = match[0]
        base_tmap_name = match[1]
        case_only = bool(match[2])

        interpretation = Interpretation.CATEGORICAL
        key, channel_map = binary_field_map[base_tmap_name]
        tensor_from_file = make_binary_tensor_from_file(key=key)
        time_series_filter = make_get_ici_start_dates(case_only=case_only)
        validators = validator_not_all_zero

    # Try to match ICI categorical tensor maps
    pattern = _make_regex(categorical_field_map)
    match = re.match(pattern, tmap_name)
    if match is not None:
        new_tmap_name = match[0]
        base_tmap_name = match[1]
        case_only = bool(match[2])

        interpretation = Interpretation.CATEGORICAL
        key, channel_map = categorical_field_map[base_tmap_name]
        tensor_from_file = make_categorical_tensor_from_file(key=key)
        time_series_filter = make_get_ici_start_dates(case_only=case_only)
        validators = validator_not_all_zero

    if new_tmap_name is not None:
        tmaps[new_tmap_name] = TensorMap(
            name=new_tmap_name,
            shape=shape,
            channel_map=channel_map,
            interpretation=interpretation,
            tensor_from_file=tensor_from_file,
            time_series_filter=time_series_filter,
            time_series_limit=0,
            validators=validators,
            normalizers=normalizers,
            path_prefix=ICI_PREFIX,
        )

    return tmaps
