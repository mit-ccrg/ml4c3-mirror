# Imports: standard library
import re
from typing import Dict, Union, Callable
from datetime import timedelta

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from definitions.ecg import ECG_PREFIX
from definitions.c3po import C3PO_PREFIX, C3PO_DEATH_DATE_COLUMN
from ml4c3.normalizer import RobustScalePopulation
from ml4c3.validators import RangeValidator, validator_no_nans, validator_not_all_zero
from tensormap.TensorMap import TensorMap, PatientData, Interpretation

tmaps: Dict[str, TensorMap] = {}


def get_c3po_date_of_death(data: PatientData) -> pd.Timestamp:
    return pd.to_datetime(data[C3PO_PREFIX][C3PO_DEATH_DATE_COLUMN].iloc[0])


def make_tensor_from_file_c3po_death(years: int):
    def tensor_from_file_death(tm: TensorMap, data: PatientData) -> np.ndarray:
        """
        Checks date of death for a patient, and compares against dates of all ECGs.
        If the ECG is within the specified number of years of the date of death,
        the ECG is labeled as an instance of the positive class.
        """
        death_date = get_c3po_date_of_death(data=data)

        # Get list of ECG dates as datetime objects
        ecg_dates = [pd.to_datetime(dt) for dt in data[ECG_PREFIX].keys()]
        shape = (len(ecg_dates), len(tm.channel_map))
        tensor = np.zeros(shape, dtype=np.float32)

        # If the death date is NaN, then the patient did not die;
        # each ECG should be labelled appropriately
        if pd.isnull(death_date):
            tensor[:, tm.channel_map["no_death"]] = 1
            return tensor

        if ecg_dates[-1] > death_date:
            return
        else:
            earliest_ecg_date = death_date + timedelta(days=-years * 365)
            for idx, ecg_date in enumerate(ecg_dates):
                if ecg_date > earliest_ecg_date:
                    tensor[idx, tm.channel_map["death"]] = 1
                else:
                    tensor[idx, tm.channel_map["no_death"]] = 1
        return tensor

    return tensor_from_file_death


def make_c3po_death_tmap(
    tmap_name: str,
    tmaps: Dict[str, TensorMap],
) -> Dict[str, TensorMap]:
    pattern = "c3po_death_(\d+)_years_pre_ecg"
    years = re.match(pattern, tmap_name)
    if years is None:
        return tmaps

    tmaps[tmap_name] = TensorMap(
        name=tmap_name,
        channel_map={"no_death": 0, "death": 1},
        interpretation=Interpretation.CATEGORICAL,
        tensor_from_file=make_tensor_from_file_c3po_death(years=int(years[1])),
        validators=validator_not_all_zero,
        time_series_limit=0,
        path_prefix=C3PO_PREFIX,
    )
    return tmaps
