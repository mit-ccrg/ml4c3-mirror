# Imports: standard library
import re
import copy
import logging
import datetime
from typing import Dict, List, Union, Callable, Optional

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from ml4c3.metrics import weighted_crossentropy
from ml4c3.definitions.ecg import ECG_PREFIX, ECG_DATETIME_FORMAT
from ml4c3.definitions.sts import STS_PREFIX, STS_DATE_FORMAT, STS_SURGERY_DATE_COLUMN
from ml4c3.definitions.echo import (
    ECHO_PREFIX,
    ECHO_MRN_COLUMN,
    ECHO_DATETIME_COLUMN,
    ECHO_DATETIME_FORMAT,
)
from ml4c3.tensormap.TensorMap import Dates, TensorMap, PatientData


def update_tmaps_weighted_loss(
    tmap_name: str,
    tmaps: Dict[str, TensorMap],
) -> Dict[str, TensorMap]:
    """Make new tmap from base name, modifying loss weight"""
    if "_weighted_loss_" not in tmap_name:
        return tmaps
    base_name, weight = tmap_name.split("_weighted_loss_")
    if base_name not in tmaps:
        raise ValueError(
            f"Base tmap {base_name} not in existing tmaps. "
            f"Cannot modify weighted loss.",
        )
    weight = weight.split("_")[0]
    tmap = copy.deepcopy(tmaps[base_name])
    new_tmap_name = f"{base_name}_weighted_loss_{weight}"
    tmap.name = new_tmap_name
    tmap.loss = weighted_crossentropy([1.0, float(weight)], new_tmap_name)
    tmaps[new_tmap_name] = tmap
    return tmaps


random_date_selections: Dict[str, Union[List[str], pd.Series]] = dict()


def update_tmaps_time_series(
    tmap_name: str,
    tmaps: Dict[str, TensorMap],
    time_series_limit: Optional[int] = None,
) -> Dict[str, TensorMap]:
    """Given the name of a needed tensor maps, e.g. ["ecg_age_newest"], and its base
    TMap, e.g. tmaps["ecg_age"], this function creates new tmap with the name of the
    needed tmap and the correct shape, but otherwise inherits properties from the base
    tmap. Next, updates new tmap to tmaps dict.
    """
    if "_newest" in tmap_name:
        base_split = "_newest"
    elif "_oldest" in tmap_name:
        base_split = "_oldest"
    elif "_random" in tmap_name:
        base_split = "_random"
    else:
        return tmaps
    base_name, _ = tmap_name.split(base_split)

    if base_name not in tmaps:
        raise ValueError(
            f"Base tmap {base_name} not in existing tmaps. Cannot modify time series.",
        )
    base_tmap = tmaps[base_name]

    def updated_time_series_filter(data: PatientData) -> Dates:
        _dates = base_tmap.time_series_filter(data)
        _dates = (
            _dates.sort_values() if isinstance(_dates, pd.Series) else sorted(_dates)
        )
        tsl = 1 if time_series_limit is None else time_series_limit
        if "_random" in tmap_name:
            if data.id in random_date_selections:
                return random_date_selections[data.id]
            if len(_dates) < tsl:
                tsl = len(_dates)
            _dates = (
                _dates.sample(tsl, replace=False)
                if isinstance(_dates, pd.Series)
                else np.random.choice(_dates, tsl, replace=False)
            )
            random_date_selections[data.id] = _dates
            return _dates
        elif "_oldest" in tmap_name:
            return _dates[:tsl]
        elif "_newest" in tmap_name:
            return _dates[-tsl:]
        else:
            raise ValueError(f"Unknown time series ordering: {tmap_name}")

    new_tmap = copy.deepcopy(base_tmap)
    new_tmap_name = f"{base_name}{base_split}"
    new_tmap.name = new_tmap_name
    new_tmap.time_series_limit = time_series_limit
    new_tmap.time_series_filter = updated_time_series_filter
    tmaps[new_tmap_name] = new_tmap
    return tmaps


def _offset_date(date: str, offset_days: int, date_format: str) -> str:
    return (
        datetime.datetime.strptime(date, date_format)
        + datetime.timedelta(days=offset_days)
    ).strftime(date_format)


def _get_dataset_specific_metadata(data_descriptor: str) -> tuple:
    if data_descriptor == "sts":
        data_prefix = STS_PREFIX
        datetime_column = STS_SURGERY_DATE_COLUMN
        datetime_format = STS_DATE_FORMAT
    elif data_descriptor == "echo":
        data_prefix = ECHO_PREFIX
        datetime_column = ECHO_DATETIME_COLUMN
        datetime_format = ECHO_DATETIME_FORMAT
    elif data_descriptor == "ecg":
        data_prefix = ECG_PREFIX
        datetime_column = None
        datetime_format = ECG_DATETIME_FORMAT
    else:
        raise ValueError("{data_descriptor} is not a valid data descriptor")
    return data_prefix, datetime_column, datetime_format


def update_tmaps_window(
    tmap_name: str,
    tmaps: Dict[str, TensorMap],
) -> Dict[str, TensorMap]:
    """Make new tensor map from base tensor map, making conditional on a date from
    another source of data. This requires a precis format for tensor map name:
        [base_tmap_name]_[N]_days_[pre/post]_[other_data_source]
    e.g.
        ecg_2500_365_days_pre_echo
        ecg_2500_365_days_pre_sts_newest
        av_peak_gradient_30_days_post_echo
    """

    # Given a tmap name such as "ecg_2500_30_days_pre_echo", obtain:
    # offset in days, e.g. "30"
    time_suffix = "_days"
    offset_days = re.findall(fr"_(\d+){time_suffix}", tmap_name)
    if len(offset_days) == 0:
        return tmaps
    else:
        offset_days = offset_days[0]

    # time window: pre vs post
    if "_pre_" in tmap_name:
        pre_or_post = "pre"
    elif "_post_" in tmap_name:
        pre_or_post = "post"
    else:
        logging.warning(f"Cannot identify pre or post in {tmap_name}")
        return tmaps

    # dataset descriptor;
    # given "ecg_2500_30_days_pre_echo", isolate "echo"
    # given "ecg_2500_30_days_pre_echo_newest", isolate "echo_newest"
    data_descriptor_prefix = f"_{pre_or_post}"
    text_after_descriptor_prefix = tmap_name.split(f"_{pre_or_post}_")[1]

    # If "echo_newest", isolate "echo"
    data_descriptor = text_after_descriptor_prefix.split("_")[0]

    offset_days = int(offset_days)
    base_name = re.findall(fr"(.*)_{offset_days}", tmap_name)[0]

    if base_name not in tmaps:
        raise ValueError(
            f"Base tmap {base_name} not in existing tmaps; cannot create {tmap_name}",
        )

    base_tmap = tmaps[base_name]
    new_tmap = copy.deepcopy(base_tmap)
    new_tmap_name = f"{base_name}_{offset_days}_days_{pre_or_post}_{data_descriptor}"
    new_tmap.name = new_tmap_name

    def get_cross_referenced_dates(data: PatientData) -> Dates:
        original_dates = base_tmap.time_series_filter(data)
        (
            data_prefix,
            datetime_column,
            datetime_format,
        ) = _get_dataset_specific_metadata(data_descriptor=data_descriptor)

        # If data for this prefix holds a dataframe
        if isinstance(data[data_prefix], pd.DataFrame):
            xref_dates = data[data_prefix][datetime_column]

        # If data is an HD5 group (e.g. if we are working with ECGs)
        else:
            xref_dates = list(data[data_prefix])

        dates = pd.Series(dtype=str) if isinstance(original_dates, pd.Series) else []

        for xref_date in xref_dates:

            if pd.isnull(xref_date):
                continue

            # If time window is "pre"
            if pre_or_post == "pre":
                start_date = _offset_date(
                    date=xref_date,
                    offset_days=offset_days * -1,
                    date_format=datetime_format,
                )
                end_date = xref_date
            # If time window is "post"
            else:
                start_date = xref_date
                end_date = _offset_date(
                    date=xref_date,
                    offset_days=offset_days,
                    date_format=datetime_format,
                )

            if isinstance(original_dates, pd.Series):
                dates = dates.append(
                    original_dates[
                        original_dates.between(start_date, end_date, inclusive=False)
                    ],
                )
            else:
                for date in original_dates:
                    if start_date < date < end_date:
                        dates.append(date)
        return dates

    new_tmap.time_series_filter = get_cross_referenced_dates
    tmaps[tmap_name] = new_tmap
    return tmaps
