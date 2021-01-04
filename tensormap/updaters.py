# Imports: standard library
import re
import copy
import datetime
from typing import Dict, List, Tuple, Union, Optional

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from ml4c3.metrics import weighted_crossentropy
from definitions.ecg import ECG_PREFIX
from definitions.ici import ICI_PREFIX, ICI_DATE_COLUMN
from definitions.sts import STS_PREFIX, STS_SURGERY_DATE_COLUMN
from definitions.echo import ECHO_PREFIX, ECHO_DATETIME_COLUMN
from definitions.globals import SECONDS_IN_DAY
from tensormap.TensorMap import (
    Dates,
    TensorMap,
    PatientData,
    Interpretation,
    make_default_time_series_filter,
)


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
        if "_oldest" in tmap_name:
            return _dates[:tsl]
        if "_newest" in tmap_name:
            return _dates[-tsl:]
        raise ValueError(f"Unknown time series ordering: {tmap_name}")

    new_tmap = copy.deepcopy(base_tmap)
    new_tmap_name = f"{base_name}{base_split}"
    new_tmap.name = new_tmap_name
    new_tmap.time_series_limit = time_series_limit
    new_tmap.time_series_filter = updated_time_series_filter
    tmaps[new_tmap_name] = new_tmap
    return tmaps


def _get_dataset_metadata(dataset_name: str) -> Tuple[str, str]:
    if dataset_name == "sts":
        prefix = STS_PREFIX
        datetime_column = STS_SURGERY_DATE_COLUMN
    elif dataset_name == "echo":
        prefix = ECHO_PREFIX
        datetime_column = ECHO_DATETIME_COLUMN
    elif dataset_name == "ecg":
        prefix = ECG_PREFIX
        datetime_column = None
    elif dataset_name == "ici":
        prefix = ICI_PREFIX
        datetime_column = ICI_DATE_COLUMN
    else:
        raise ValueError("{data_descriptor} is not a valid data descriptor")
    return prefix, datetime_column


CROSS_REFERENCE_SOURCES = [ECG_PREFIX, STS_PREFIX, ECHO_PREFIX, ICI_PREFIX]


def _days_between_tensor_from_file(tm: TensorMap, data: PatientData) -> np.ndarray:
    # Time series filter will be updated to return days for this tensor from file
    days = tm.time_series_filter(data)
    return days.to_numpy()[:, None]


def update_tmaps_window(
    tmap_name: str,
    tmaps: Dict[str, TensorMap],
) -> Dict[str, TensorMap]:
    """
    Make new tensor map from base tensor map, making conditional on a date from
    another source of data. This requires a precise format for tensor map name:
        [base_tmap_name]_[N]_days_[pre/post]_[other_data_source]
    e.g.
        ecg_2500_365_days_pre_echo
        ecg_2500_365_days_pre_sts_newest
        av_peak_gradient_30_days_post_echo

    Additionally, a special tensor map can be created to get the days between
    cross referenced events by the following format:
        [source_name]_[N]_days_[pre/post]_[other_data_source]_days_between_matched_events
    e.g.
        ecg_180_days_pre_echo_days_between_matched_events
    """

    pattern = (
        fr"(.*)_(\d+)_days_(pre|post)_({'|'.join(CROSS_REFERENCE_SOURCES)})"
        fr"(_days_between_matched_events)?"
    )
    match = re.match(pattern, tmap_name)
    if match is None:
        return tmaps

    # fmt: off
    # ecg_2500_std_180_days_pre_echo
    source_name = match[1]         # ecg_2500_std
    offset_days = int(match[2])    # 180
    pre_or_post = match[3]         # pre
    reference_name = match[4]      # echo
    days_between = match[5] or ""  # (empty string)
    # fmt: on

    new_name = (
        f"{source_name}_{offset_days}_days_{pre_or_post}_{reference_name}{days_between}"
    )

    # If the tmap should return the number of days between matched events,
    # source_name is the name of a source dataset
    if days_between:
        if source_name not in CROSS_REFERENCE_SOURCES:
            raise ValueError(
                f"Source dataset {source_name} not in known cross reference sources; "
                f"cannot create {new_name}",
            )
        source_prefix, source_dt_col = _get_dataset_metadata(dataset_name=source_name)

        # Setup time series filter, using the default time series filter if the source
        # datetime column is None
        if source_dt_col is not None:
            time_series_filter = lambda data: data[source_prefix][source_dt_col]
        else:
            time_series_filter = make_default_time_series_filter(source_prefix)

        # Create a fake base tmap which will be modified with a time series filter
        # function which returns the number of days between events
        base_tmap = TensorMap(
            name=source_name,
            shape=(1,),
            interpretation=Interpretation.CONTINUOUS,
            path_prefix=source_prefix,
            tensor_from_file=_days_between_tensor_from_file,
            time_series_limit=0,
            time_series_filter=time_series_filter,
        )

    # If not getting days between events, source_name is the name of an underlying tmap
    # to filter and must exist
    elif source_name not in tmaps:
        raise ValueError(
            f"Base tmap {source_name} not in existing tmaps; cannot create {new_name}",
        )

    # If all checks pass, get base_tmap in the case that it is an existing tmap
    else:
        base_tmap = tmaps[source_name]

    # Copy the base_tmap to modify, either a real tmap or the fake one setup to get
    # the days between events
    new_tmap = copy.deepcopy(base_tmap)

    reference_prefix, reference_dt_col = _get_dataset_metadata(
        dataset_name=reference_name,
    )

    # One-to-one matching algorithm maximizes the number of matches by pairing events
    # nearest in time, starting from the most recent event.

    # 1. Sort source dates from newest -> oldest
    # 2. Sort reference dates from newest -> oldest
    # 3. For each reference date, starting from the newest reference date
    #     a. Compute relative time window
    #     b. Take the newest source date in range
    def get_cross_referenced_dates(data: PatientData) -> Dates:
        source_dates = base_tmap.time_series_filter(data)

        # Get dates from reference data
        reference_data = data[reference_prefix]
        if isinstance(reference_data, pd.DataFrame):
            reference_dates = reference_data[reference_dt_col]  # Reference data is CSV
        else:
            reference_dates = list(reference_data)  # Reference data is HD5

        # Convert everything to pd.Series of pd.Timestamp
        source_is_list = isinstance(source_dates, list)
        source_dates = pd.Series(source_dates).sort_values(ascending=False)
        source_dates_dt = pd.to_datetime(source_dates)
        reference_dates = pd.Series(reference_dates).sort_values(ascending=False)
        reference_dates = pd.to_datetime(reference_dates)

        # Set start and end dates relative to an event
        if pre_or_post == "pre":
            start_dates = reference_dates + datetime.timedelta(days=offset_days * -1)
            end_dates = reference_dates
        else:
            start_dates = reference_dates
            end_dates = reference_dates + datetime.timedelta(days=offset_days)

        dates = pd.Series(dtype=object)
        day_differences = pd.Series(dtype=object)
        for start_date, end_date, reference_date in zip(
            start_dates,
            end_dates,
            reference_dates,
        ):
            # Get newest source date in range of start and end dates
            matched_date = source_dates_dt[
                source_dates_dt.between(start_date, end_date, inclusive=False)
            ][:1]

            # If computing the days between events, calculate the day difference between
            # the reference date and the matched date
            if days_between:
                difference = reference_date - matched_date
                difference = difference.dt.total_seconds() / SECONDS_IN_DAY
                day_differences = day_differences.append(difference)

            # If not computing the days between events, return the actual dates
            else:
                # Computation is done on pd.Timestamp objects but returned list should
                # use the original strings/format in source_dates
                dates = dates.append(source_dates[matched_date.index])

            # Remove the matched date from further matching
            source_dates_dt = source_dates_dt.drop(matched_date.index)

        if len(dates) == 0 and len(day_differences) == 0:
            raise ValueError("No cross referenced dates")

        if days_between:
            return day_differences
        elif source_is_list:
            return list(dates)
        else:
            return dates

    new_tmap.time_series_filter = get_cross_referenced_dates
    new_tmap.name = new_name
    tmaps[new_name] = new_tmap
    return tmaps
