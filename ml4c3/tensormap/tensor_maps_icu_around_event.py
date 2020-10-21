# type: ignore
# Imports: standard library
import re
from typing import Tuple, Optional

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from ml4c3.definitions.icu import ICU_TMAPS_SUMMARY_PATH
from ml4c3.tensormap.TensorMap import Axis, TensorMap
from ml4c3.tensormap.tensor_maps_icu_alarms import get_tmap as get_alarm_tmap
from ml4c3.tensormap.tensor_maps_icu_events import get_tmap as get_event_tmap
from ml4c3.tensormap.tensor_maps_icu_bm_signals import get_tmap as get_bm_tmap
from ml4c3.tensormap.tensor_maps_icu_medications import get_tmap as get_med_tmap
from ml4c3.tensormap.tensor_maps_icu_ecg_features import get_tmap as get_ecg_tmap
from ml4c3.tensormap.tensor_maps_icu_measurements import get_tmap as get_measure_tmap
from ml4c3.tensormap.tensor_maps_icu_signal_metrics import compute_feature
from ml4c3.tensormap.tensor_maps_icu_first_visit_with_signal import (
    get_tmap as get_visit_tmap,
)

# pylint: disable=unused-argument


def get_offset_time(event_time, period, time, time_2):
    if period == "pre":
        start = event_time - time * 60 * 60
        end = event_time - time_2 * 60 * 60
    else:
        start = event_time + time * 60 * 60
        end = event_time + time_2 * 60 * 60
    return start, end


def make_around_event_series_tensor_from_file(
    time: int,
    time_2: int,
    period: str,
    event_tm: TensorMap,
    visit_tm: TensorMap,
    signal_tm: TensorMap,
    **kwargs,
):
    def _tensor_from_file(tm, hd5, **kwargs):
        visit = visit_tm.tensor_from_file(visit_tm, hd5, **kwargs)[0]
        event_time = event_tm.tensor_from_file(event_tm, hd5, visits=visit, **kwargs)
        event_time = event_time[0][0]
        start, end = get_offset_time(event_time, period, time, time_2)
        _tensor = signal_tm.tensor_from_file(signal_tm, hd5, visits=visit, **kwargs)
        for _visit_tensor in _tensor:
            indices = np.where(
                np.logical_and(start < _visit_tensor[1], end > _visit_tensor[1]),
            )[0]
            tensor = _visit_tensor[:, indices]
            if len(tensor) > 0:
                return tensor
        raise ValueError(f"{tm.name}: could not find any values in time window")

    return _tensor_from_file


def make_around_event_tensor_from_file(
    time: int,
    time_2: int,
    period: str,
    feature: str,
    event_tm: TensorMap,
    visit_tm: TensorMap,
    signal_tm: TensorMap,
    signal_time_tm: TensorMap,
    imputation_type: str = None,
    **kwargs,
):
    metadata = pd.read_csv(ICU_TMAPS_SUMMARY_PATH, index_col=0)

    def _tensor_from_file(tm, hd5, **kwargs):
        visit = visit_tm.tensor_from_file(visit_tm, hd5, **kwargs)[0]
        event_time = event_tm.tensor_from_file(event_tm, hd5, visits=visit, **kwargs)
        event_time = event_time[0][0]
        start, end = get_offset_time(event_time, period, time, time_2)
        time_tensor = signal_time_tm.tensor_from_file(
            signal_time_tm, hd5, visits=visit, **kwargs,
        )
        indices = np.where(np.logical_and(start < time_tensor, end > time_tensor)[0])[0]
        if imputation_type:
            name = tm.name.replace(f"_{imputation_type}", "").replace("_imputation", "")
            imputation = metadata.loc[name][imputation_type]
        if len(indices) == 0:
            if imputation_type:
                return np.array([imputation])
            raise KeyError(f"Not in all windows {tm.name}.")
        tensor = compute_feature(signal_tm, hd5, visit, indices, feature, **kwargs)
        if imputation_type:
            tensor = np.nan_to_num(tensor, nan=imputation)
        elif str(np.min(tensor)) == "nan":
            raise KeyError(f"Not in all windows {tm.name}.")
        return tensor

    return _tensor_from_file


def make_around_event_double_tensor_from_file(
    time: int,
    time_2: int,
    window: int,
    period: str,
    feature: str,
    event_tm: TensorMap,
    visit_tm: TensorMap,
    signal_tm: TensorMap,
    signal_time_tm: TensorMap,
    imputation_type=None,
):
    def _tensor_from_file(tm, hd5, **kwargs):
        def offset_window(time, window, period):
            if period == "pre":
                return time - window
            return time + window

        first_window_tensor = make_around_event_tensor_from_file(
            time=time,
            time_2=offset_window(time, window, period),
            period=period,
            feature=feature,
            event_tm=event_tm,
            visit_tm=visit_tm,
            signal_tm=signal_tm,
            signal_time_tm=signal_time_tm,
            imputation_type=imputation_type,
        )
        second_window_tensor = make_around_event_tensor_from_file(
            time=time_2,
            time_2=offset_window(time_2, window, period),
            period=period,
            feature=feature,
            event_tm=event_tm,
            visit_tm=visit_tm,
            signal_tm=signal_tm,
            signal_time_tm=signal_time_tm,
            imputation_type=imputation_type,
        )
        first_window = first_window_tensor(tm, hd5, **kwargs)
        second_window = second_window_tensor(tm, hd5, **kwargs)
        tensor = np.array([first_window, second_window])
        return tensor

    return _tensor_from_file


# Creates tensor maps which get signals relative to some event or procedure.
# Possible tensor maps and their naming patterns are:
#  1. Get all signal values/times/timeseries between two time points pre/post an event.
#     pattern: f"{signal}_timeseries_{time_2}_to_{time}_hrs_{period}_{event_procedure}"
#     example: pulse_timeseries_3_to_6_hrs_pre_arrest_start_date
#  2. Get min/max/mean/median/last value of signal between two points relative an event.
#     pattern: f"{signal}_{time_2}_to_{time}_hrs_{period}_{event_procedure}_{feature}"
#     example: pulse_value_3_to_6_hrs_pre_arrest_start_date_last
#  3. Get aggregate value of signal within a time window at two time points from event.
#     pattern: f"{signal}_{time_2}_and_{time}_hrs_{period}_{event_procedure}_{window}"
#              f"_hrs_window_{feature}"
#     example: pulse_value_3_and_6_hrs_pre_arrest_start_date_1_hr_window_mean
#              (gets the mean pulse at 3-4 hours and 6-7 hours pre arrest)
#  4. Get min/max/mean/median/last value of signal between two points relative an event,
#     if there is no signal between the two points, or the values is a nan,
#     it will return the mean value given in ICU_TMAPS_SUMMARY_PATH.
#     pattern: f"{signal}_{time_2}_to_{time}_hrs_{period}_{event_procedure}_{feature}"
#              f"_mean_imputation"
#     example: pulse_value_3_to_6_hrs_pre_arrest_start_date_last_mean_imputation
#  5. Get aggregate value of signal within a time window at two time points from event,
#     if there is no value, or the value is a nan, it will return the mean value
#     given in ICU_TMAPS_SUMMARY_PATH.
#     pattern: f"{signal}_{time_2}_and_{time}_hrs_{period}_{event_procedure}_{window}"
#              f"_hrs_window_{feature}_mean_imputation"
#     example: pulse_value_3_and_6_hrs_pre_arrest_start_date_1_hr_window_mean
#              _mean_imputation
#


def get_tmap(tmap_name: str) -> Optional[TensorMap]:
    match = None

    time = None
    time_2 = None
    period = None
    window = None
    feature = None
    signal_tm = None
    event_proc_tm = None
    imputation_type = None

    shape: Optional[Tuple[Axis, ...]] = None
    make_tensor_from_file = None
    time_series_limit = None

    if not match:
        pattern = re.compile(r"^(.*)_timeseries_(\d+)_to_(\d+)_hrs_(pre|post)_(.*)$")
        match = pattern.findall(tmap_name)
        if match:
            signal, time, time_2, period, event_proc_tm = match[0]
            signal_tm = f"{signal}_timeseries"
            shape = (None,)
            make_tensor_from_file = make_around_event_series_tensor_from_file

    if not match:
        pattern = re.compile(
            r"^(.*)_(\d+)_to_(\d+)_hrs_(pre|post)_(.*)"
            r"_(min|max|mean|median|std|first|last)$",
        )
        match = pattern.findall(tmap_name)
        if match:
            signal_tm, time, time_2, period, event_proc_tm, feature = match[0]
            shape = (1,)
            make_tensor_from_file = make_around_event_tensor_from_file

    if not match:
        pattern = re.compile(
            r"^(.*)_(\d+)_to_(\d+)_hrs_(pre|post)_(.*)"
            r"_(min|max|mean|median|last)_mean_imputation$",
        )
        match = pattern.findall(tmap_name)
        if match:
            signal_tm, time, time_2, period, event_proc_tm, feature = match[0]
            shape = (1,)
            make_tensor_from_file = make_around_event_tensor_from_file
            imputation_type = "mean"

    if not match:
        pattern = re.compile(
            r"^(.*)_(\d+)_and_(\d+)_hrs_(pre|post)_(.*)_(\d+)"
            r"_hr_window_(min|max|mean|median|last)$",
        )
        match = pattern.findall(tmap_name)
        if match:
            signal_tm, time, time_2, period, event_proc_tm, window, feature = match[0]
            shape = (1,)
            make_tensor_from_file = make_around_event_double_tensor_from_file
            time_series_limit = 2

    if not match:
        pattern = re.compile(
            r"^(.*)_(\d+)_and_(\d+)_hrs_(pre|post)_(.*)_(\d+)"
            r"_hr_window_(min|max|mean|median|last)_mean_imputation$",
        )
        match = pattern.findall(tmap_name)
        if match:
            signal_tm, time, time_2, period, event_proc_tm, window, feature = match[0]
            shape = (1,)
            make_tensor_from_file = make_around_event_double_tensor_from_file
            time_series_limit = 2
            imputation_type = "mean"

    if not match:
        return None

    def _get_tmap(_tmap_name):
        _tm = None
        for _get in [
            get_bm_tmap,
            get_med_tmap,
            get_ecg_tmap,
            get_alarm_tmap,
            get_measure_tmap,
        ]:
            _tm = _get(_tmap_name)
            if _tm is not None:
                break
        return _tm

    time_tm = None
    if not signal_tm.endswith("_timeseries"):
        time_tm = signal_tm.replace("_scaled", "").replace("_mean_imputation", "")
        if time_tm.endswith("_value"):
            time_tm = time_tm.replace("_value", "_time")
        elif time_tm.endswith("_dose"):
            time_tm = time_tm.replace("_dose", "_time")
        elif time_tm.endswith("_duration", "_init_date"):
            time_tm = time_tm.replace("_duration", "_init_date")
        time_tm = _get_tmap(time_tm)

    visit_tm = get_visit_tmap(
        re.sub(r"(end_date|start_date)", "first_visit", event_proc_tm),
    )

    time = int(time)
    time_2 = int(time_2)
    signal_tm = _get_tmap(signal_tm)
    event_proc_tm = get_event_tmap(event_proc_tm)
    if window is not None:
        window = int(window)

    return TensorMap(
        name=tmap_name,
        shape=shape,
        time_series_limit=time_series_limit,
        tensor_from_file=make_tensor_from_file(
            time=time,
            time_2=time_2,
            period=period,
            window=window,
            feature=feature,
            event_tm=event_proc_tm,
            visit_tm=visit_tm,
            signal_tm=signal_tm,
            signal_time_tm=time_tm,
            imputation_type=imputation_type,
        ),
        channel_map=signal_tm.channel_map,
        path_prefix=signal_tm.path_prefix,
        interpretation=signal_tm.interpretation,
    )
