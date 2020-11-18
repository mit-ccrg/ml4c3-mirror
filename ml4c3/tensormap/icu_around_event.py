# type: ignore
# Imports: standard library
import re
from typing import Tuple, Optional

# Imports: third party
import numpy as np

# Imports: first party
from ml4c3.tensormap.TensorMap import Axis, TensorMap
from ml4c3.tensormap.icu_signals import get_tmap as get_signal_tmap
from ml4c3.tensormap.icu_ecg_features import get_tmap as get_ecg_tmap
from ml4c3.tensormap.icu_signal_metrics import compute_feature
from ml4c3.tensormap.icu_first_visit_with_signal import get_tmap as get_visit_tmap

# pylint: disable=unused-argument


def make_around_event_tensor_from_file(
    time: int,
    time_2: int,
    period: str,
    feature: str,
    event_tm: TensorMap,
    visit_tm: TensorMap,
    signal_tm: TensorMap,
    signal_time_tm: TensorMap = None,
    imputation_type: str = None,
    **kwargs,
):
    def _tensor_from_file(tm, hd5, **kwargs):
        visit = visit_tm.tensor_from_file(visit_tm, hd5, **kwargs)[0]
        event_time = event_tm.tensor_from_file(event_tm, hd5, visits=visit, **kwargs)
        event_time = event_time[0][0]
        if period == "pre":
            start = event_time - time_2 * 60 * 60
            end = event_time - time * 60 * 60
        else:
            start = event_time + time_2 * 60 * 60
            end = event_time + time * 60 * 60
        if signal_tm.name.endswith("_timeseries"):
            time_tensor = np.array(
                [
                    signal_tm.tensor_from_file(signal_tm, hd5, visits=visit, **kwargs)[
                        0
                    ][1],
                ],
            )
        else:
            time_tensor = signal_time_tm.tensor_from_file(
                signal_time_tm,
                hd5,
                visits=visit,
                **kwargs,
            )
        indices = np.where(np.logical_and(start < time_tensor, end > time_tensor)[0])[0]
        if len(indices) == 0 and not imputation_type:
            raise KeyError(f"Not in all windows {tm.name}.")
        tensor = compute_feature(
            tm=signal_tm,
            hd5=hd5,
            visit=visit,
            indices=indices,
            feature=feature,
            imputation_type=imputation_type,
            period=period,
            **kwargs,
        )
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
                return time + window
            return time - window

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
#  2. Get two np.arrays of signal value/times/timeseries within a time window at
#     two time points from event.
#     pattern: f"{signal}_{time_2}_and_{time}_hrs_{period}_{event_procedure}_{window}"
#              f"_hrs_window"
#     example: pulse_value_3_and_6_hrs_pre_arrest_start_date_1_hrs_window
#              (gets the pulses values between 3-4 hours and 6-7 hours pre arrest)
#  3. Get a feature in the specified window by adding _{feature_name} at the end
#     of the tmap.
#     pattern: f"{signal}_{time_2}_and_{time}_hrs_{period}_{event_procedure}_{window}"
#              f"_hrs_window_{feature_name}"
#     example: pulse_value_3_and_6_hrs_pre_arrest_start_date_1_hr_window_min
#  4. Imput missing values by adding _{imputation_type}_imputation at the end
#     of the tmap name.
#     pattern: f"{signal}_{time_2}_and_{time}_hrs_{period}_{event_procedure}_{window}"
#              f"_hrs_window_{imputation_type}_imputation"
#     example: pulse_value_3_and_6_hrs_pre_arrest_start_date_1_hr_window_mean_imputation
#  5. Compute a feature and imput missing values.
#     pattern: f"{signal}_{time_2}_and_{time}_hrs_{period}_{event_procedure}_{window}"
#              f"_hrs_window_{feature_name}_{imputation_type}_imputation"
#     example: pulse_value_3_and_6_hrs_pre_arrest_start_date_1_hr_window
#              _min_mean_imputation
#


def get_tmap(tmap_name: str) -> Optional[TensorMap]:
    match = None

    time = None
    time_2 = None
    period = None
    window = None
    feature = "raw"
    signal_tm = None
    event_proc_tm = None
    imputation_type = None

    shape: Optional[Tuple[Axis, ...]] = None
    make_tensor_from_file = None
    time_series_limit = None
    tmap_match_name = tmap_name

    pattern = re.compile(r"^(.*)_(mean_imputation)$")
    match = pattern.findall(tmap_match_name)
    if match:
        _, imputation_type = match[0]
        tmap_match_name = tmap_match_name.replace(f"_{imputation_type}", "")
        match = None

    features = "min|max|mean|median|std|first|last|count|mean_slope|mean_crossing_rate"
    pattern = re.compile(fr"^(.*)_({features})$")
    match = pattern.findall(tmap_match_name)
    if match:
        _, feature = match[0]
        tmap_match_name = tmap_match_name.replace(f"_{feature}", "")
        match = None

    if not match:
        pattern = re.compile(
            r"^(.*)_(\d+)_to_(\d+)_hrs_(pre|post)_(.*)$",
        )
        match = pattern.findall(tmap_match_name)
        if match:
            signal_tm, time, time_2, period, event_proc_tm = match[0]
            make_tensor_from_file = make_around_event_tensor_from_file
    if not match:
        pattern = re.compile(
            r"^(.*)_(\d+)_and_(\d+)_hrs_(pre|post)_(.*)_(\d+)_hrs_window$",
        )
        match = pattern.findall(tmap_match_name)
        if match:
            signal_tm, time, time_2, period, event_proc_tm, window = match[0]
            make_tensor_from_file = make_around_event_double_tensor_from_file
            time_series_limit = 2

    if not match:
        return None

    if signal_tm.endswith("_timeseries"):
        if feature == "raw":
            shape = (None, 2)
        else:
            shape = (2,)
    else:
        if feature == "raw":
            shape = (None,)
        else:
            shape = (1,)

    def _get_tmap(_tmap_name):
        _tm = None
        for _get in [
            get_ecg_tmap,
            get_signal_tmap,
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
    event_proc_tm = get_signal_tmap(event_proc_tm)
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
