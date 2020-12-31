# type: ignore
# Imports: standard library
import re
from typing import Tuple, Optional

# Imports: third party
import numpy as np

# Imports: first party
from ml4c3.utils import get_unix_timestamps
from ml4c3.validators import validator_no_nans
from tensormap.TensorMap import Axis, TensorMap, Interpretation
from tensormap.icu_signals import get_tmap as get_signal_tmap
from tensormap.icu_signals import admin_age_tensor_from_file
from tensormap.icu_ecg_features import get_tmap as get_ecg_tmap
from tensormap.icu_signal_metrics import compute_feature
from tensormap.icu_first_visit_with_signal import get_tmap as get_visit_tmap

# pylint: disable=unused-argument

FEATURES = "min|max|mean|median|std|first|last|count|mean_slope|mean_crossing_rate"


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


def make_around_event_tensor_from_file(
    time: int,
    window: int,
    period: str,
    feature: str,
    event_tm: TensorMap,
    visit_tm: TensorMap,
    signal_tm: TensorMap,
    signal_time_tm: TensorMap = None,
    imputation_type: str = None,
    flag_time_unix: bool = False,
    **kwargs,
):
    def _tensor_from_file(tm, hd5, **kwargs):
        visit = visit_tm.tensor_from_file(visit_tm, hd5, **kwargs)[0]
        if flag_time_unix:
            start = time
            end = time + window * 60 * 60
        else:
            event_time = event_tm.tensor_from_file(
                event_tm, hd5, visits=visit, unix_dates=True, **kwargs
            )
            event_time = event_time[0][0]
            if period == "pre":
                start = event_time - (time + window) * 60 * 60
                end = event_time - time * 60 * 60
            else:
                start = event_time + time * 60 * 60
                end = event_time + (time + window) * 60 * 60
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

        tensor = compute_feature(
            tm=signal_tm,
            hd5=hd5,
            visit=visit,
            indices=indices,
            feature=feature,
            period=period,
            imputation_type=imputation_type,
            **kwargs,
        )
        return tensor

    return _tensor_from_file


def make_around_event_double_tensor_from_file(
    time: int,
    time_2: int,
    window: int,
    period: str,
    period_2: str,
    feature: str,
    event_tm: TensorMap,
    event_tm_2: TensorMap,
    visit_tm: TensorMap,
    signal_tm: TensorMap,
    signal_time_tm: TensorMap,
    imputation_type: str = None,
):
    def _tensor_from_file(tm, hd5, **kwargs):

        first_window_tensor = make_around_event_tensor_from_file(
            time=time,
            window=window,
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
            window=window,
            period=period_2,
            feature=feature,
            event_tm=event_tm_2,
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


def create_around_tmap(tmap_name: str) -> Optional[TensorMap]:
    match = None

    time = None
    time_2 = None
    period = None
    period_2 = None
    window = None
    feature = "raw"
    signal_tm = None
    event_proc_tm = None
    event_proc_tm_2 = None
    imputation_type = None

    shape: Optional[Tuple[Axis, ...]] = None
    make_tensor_from_file = None
    time_series_limit = None
    tmap_match_name = tmap_name

    if tmap_name.endswith("_explore"):
        return None

    pattern = re.compile(r"^(.*)_(mean_imputation|sample_and_hold)$")
    match = pattern.findall(tmap_match_name)
    if match:
        _, imputation_type = match[0]
        tmap_match_name = tmap_match_name.replace(f"_{imputation_type}", "")
        match = None

    pattern = re.compile(fr"^(.*)_({FEATURES})$")
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
            window = abs(int(time) - int(time_2))
    if not match:
        pattern = re.compile(
            r"^(.*)_(\d+)_and_(\d+)_hrs_(pre|post)_(.*)_(\d+)_hrs_window$",
        )
        match = pattern.findall(tmap_match_name)
        if match:
            signal_tm, time, time_2, period, event_proc_tm, window = match[0]
            make_tensor_from_file = make_around_event_double_tensor_from_file
            time_series_limit = 2
            period_2 = period
            event_proc_tm_2 = event_proc_tm
    if not match:
        pattern = re.compile(
            r"^(.*)_(\d+)_hrs_(pre|post)_(.*)_(\d+)_hrs_(pre|post)_(.*)"
            r"_(\d+)_hrs_window$",
        )
        match = pattern.findall(tmap_match_name)
        if match:
            (
                signal_tm,
                time,
                period,
                event_proc_tm,
                time_2,
                period_2,
                event_proc_tm_2,
                window,
            ) = match[0]
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
    window = int(window)
    signal_tm = _get_tmap(signal_tm)
    event_proc_tm = get_signal_tmap(event_proc_tm)
    if event_proc_tm_2 is not None:
        event_proc_tm_2 = get_signal_tmap(event_proc_tm_2)

    return TensorMap(
        name=tmap_name,
        shape=shape,
        time_series_limit=time_series_limit,
        tensor_from_file=make_tensor_from_file(
            time=time,
            time_2=time_2,
            period=period,
            period_2=period_2,
            window=window,
            feature=feature,
            event_tm=event_proc_tm,
            event_tm_2=event_proc_tm_2,
            visit_tm=visit_tm,
            signal_tm=signal_tm,
            signal_time_tm=time_tm,
            imputation_type=imputation_type,
        ),
        channel_map=signal_tm.channel_map,
        path_prefix=signal_tm.path_prefix,
        interpretation=signal_tm.interpretation,
        validators=validator_no_nans,
    )


def make_around_event_explore_tensor_from_file(
    around_tm_name,
):
    def _tensor_from_file(tm, hd5, **kwargs):
        tensor = np.array([])
        for feature in tm.channel_map:
            name = f"{around_tm_name}_{feature}"
            if feature == "mean_slope":
                name = name.replace("_value", "_timeseries")
            around_tm = create_around_tmap(name)
            try:
                value = around_tm.tensor_from_file(around_tm, hd5, **kwargs)
            except:
                value = np.nan
            tensor = np.append(tensor, value)
        return tensor

    return _tensor_from_file


def create_around_explore_tmap(tmap_name: str) -> Optional[TensorMap]:
    match = None
    pattern = re.compile(
        r"^(.*)_(\d+)_to_(\d+)_hrs_(pre|post)_(.*)_explore$",
    )
    match = pattern.findall(tmap_name)
    if match:
        make_tensor_from_file = make_around_event_explore_tensor_from_file(
            tmap_name.replace("_explore", ""),
        )
        channel_map = {
            "min": 0,
            "max": 1,
            "mean": 2,
            "std": 3,
            "first": 4,
            "last": 5,
            "count": 6,
            # "mean_slope": 7,
            # "mean_crossing_rate": 8,
        }
        path_prefix = create_around_tmap(tmap_name.replace("_explore", "")).path_prefix
    if not match:
        return None

    return TensorMap(
        name=tmap_name,
        tensor_from_file=make_tensor_from_file,
        channel_map=channel_map,
        path_prefix=path_prefix,
        interpretation=Interpretation.CONTINUOUS,
    )


def get_sliding_windows(
    hd5,
    window: int,
    step: int,
    event_tm_1: TensorMap,
    event_tm_2: TensorMap,
    visit_tm: TensorMap,
    **kwargs,
):
    if not hasattr(get_sliding_windows, "windows"):
        get_sliding_windows.windows_cache = {}
    if hd5.id in get_sliding_windows.windows_cache:
        return get_sliding_windows.windows_cache[hd5.id]
    visit = visit_tm.tensor_from_file(visit_tm, hd5, **kwargs)[0]
    event_time_1 = event_tm_1.tensor_from_file(
        event_tm_1, hd5, visits=visit, unix_dates=True, **kwargs
    )
    event_time_1 = event_time_1[0][0]
    event_time_2 = event_tm_2.tensor_from_file(event_tm_2, hd5, visits=visit, **kwargs)
    event_time_2 = event_time_2[0][0]
    windows = np.arange(
        event_time_2 - window * 60 * 60,
        event_time_1,
        -step * 60 * 60,
    )[::-1]
    get_sliding_windows.windows_cache[hd5.id] = windows
    return windows


def make_sliding_window_tensor_from_file(
    window: int,
    step: int,
    event_tm_1: TensorMap,
    event_tm_2: TensorMap,
    visit_tm: TensorMap,
    signal_tm: TensorMap,
    signal_time_tm: TensorMap,
    feature: str,
    imputation_type=None,
):
    def _tensor_from_file(tm, hd5, **kwargs):
        tensor = np.array([])
        windows = get_sliding_windows(
            hd5=hd5,
            window=window,
            step=step,
            event_tm_1=event_tm_1,
            event_tm_2=event_tm_2,
            visit_tm=visit_tm,
        )
        for time in windows:
            window_tensor = make_around_event_tensor_from_file(
                time=time,
                window=window,
                period="pre",
                event_tm=event_tm_2,
                visit_tm=visit_tm,
                signal_tm=signal_tm,
                signal_time_tm=signal_time_tm,
                feature=feature,
                imputation_type=imputation_type,
                flag_time_unix=True,
            )
            new_window = window_tensor(tm, hd5, **kwargs)
            tensor = np.append(tensor, new_window)
        return tensor

    return _tensor_from_file


def create_sliding_window_tmap(tmap_name: str) -> Optional[TensorMap]:
    match = None

    imputation_type = None
    feature = "raw"
    signal_tm = None
    event_proc_tm_1 = None
    event_proc_tm_2 = None
    window = None
    step = None

    shape: Optional[Tuple[Axis, ...]] = None
    make_tensor_from_file = None
    tmap_match_name = tmap_name

    pattern = re.compile(r"^(.*)_(mean_imputation)$")
    match = pattern.findall(tmap_match_name)
    if match:
        _, imputation_type = match[0]
        tmap_match_name = tmap_match_name.replace(f"_{imputation_type}", "")
        match = None

    pattern = re.compile(fr"^(.*)_({FEATURES})$")
    match = pattern.findall(tmap_match_name)
    if match:
        _, feature = match[0]
        tmap_match_name = tmap_match_name.replace(f"_{feature}", "")
        match = None

    if not match:
        pattern = re.compile(
            r"^(.*)_(\d+)_hrs_sliding_window_(.*)_to_(.*)_(\d+)_hrs_step$",
        )
        match = pattern.findall(tmap_match_name)
        if match:
            signal_tm, window, event_proc_tm_1, event_proc_tm_2, step = match[0]
            make_tensor_from_file = make_sliding_window_tensor_from_file

    if not match:
        return None

    if feature == "raw":
        shape = (None, None)
    else:
        shape = (None,)
    if signal_tm.endswith("_timeseries"):
        shape += (2,)

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
        re.sub(r"(end_date|start_date)", "first_visit", event_proc_tm_2),
    )

    signal_tm = _get_tmap(signal_tm)
    event_proc_tm_1 = get_signal_tmap(event_proc_tm_1)
    event_proc_tm_2 = get_signal_tmap(event_proc_tm_2)
    window = int(window)
    step = int(step)

    return TensorMap(
        name=tmap_name,
        shape=shape,
        tensor_from_file=make_tensor_from_file(
            window=window,
            step=step,
            event_tm_1=event_proc_tm_1,
            event_tm_2=event_proc_tm_2,
            visit_tm=visit_tm,
            signal_tm=signal_tm,
            signal_time_tm=time_tm,
            feature=feature,
            imputation_type=imputation_type,
        ),
        channel_map=signal_tm.channel_map,
        path_prefix=signal_tm.path_prefix,
        interpretation=signal_tm.interpretation,
        validators=validator_no_nans,
    )


def make_sliding_window_outcome_tensor_from_file(
    window: int,
    step: int,
    prediction: int,
    event_tm_1: TensorMap,
    event_tm_2: TensorMap,
    visit_tm: TensorMap,
):
    def _tensor_from_file(tm, hd5, **kwargs):
        windows = get_sliding_windows(
            hd5=hd5,
            window=window,
            step=step,
            event_tm_1=event_tm_1,
            event_tm_2=event_tm_2,
            visit_tm=visit_tm,
        )
        visit = visit_tm.tensor_from_file(visit_tm, hd5, **kwargs)[0]
        event_time_2 = event_tm_2.tensor_from_file(
            event_tm_2, hd5, visits=visit, **kwargs
        )
        event_time_2 = event_time_2[0][0]
        time = windows[0]
        if time + prediction * 60 * 60 >= event_time_2:
            tensor = np.array([[0, 1]])
        else:
            tensor = np.array([[1, 0]])
        for time in windows[1:]:
            if time + prediction * 60 * 60 >= event_time_2:
                new_window = np.array([[0, 1]])
            else:
                new_window = np.array([[1, 0]])
            tensor = np.append(tensor, new_window, axis=0)
        return tensor

    return _tensor_from_file


def create_sliding_window_outcome_tmap(tmap_name: str) -> Optional[TensorMap]:
    match = None

    event_proc_tm_1 = None
    event_proc_tm_2 = None
    window = None
    step = None
    prediction = None

    shape: Optional[Tuple[Axis, ...]] = None
    make_tensor_from_file = None
    tmap_match_name = tmap_name

    if not match:
        pattern = re.compile(
            r"(\d+)_hrs_sliding_window_(.*)_to_(.*)_(\d+)_hrs_step"
            r"_(\d+)_hrs_prediction$",
        )
        match = pattern.findall(tmap_match_name)
        if match:
            window, event_proc_tm_1, event_proc_tm_2, step, prediction = match[0]
            make_tensor_from_file = make_sliding_window_outcome_tensor_from_file

    if not match:
        return None

    shape = (None, 2)

    visit_tm = get_visit_tmap(
        re.sub(r"(end_date|start_date)", "first_visit", event_proc_tm_2),
    )

    event_proc_tm_1 = get_signal_tmap(event_proc_tm_1)
    event_proc_tm_2 = get_signal_tmap(event_proc_tm_2)
    window = int(window)
    step = int(step)
    prediction = int(prediction)

    name = event_proc_tm_2.name
    channel_map = ({f"no_{name}": 0, name: 1},)

    return TensorMap(
        name=tmap_name,
        shape=shape,
        tensor_from_file=make_tensor_from_file(
            window=window,
            step=step,
            prediction=prediction,
            event_tm_1=event_proc_tm_1,
            event_tm_2=event_proc_tm_2,
            visit_tm=visit_tm,
        ),
        channel_map=channel_map,
        path_prefix=event_proc_tm_2.path_prefix,
        interpretation=event_proc_tm_2.interpretation,
        validators=validator_no_nans,
    )


def length_of_stay_event_tensor_from_file(visit_tm, event_tm, hrs_to_event, period):
    def _tensor_from_file(tm, hd5, **kwargs):
        base_path = tm.path_prefix
        visit = visit_tm.tensor_from_file(visit_tm, hd5, **kwargs)[0]
        event_time = event_tm.tensor_from_file(event_tm, hd5, visits=visit, **kwargs)[
            0
        ][0]
        path = base_path.replace("*", visit)
        admin_date = get_unix_timestamps(hd5[path].attrs["admin_date"])
        sign = -1 if period == "pre" else 1
        tensor = np.array(
            [event_time + sign * int(hrs_to_event) * 60 * 60 - admin_date],
        )
        return tensor

    return _tensor_from_file


def length_of_stay_event_double_tensor_from_file(
    visit_tm,
    event_tm,
    hrs_to_event_1,
    hrs_to_event_2,
    period,
):
    def _tensor_from_file(tm, hd5, **kwargs):
        base_path = tm.path_prefix
        visit = visit_tm.tensor_from_file(visit_tm, hd5, **kwargs)[0]
        event_time = event_tm.tensor_from_file(event_tm, hd5, visits=visit, **kwargs)[
            0
        ][0]
        path = base_path.replace("*", visit)
        admin_date = get_unix_timestamps(hd5[path].attrs["admin_date"])
        sign = -1 if period == "pre" else 1
        tensor = np.array(
            [
                np.array(
                    [event_time + sign * int(hrs_to_event_1) * 60 * 60 - admin_date],
                ),
                np.array(
                    [event_time + sign * int(hrs_to_event_2) * 60 * 60 - admin_date],
                ),
            ],
        )
        return tensor

    return _tensor_from_file


def length_of_stay_sliding_window_tensor_from_file(
    visit_tm: TensorMap,
    window: int,
    step: int,
    event_tm_1: TensorMap,
    event_tm_2: TensorMap,
):
    def _tensor_from_file(tm, hd5, **kwargs):
        base_path = tm.path_prefix
        visit = visit_tm.tensor_from_file(visit_tm, hd5, **kwargs)[0]
        path = base_path.replace("*", visit)
        admin_date = get_unix_timestamps(hd5[path].attrs["admin_date"])
        window = get_sliding_windows(
            hd5=hd5,
            window=window,
            step=step,
            event_tm_1=event_tm_1,
            event_tm_2=event_tm_2,
            visit_tm=visit_tm,
        )
        return window - admin_date

    return _tensor_from_file


def admin_age_event_visit_tensor_from_file(
    visit_tm: TensorMap,
    samples: int = 1,
    window: int = None,
    step: int = None,
    event_tm_1: TensorMap = None,
    event_tm_2: TensorMap = None,
):
    def _tensor_from_file(tm, hd5, **kwargs):
        visit = visit_tm.tensor_from_file(visit_tm, hd5, **kwargs)[0]
        tensor = admin_age_tensor_from_file(tm, hd5, visits=visit, **kwargs)
        if window and step and event_tm_1 and event_tm_2:
            samples = get_sliding_windows(
                hd5=hd5,
                window=window,
                step=step,
                event_tm_1=event_tm_1,
                event_tm_2=event_tm_2,
                visit_tm=visit_tm,
            ).size
        return np.array([tensor] * samples)

    return _tensor_from_file


def create_static_around_tmap(tm_name: str):
    tm = None
    match = None

    if not match:
        pattern = re.compile(r"^age_(.*)_(single|double)$")
        match = pattern.findall(tm_name)
        if match:
            event_proc, samples = match[0]
            visit_tm = get_visit_tmap(
                event_proc.replace("end_date", "first_visit").replace(
                    "start_date",
                    "first_visit",
                ),
            )
            samples = 1 if samples == "single" else 2
            tm = TensorMap(
                name=tm_name,
                shape=(samples,),
                interpretation=Interpretation.CONTINUOUS,
                tensor_from_file=admin_age_event_visit_tensor_from_file(
                    visit_tm=visit_tm,
                    samples=samples,
                ),
                path_prefix="edw/*",
            )
    if not match:
        pattern = re.compile(
            r"^age_(\d+)_hrs_sliding_window_(.*)_to_(.*)_(\d+)_hrs_step$",
        )
        match = pattern.findall(tm_name)
        if match:
            window, event_proc_tm_1, event_proc_tm_2, step = match[0]
            visit_tm = get_visit_tmap(
                event_proc_tm_1.replace("end_date", "first_visit").replace(
                    "start_date",
                    "first_visit",
                ),
            )
            event_proc_tm_1 = get_signal_tmap(event_proc_tm_1)
            event_proc_tm_2 = get_signal_tmap(event_proc_tm_2)
            window = int(window)
            step = int(step)
            tm = TensorMap(
                name=tm_name,
                shape=(samples,),
                interpretation=Interpretation.CONTINUOUS,
                tensor_from_file=admin_age_event_visit_tensor_from_file(
                    visit_tm=visit_tm,
                    window=window,
                    step=step,
                    event_tm_1=event_proc_tm_1,
                    event_tm_2=event_proc_tm_2,
                ),
                path_prefix="edw/*",
            )

    if not match:
        pattern = re.compile(r"^length_of_stay_(\d+)_hrs_(pre|post)_(.*)$")
        match = pattern.findall(tm_name)
        if match:
            time, period, event_proc_tm = match[0]
            visit_tm = get_visit_tmap(
                event_proc_tm.replace("end_date", "first_visit").replace(
                    "start_date",
                    "first_visit",
                ),
            )
            event_proc_tm = get_signal_tmap(event_proc_tm)
            tm = TensorMap(
                name=tm_name,
                shape=(1,),
                interpretation=Interpretation.CONTINUOUS,
                tensor_from_file=length_of_stay_event_tensor_from_file(
                    visit_tm,
                    event_proc_tm,
                    time,
                    period,
                ),
                path_prefix="edw/*",
            )
    if not match:
        pattern = re.compile(r"^length_of_stay_(\d+)_and_(\d+)_hrs_(pre|post)_(.*)$")
        match = pattern.findall(tm_name)
        if match:
            time_1, time_2, period, event_proc_tm = match[0]
            visit_tm = get_visit_tmap(
                event_proc_tm.replace("end_date", "first_visit").replace(
                    "start_date",
                    "first_visit",
                ),
            )
            event_proc_tm = get_signal_tmap(event_proc_tm)
            tm = TensorMap(
                name=tm_name,
                shape=(1,),
                interpretation=Interpretation.CONTINUOUS,
                tensor_from_file=length_of_stay_event_double_tensor_from_file(
                    visit_tm,
                    event_proc_tm,
                    time_1,
                    time_2,
                    period,
                ),
                path_prefix="edw/*",
                time_series_limit=2,
            )
    if not match:
        pattern = re.compile(
            r"^length_of_stay_(\d+)_hrs_sliding_window_(.*)_to_(.*)_(\d+)_hrs_step$",
        )
        match = pattern.findall(tm_name)
        if match:
            window, event_proc_tm_1, event_proc_tm_2, step = match[0]
            visit_tm = get_visit_tmap(
                event_proc_tm_1.replace("end_date", "first_visit").replace(
                    "start_date",
                    "first_visit",
                ),
            )
            event_proc_tm_1 = get_signal_tmap(event_proc_tm_1)
            event_proc_tm_2 = get_signal_tmap(event_proc_tm_2)
            window = int(window)
            step = int(step)
            tm = TensorMap(
                name=tm_name,
                shape=(samples,),
                interpretation=Interpretation.CONTINUOUS,
                tensor_from_file=length_of_stay_sliding_window_tensor_from_file(
                    visit_tm=visit_tm,
                    window=window,
                    step=step,
                    event_tm_1=event_proc_tm_1,
                    event_tm_2=event_proc_tm_2,
                ),
                path_prefix="edw/*",
            )
    return tm


def get_tmap(tm_name: str) -> Optional[TensorMap]:

    tm = create_around_tmap(tm_name)
    if tm is not None:
        return tm

    tm = create_around_explore_tmap(tm_name)
    if tm is not None:
        return tm

    tm = create_sliding_window_tmap(tm_name)
    if tm is not None:
        return tm

    tm = create_sliding_window_outcome_tmap(tm_name)
    if tm is not None:
        return tm

    tm = create_static_around_tmap(tm_name)
    if tm is not None:
        return tm

    return None