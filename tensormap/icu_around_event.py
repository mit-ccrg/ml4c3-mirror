# type: ignore
# Imports: standard library
import re
from typing import List, Tuple, Union, Optional

# Imports: third party
import numpy as np

# Imports: first party
from ml4c3.utils import get_unix_timestamps
from tensormap.TensorMap import Axis, TensorMap, Interpretation
from tensormap.validators import validator_no_nans
from tensormap.icu_signals import get_tmap as get_signal_tmap
from tensormap.icu_signals import admin_age_tensor_from_file
from tensormap.icu_ecg_features import get_tmap as get_ecg_tmap
from tensormap.icu_signal_metrics import compute_feature
from tensormap.icu_first_visit_with_signal import get_tmap as get_visit_tmap

FEATURES = (
    "min|max|mean|median|std|first|last|count|mean_slope|mean_crossing_rate|"
    r"\d+_last_values"
)


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


def get_time_tm(signal_tm):
    """
    From a tensormap that gives the value of a signal (+ time), obtain the
    tensormap that just returns the time array.
    """
    if not signal_tm:
        return None
    time_tm = signal_tm.replace("_scaled", "").replace("_mean_imputation", "")
    if time_tm.endswith("_timeseries"):
        time_tm = time_tm.replace("_timeseries", "_time")
    elif time_tm.endswith("_value"):
        time_tm = time_tm.replace("_value", "_time")
    elif time_tm.endswith("_dose"):
        time_tm = time_tm.replace("_dose", "_time")
    elif time_tm.endswith("_duration"):
        time_tm = time_tm.replace("_duration", "_init_date")
    time_tm = _get_tmap(time_tm)
    return time_tm


def get_sliding_windows(
    hd5,
    window: int,
    step: int,
    event_tm_1: TensorMap,
    event_tm_2: TensorMap,
    visit_tm: TensorMap,
    buffer_adm_time: int = 24,
    **kwargs,
):
    """
    Create a sliding window from the time associated to <event_tm_1> to <event_tm_2>
    with step size <step> and window length <window>.
    """

    if not hasattr(get_sliding_windows, "windows_cache"):
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
        event_time_1 + (buffer_adm_time + window) * 60 * 60,
        event_time_2,
        step * 60 * 60,
    )
    get_sliding_windows.windows_cache[hd5.id] = windows
    if windows.size == 0:
        raise ValueError(
            "It is not possible to compute a sliding window with the given parameters.",
        )
    return windows


def make_around_event_tensor_from_file(
    times: Union[int, List[int]],
    window: int,
    periods: Union[str, List[str]],
    features: List[str],
    event_tms: Union[TensorMap, List[TensorMap]],
    visit_tm: TensorMap,
    signal_tm: TensorMap,
    signal_time_tm: TensorMap,
    imputation_type: str = None,
    sample_and_hold: bool = True,
    flag_time_unix: bool = False,
):
    if not isinstance(times, (list, np.ndarray)):
        times = [times]
    if not isinstance(periods, (list, np.ndarray)):
        periods = [periods]
    if not isinstance(event_tms, (list, np.ndarray)):
        event_tms = [event_tms]
    if len(times) != len(periods) or len(periods) != len(event_tms):
        raise ValueError(
            "The number of elements for times, periods and event_tms has to be "
            "the same.",
        )

    def around_event_tensor(time, period, event_tm, hd5, **kwargs):
        visit = visit_tm.tensor_from_file(visit_tm, hd5, **kwargs)[0]
        if flag_time_unix:
            if period == "pre":
                start = time - window * 60 * 60
                end = time
            else:
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
        time_tensor = signal_time_tm.tensor_from_file(
            signal_time_tm,
            hd5,
            visits=visit,
            **kwargs,
        )
        indices = np.where(np.logical_and(start < time_tensor, end > time_tensor)[0])[0]
        tensor = np.array([])
        for feature in features:
            new_tensor = compute_feature(
                tm=signal_tm,
                hd5=hd5,
                visit=visit,
                indices=indices,
                feature=feature,
                period=period,
                imputation_type=imputation_type,
                **kwargs,
            )
            if feature == "raw":
                return new_tensor
            tensor = np.append(tensor, new_tensor)
        return tensor

    def _tensor_from_file(tm, hd5, **kwargs):  # pylint: disable=unused-argument
        if features == ["raw"] or features[-1].endswith("_last_values"):
            tensor = np.array([[]])
        else:
            tensor = np.array([])
        for i, _ in enumerate(times):
            window_tensor = around_event_tensor(
                time=times[i],
                period=periods[i],
                event_tm=event_tms[i],
                hd5=hd5,
                **kwargs,
            )
            if "_explore" in tm.name:
                tensor = np.array(window_tensor)
            elif i == 0:
                tensor = np.array([window_tensor])
            else:
                if sample_and_hold and np.isnan(window_tensor).all():
                    window_tensor = tensor[-1]
                tensor = np.append(tensor, np.array([window_tensor]), axis=0)
        return tensor

    return _tensor_from_file


def make_sliding_window_tensor_from_file(
    window: int,
    step: int,
    features: List[str],
    event_tm_1: TensorMap,
    event_tm_2: TensorMap,
    visit_tm: TensorMap,
    signal_tm: TensorMap,
    signal_time_tm: TensorMap,
    imputation_type=None,
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
        window_tensor = make_around_event_tensor_from_file(
            times=windows,
            window=window,
            periods=["pre"] * windows.size,
            features=features,
            event_tms=[event_tm_2] * windows.size,
            visit_tm=visit_tm,
            signal_tm=signal_tm,
            signal_time_tm=signal_time_tm,
            imputation_type=imputation_type,
            flag_time_unix=True,
        )
        tensor = window_tensor(tm, hd5, **kwargs)
        return tensor

    return _tensor_from_file


def make_sliding_window_outcome_tensor_from_file(
    window: int,
    step: int,
    prediction: int,
    gap: int,
    event_tm_1: TensorMap,
    event_tm_2: TensorMap,
    visit_tm: TensorMap,
):
    def _tensor_from_file(tm, hd5, **kwargs):  # pylint: disable=unused-argument
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
        prediction_array = windows + prediction * 60 * 60 >= event_time_2
        gap_array = windows + gap * 60 * 60 >= event_time_2
        labels = prediction_array - 2 * gap_array
        tensor = np.array(
            list(
                map(
                    lambda x: [np.nan, np.nan]
                    if x < 0
                    else ([0, 1] if x == 1 else [1, 0]),
                    labels,
                ),
            ),
        )
        return tensor

    return _tensor_from_file


def create_around_tmap(tmap_name: str) -> Optional[TensorMap]:
    match = None

    time_2 = None
    period_2 = None
    event_proc_tm_2 = None
    imputation_type = None
    features = []

    shape: Optional[Tuple[Axis, ...]] = None
    make_tensor_from_file = None
    tmap_match_name = tmap_name

    pattern = re.compile(r"^(.*)_(mean_imputation|sample_and_hold)$")
    match = pattern.findall(tmap_match_name)
    if match:
        _, imputation_type = match[0]
        tmap_match_name = tmap_match_name.replace(f"_{imputation_type}", "")
        match = None

    pattern = re.compile(fr"^(.*)_({FEATURES})$")
    match = pattern.findall(tmap_match_name)
    k = 0
    channel_map = {}
    while match:
        _, new_feature = match[0]
        tmap_match_name = tmap_match_name.replace(f"_{new_feature}", "")
        features.append(new_feature)
        channel_map[f"{new_feature}"] = k
        match = pattern.findall(tmap_match_name)
        k += 1
    if not features:
        features = ["raw"]
    if len(features) == 1:
        channel_map = None

    if tmap_match_name.endswith("_explore"):
        tmap_name = tmap_match_name
        tmap_match_name = tmap_match_name.replace("_explore", "")

    if not match:
        pattern = re.compile(
            r"^(.*)_(\d+)_hrs_(pre|post)_(.*)_(\d+)_hrs_(pre|post)_(.*)"
            r"_(\d+)_hrs_window$",
        )
        match = pattern.findall(tmap_match_name)
        if match:
            (
                signal_tm,
                time_1,
                period_1,
                event_proc_tm,
                time_2,
                period_2,
                event_proc_tm_2,
                window,
            ) = match[0]
            make_tensor_from_file = make_around_event_tensor_from_file
            times = [int(time_1), int(time_2)]
            periods = [period_1, period_2]

    if not match:
        pattern = re.compile(
            r"^(.*)_(\d+)_hrs_(pre|post)_(.*)_(\d+)_hrs_window$",
        )
        match = pattern.findall(tmap_match_name)
        if match:
            signal_tm, time, period, event_proc_tm, window = match[0]
            make_tensor_from_file = make_around_event_tensor_from_file
            times = [int(time)]
            periods = [period]

    if not match:
        return None

    if signal_tm.endswith("_timeseries"):
        if features == ["raw"]:
            shape = (None, 2)
        else:
            shape = (2,)
    else:
        if features == ["raw"]:
            shape = (None,)
        else:
            shape = (len(features),)

    time_tm = get_time_tm(signal_tm)

    visit_tm = get_visit_tmap(
        re.sub(r"(end_date|start_date)", "first_visit", event_proc_tm),
    )

    window = int(window)
    signal_tm = _get_tmap(signal_tm)
    event_proc_tms = [get_signal_tmap(event_proc_tm)]
    if event_proc_tm_2 is not None:
        event_proc_tms.append(get_signal_tmap(event_proc_tm_2))

    return TensorMap(
        name=tmap_name,
        shape=shape,
        tensor_from_file=make_tensor_from_file(
            times=times,
            periods=periods,
            window=window,
            features=features,
            event_tms=event_proc_tms,
            visit_tm=visit_tm,
            signal_tm=signal_tm,
            signal_time_tm=time_tm,
            imputation_type=imputation_type,
        ),
        channel_map=channel_map,
        path_prefix=signal_tm.path_prefix,
        interpretation=signal_tm.interpretation,
        validators=validator_no_nans,
    )


def create_sliding_window_tmap(tmap_name: str) -> Optional[TensorMap]:
    match = None

    imputation_type = None
    features = []
    tmap_match_name = tmap_name

    pattern = re.compile(r"^(.*)_(mean_imputation)$")
    match = pattern.findall(tmap_match_name)
    if match:
        _, imputation_type = match[0]
        tmap_match_name = tmap_match_name.replace(f"_{imputation_type}", "")
        match = None

    pattern = re.compile(fr"^(.*)_({FEATURES})$")
    match = pattern.findall(tmap_match_name)
    k = 0
    channel_map = {}
    while match:
        _, new_feature = match[0]
        tmap_match_name = tmap_match_name.replace(f"_{new_feature}", "")
        features.append(new_feature)
        channel_map[f"{new_feature}"] = k
        match = pattern.findall(tmap_match_name)
        k += 1
    if not features:
        features = ["raw"]
    if len(features) == 1:
        channel_map = None

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

    if features == ["raw"]:
        shape = (None,)
    else:
        shape = (len(features),)
    if signal_tm.endswith("_timeseries"):
        shape += (2,)

    time_tm = get_time_tm(signal_tm)
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
            features=features,
            imputation_type=imputation_type,
        ),
        channel_map=channel_map,
        path_prefix=signal_tm.path_prefix,
        interpretation=signal_tm.interpretation,
        validators=validator_no_nans,
        time_series_limit=0,
    )


def create_sliding_window_outcome_tmap(tmap_name: str) -> Optional[TensorMap]:
    match = None
    if not match:
        pattern = re.compile(
            r"(\d+)_hrs_sliding_window_(.*)_to_(.*)_(\d+)_hrs_step"
            r"_(\d+)_hrs_prediction_(\d+)_hrs_gap$",
        )
        match = pattern.findall(tmap_name)
        if match:
            window, event_proc_tm_1, event_proc_tm_2, step, prediction, gap = match[0]
            make_tensor_from_file = make_sliding_window_outcome_tensor_from_file

            visit_tm = get_visit_tmap(
                re.sub(r"(end_date|start_date)", "first_visit", event_proc_tm_2),
            )
            event_proc_tm_1 = get_signal_tmap(event_proc_tm_1)
            event_proc_tm_2 = get_signal_tmap(event_proc_tm_2)
            window = int(window)
            step = int(step)
            prediction = int(prediction)
            gap = int(gap)
            name = event_proc_tm_2.name
            return TensorMap(
                name=tmap_name,
                shape=(2,),
                tensor_from_file=make_tensor_from_file(
                    window=window,
                    step=step,
                    prediction=prediction,
                    gap=gap,
                    event_tm_1=event_proc_tm_1,
                    event_tm_2=event_proc_tm_2,
                    visit_tm=visit_tm,
                ),
                channel_map={f"no_{name}": 0, name: 1},
                path_prefix=event_proc_tm_2.path_prefix,
                interpretation=Interpretation.CATEGORICAL,
                validators=validator_no_nans,
                time_series_limit=0,
            )
    return None


def length_of_stay_event_tensor_from_file(
    visit_tm: TensorMap,
    event_tm: TensorMap,
    hrs_to_event: Union[int, float, List[Union[int, float]]],
    periods: Union[str, List[str]],
):
    def _tensor_from_file(tm, hd5, **kwargs):
        base_path = tm.path_prefix
        visit = visit_tm.tensor_from_file(visit_tm, hd5, **kwargs)[0]
        path = base_path.replace("*", visit)
        admin_date = get_unix_timestamps(hd5[path].attrs["admin_date"])
        event_time = event_tm.tensor_from_file(event_tm, hd5, visits=visit, **kwargs)[
            0
        ][0]
        sign = np.array(list(map(lambda x: -1 if x == "pre" else 1, periods)))
        tensor = event_time + sign * hrs_to_event * 60 * 60 - admin_date
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
        windows = get_sliding_windows(
            hd5=hd5,
            window=window,
            step=step,
            event_tm_1=event_tm_1,
            event_tm_2=event_tm_2,
            visit_tm=visit_tm,
        )
        return windows - admin_date

    return _tensor_from_file


def admin_age_event_visit_tensor_from_file(
    visit_tm: TensorMap,
    samples: int = 1,  # pylint: disable=unused-argument
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
        return TensorMap(
            name=tm_name,
            shape=(samples,),
            interpretation=Interpretation.CONTINUOUS,
            tensor_from_file=admin_age_event_visit_tensor_from_file(
                visit_tm=visit_tm,
                samples=samples,
            ),
            path_prefix="edw/*",
        )
    pattern = re.compile(
        r"^(age|length_of_stay)_(\d+)_hrs_sliding_window_(.*)"
        r"_to_(.*)_(\d+)_hrs_step$",
    )
    match = pattern.findall(tm_name)
    if match:
        signal, window, event_proc_tm_1, event_proc_tm_2, step = match[0]
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
        if signal == "age":
            tensor_from_file = admin_age_event_visit_tensor_from_file
        else:
            tensor_from_file = length_of_stay_sliding_window_tensor_from_file
        return TensorMap(
            name=tm_name,
            shape=(1,),
            interpretation=Interpretation.CONTINUOUS,
            tensor_from_file=tensor_from_file(
                visit_tm=visit_tm,
                window=window,
                step=step,
                event_tm_1=event_proc_tm_1,
                event_tm_2=event_proc_tm_2,
            ),
            path_prefix="edw/*",
        )
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
        hrs_to_event = [int(time)]
        periods = [period]
    else:
        pattern = re.compile(
            r"^length_of_stay_(\d+)_hrs_(pre|post)_(.*)" r"_(\d+)_hrs_(pre|post)_(.*)$",
        )
        match = pattern.findall(tm_name)
        if match:
            time_1, period_1, time_2, period_2, event_proc_tm = match[0]
            hrs_to_event = [int(time_1), int(time_2)]
            periods = [period_1, period_2]
    if match:
        visit_tm = get_visit_tmap(
            event_proc_tm.replace("end_date", "first_visit").replace(
                "start_date",
                "first_visit",
            ),
        )
        event_proc_tm = get_signal_tmap(event_proc_tm)
        return TensorMap(
            name=tm_name,
            shape=(1,),
            interpretation=Interpretation.CONTINUOUS,
            tensor_from_file=length_of_stay_event_tensor_from_file(
                visit_tm=visit_tm,
                event_tm=event_proc_tm,
                hrs_to_event=hrs_to_event,
                periods=periods,
            ),
            path_prefix="edw/*",
        )
    return None


def get_tmap(tm_name: str) -> Optional[TensorMap]:

    tm = create_around_tmap(tm_name)
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
