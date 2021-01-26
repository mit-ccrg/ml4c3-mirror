# pylint: disable=too-many-return-statements
# Imports: standard library
import logging
from typing import Dict, Tuple, Callable, Optional

# Imports: third party
import numpy as np

# Imports: first party
from ml4c3.utils import get_unix_timestamps
from definitions.edw import EDW_PREFIX
from definitions.icu import ICU_SCALE_UNITS
from tensormap.TensorMap import (
    TensorMap,
    PatientData,
    Interpretation,
    get_visits,
    get_local_timestamps,
)
from tensormap.validators import (
    validator_no_empty,
    validator_no_negative,
    validator_not_all_zero,
)
from definitions.icu_tmaps import DEFINED_TMAPS
from tensormap.icu_first_visit_with_signal import get_tmap as get_visit_tmap


def make_alarm_array_tensor_from_file():
    def _tensor_from_file(tm, hd5, **kwargs):
        visits = get_visits(tm, hd5, **kwargs)
        flag_dates = kwargs.get("readable_dates")
        max_size = max(hd5[tm.path_prefix.replace("*", v)].size for v in visits)
        shape = (len(visits), max_size)
        if tm.path_prefix.endswith("start_date") and flag_dates:
            tensor = np.zeros(shape, dtype=object)
        else:
            tensor = np.zeros(shape)
        for i, visit in enumerate(visits):
            path = tm.path_prefix.replace("*", visit)
            data = hd5[path][()]
            if path.endswith("start_date") and flag_dates:
                tensor[i][: data.shape[0]] = get_local_timestamps(data)
            else:
                tensor[i][: data.shape[0]] = data

        return tensor

    return _tensor_from_file


def make_alarm_attribute_tensor_from_file(key: str):
    def _tensor_from_file(tm, hd5, **kwargs):
        visits = get_visits(tm, hd5, **kwargs)

        shape = (len(visits),) + tm.shape[1:]
        tensor = np.full(shape, "", object)
        for i, visit in enumerate(visits):
            path = tm.path_prefix.replace("*", visit)
            tensor[i] = hd5[path].attrs[key]

        tensor.astype(int)

        return tensor

    return _tensor_from_file


def create_alarm_tmap(tm_name: str, alarm_name: str) -> Optional[TensorMap]:
    tm = None
    if tm_name == f"{alarm_name}_init_date":
        tm = TensorMap(
            name=tm_name,
            shape=(None, None),  # type: ignore
            interpretation=Interpretation.EVENT,
            tensor_from_file=make_alarm_array_tensor_from_file(),
            path_prefix=f"bedmaster/*/alarms/{alarm_name}/start_date",
        )
    elif tm_name == f"{alarm_name}_duration":
        tm = TensorMap(
            name=tm_name,
            shape=(None, None),  # type: ignore
            interpretation=Interpretation.CONTINUOUS,
            tensor_from_file=make_alarm_array_tensor_from_file(),
            path_prefix=f"bedmaster/*/alarms/{alarm_name}/duration",
        )
    elif tm_name == f"{alarm_name}_level":
        tm = TensorMap(
            name=tm_name,
            shape=(None,),  # type: ignore
            interpretation=Interpretation.CONTINUOUS,
            tensor_from_file=make_alarm_attribute_tensor_from_file("level"),
            path_prefix=f"bedmaster/*/alarms/{alarm_name}",
        )

    return tm


def get_timeseries(hd5, signal_path, field, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    interpolation = kwargs.get("interpolation", "raw")
    if "waveform" not in signal_path and interpolation != "raw":
        logging.warning(
            "Using interpolation on a signal that is not a waveform."
            "This can create unwanted results.",
        )

    data = (
        get_value(hd5, signal_path, **kwargs)
        if field == "value"
        else hd5[f"{signal_path}/{field}"][()]
    )

    if interpolation == "linspace":
        raw_time = hd5[f"{signal_path}/time"]
        npoints = data.size
        time = np.linspace(raw_time[0], raw_time[-1], npoints)
    elif interpolation == "raw":
        samples = hd5[f"{signal_path}/samples_per_ts"][()]
        raw_time = hd5[f"{signal_path}/time"][()]
        time = []
        for idx, num_samples in enumerate(samples):
            time.append(raw_time[idx])
            time.extend([np.nan] * (int(num_samples) - 1))
        time = np.array(time, dtype=float)
    elif interpolation == "complete_no_nans":
        samples = hd5[f"{signal_path}/samples_per_ts"][()]
        raw_time = hd5[f"{signal_path}/time"][()]
        time = []
        for idx, num_samples in enumerate(samples):
            end_time = (
                raw_time[idx + 1] if idx != samples.size - 1 else raw_time[idx] + 0.25
            )
            time.extend(
                np.linspace(
                    raw_time[idx],
                    end_time,
                    int(num_samples),
                    endpoint=False,
                ).tolist(),
            )
        time = np.array(time, dtype=float)
    elif interpolation == "complete_nans_end":
        samples = hd5[f"{signal_path}/samples_per_ts"]
        raw_time = hd5[f"{signal_path}/time"]
        sample_freq = hd5[f"{signal_path}/sample_freq"]
        time = []
        for idx, num_samples in enumerate(samples):
            curr_sf = get_sample_freq_at_index(sample_freq, idx)
            expected_num_samples = curr_sf / 4
            nans = int(expected_num_samples - num_samples)
            end_time = (
                raw_time[idx + 1] if idx != samples.size - 1 else raw_time[idx] + 0.25
            )
            data = np.insert(data, len(time) + int(num_samples), [np.nan] * int(nans))
            time.extend(
                np.linspace(
                    raw_time[idx],
                    end_time,
                    int(expected_num_samples),
                    endpoint=False,
                ).tolist(),
            )

        time = np.array(time)
    else:
        raise ValueError(f"Wrong time interpolation type: {interpolation}")

    return data, time


def get_sample_freq_at_index(sample_freqs, idx):
    candidate_freq = -1
    for sample_freq, start_idx in sample_freqs:
        if idx == start_idx:
            return sample_freq
        if idx > start_idx:
            candidate_freq = sample_freq
        else:
            break
    return candidate_freq


def get_value(hd5, signal_path, **kwargs) -> np.ndarray:
    raw_values = kwargs.get("raw_values", False)
    if not raw_values:
        scale_factor = hd5[f"{signal_path}"].attrs["scale_factor"]
        signal_name = signal_path.split("/")[-1].upper()

        if scale_factor == 0:
            tabulated_signals = ICU_SCALE_UNITS
            if signal_name in tabulated_signals:
                scale_factor = tabulated_signals[signal_name]["scaling_factor"]
            else:
                logging.warning(
                    f"Can't find a scaling factor for signal {signal_name}."
                    f" raw values will be used.",
                )

        if scale_factor not in (0, 1):
            return hd5[f"{signal_path}/value"][()] * scale_factor

    return hd5[f"{signal_path}/value"][()]


def make_bedmaster_signal_tensor_from_file(field: str, dtype=None):
    def _tensor_from_file(tm, hd5, **kwargs) -> np.ndarray:
        visit_ids = get_visits(tm, hd5, **kwargs)
        base_path = tm.path_prefix
        if tm.is_continuous:
            max_size = max(
                hd5[f"{base_path}/{field}".replace("*", v)].size for v in visit_ids
            )
            shape: Tuple = (len(visit_ids), max_size)
            tensor = np.zeros(shape, dtype=dtype)
            for i, visit in enumerate(visit_ids):
                path = base_path.replace("*", visit)
                data = (
                    hd5[f"{path}/{field}"]
                    if field != "value"
                    else get_value(hd5, path, **kwargs)
                )
                tensor[i][: data.shape[0]] = data[()]
        elif tm.is_timeseries:
            flag_dates = kwargs.get("readable_dates")
            max_size = max(
                hd5[f"{base_path}/{field}".replace("*", v)].size
                if kwargs.get("interpolation") != "complete_nans_end"
                else get_timeseries(hd5, base_path.replace("*", v), field, **kwargs)[
                    0
                ].size
                for v in visit_ids
            )
            shape = (len(visit_ids), 2, max_size)
            if flag_dates:
                tensor = np.zeros(shape, dtype=object)
            else:
                tensor = np.zeros(shape)
            for i, visit in enumerate(visit_ids):
                path = base_path.replace("*", visit)
                data, unix_array = get_timeseries(hd5, path, field, **kwargs)
                tensor[i][0][: data.shape[0]] = data
                if flag_dates:
                    tensor[i][1][: unix_array.shape[0]] = get_local_timestamps(
                        unix_array,
                    )
                else:
                    tensor[i][1][: unix_array.shape[0]] = unix_array
        elif tm.is_event:
            flag_dates = kwargs.get("readable_dates")
            max_size = max(
                hd5[f"{base_path}/{field}".replace("*", v)].size
                if kwargs.get("interpolation") != "complete_nans_end"
                else get_timeseries(hd5, base_path.replace("*", v), field, **kwargs)[
                    0
                ].size
                for v in visit_ids
            )
            shape = (len(visit_ids), max_size)
            if flag_dates:
                tensor = np.zeros(shape, dtype=object)
            else:
                tensor = np.zeros(shape)
            for i, visit in enumerate(visit_ids):
                path = base_path.replace("*", visit)
                _, unix_array = get_timeseries(hd5, path, field, **kwargs)
                if flag_dates:
                    tensor[i][: unix_array.shape[0]] = get_local_timestamps(unix_array)
                else:
                    tensor[i][: unix_array.shape[0]] = unix_array
        else:
            raise ValueError(
                f"Incorrect interpretation "
                f"'{tm.interpretation}' for Bedmaster tmap {tm.name}.",
            )
        return tensor

    return _tensor_from_file


def make_bedmaster_metadata_tensor_from_file(field: str, numeric: bool = True):
    def _tensor_from_file(tm, hd5, **kwargs):
        visits = get_visits(tm, hd5, **kwargs)
        base_path = tm.path_prefix

        storage_type = type(hd5[base_path.replace("*", visits[0])].attrs[field])
        if storage_type == np.ndarray:
            max_size = max(
                hd5[base_path.replace("*", v)].attrs[field].size for v in visits
            )
            shape = (len(visits), max_size)
        else:
            shape = (len(visits),)

        tensor = np.zeros(shape) if numeric else np.full(shape, "", object)

        for i, visit in enumerate(visits):
            path = base_path.replace("*", visit)
            if storage_type == np.ndarray:
                data = hd5[path].attrs[field]
                tensor[i][: data.shape[0]] = data
            else:
                tensor[i] = hd5[path].attrs[field]

        return tensor if numeric else tensor.astype("U")

    return _tensor_from_file


def create_bedmaster_signal_tmap(
    signal_name: str,
    signal_type: str,
    tmap_name: str,
    field: str,
    interpretation: Interpretation,
    dtype=None,
):
    tmap = TensorMap(
        name=tmap_name,
        shape=(None, None, None),
        interpretation=interpretation,
        tensor_from_file=make_bedmaster_signal_tensor_from_file(field, dtype),
        path_prefix=f"bedmaster/*/{signal_type}/{signal_name}",
    )
    return tmap


def create_bedmaster_signal_metadata_tmap(
    signal_name: str,
    signal_type: str,
    field: str,
    numeric: bool = True,
):
    tmap = TensorMap(
        name=f"{signal_name}_{field}",
        shape=(None,),
        interpretation=Interpretation.CONTINUOUS
        if numeric
        else Interpretation.LANGUAGE,
        tensor_from_file=make_bedmaster_metadata_tensor_from_file(field, numeric),
        path_prefix=f"bedmaster/*/{signal_type}/{signal_name}",
    )
    return tmap


def get_bedmaster_signal_tmap(tmap_name: str, signal_name: str, signal_type: str):
    tm = None
    if tmap_name == f"{signal_name}_timeseries":
        tm = create_bedmaster_signal_tmap(
            signal_name=signal_name,
            signal_type=signal_type,
            tmap_name=tmap_name,
            field="value",
            interpretation=Interpretation.TIMESERIES,
        )
    elif tmap_name == f"{signal_name}_value":
        tm = create_bedmaster_signal_tmap(
            signal_name=signal_name,
            signal_type=signal_type,
            tmap_name=tmap_name,
            field="value",
            interpretation=Interpretation.CONTINUOUS,
        )
    elif tmap_name == f"{signal_name}_time":
        tm = create_bedmaster_signal_tmap(
            signal_name=signal_name,
            signal_type=signal_type,
            tmap_name=tmap_name,
            field="time",
            interpretation=Interpretation.EVENT,
        )
    elif tmap_name == f"{signal_name}_time_corr_arr_timeseries":
        tm = create_bedmaster_signal_tmap(
            signal_name=signal_name,
            signal_type=signal_type,
            tmap_name=tmap_name,
            field="time_corr_arr",
            interpretation=Interpretation.TIMESERIES,
        )
    elif tmap_name == f"{signal_name}_time_corr_arr_value":
        tm = create_bedmaster_signal_tmap(
            signal_name=signal_name,
            signal_type=signal_type,
            tmap_name=tmap_name,
            field="time_corr_arr",
            interpretation=Interpretation.CONTINUOUS,
        )
    elif tmap_name == f"{signal_name}_samples_per_ts_timeseries":
        tm = create_bedmaster_signal_tmap(
            signal_name=signal_name,
            signal_type=signal_type,
            tmap_name=tmap_name,
            field="samples_per_ts",
            interpretation=Interpretation.TIMESERIES,
        )
    elif tmap_name == f"{signal_name}_samples_per_ts_value":
        tm = create_bedmaster_signal_tmap(
            signal_name=signal_name,
            signal_type=signal_type,
            tmap_name=tmap_name,
            field="samples_per_ts",
            interpretation=Interpretation.CONTINUOUS,
        )
    elif tmap_name == f"{signal_name}_sample_freq":
        tm = create_bedmaster_signal_tmap(
            signal_name=signal_name,
            signal_type=signal_type,
            tmap_name=tmap_name,
            field="sample_freq",
            dtype="float,int",
            interpretation=Interpretation.CONTINUOUS,
        )
    elif tmap_name == f"{signal_name}_units":
        tm = create_bedmaster_signal_metadata_tmap(
            signal_name=signal_name,
            signal_type=signal_type,
            field="units",
            numeric=False,
        )
    elif tmap_name == f"{signal_name}_scale_factor":
        tm = create_bedmaster_signal_metadata_tmap(
            signal_name=signal_name,
            signal_type=signal_type,
            field="scale_factor",
            numeric=True,
        )
    elif tmap_name == f"{signal_name}_channel":
        tm = create_bedmaster_signal_metadata_tmap(
            signal_name=signal_name,
            signal_type=signal_type,
            field="channel",
            numeric=False,
        )
    return tm


def make_event_tensor_from_file():
    def _tensor_from_file(tm, hd5, **kwargs):
        visits = get_visits(tm, hd5, **kwargs)

        flag_dates = kwargs.get("readable_dates")

        max_size = max(hd5[tm.path_prefix.replace("*", v)].size for v in visits)
        shape = (len(visits), max_size)
        if flag_dates:
            tensor = np.zeros(shape, dtype=object)
        else:
            tensor = np.zeros(shape)
        for i, visit in enumerate(visits):
            path = tm.path_prefix.replace("*", visit)
            data = hd5[path][()]
            if flag_dates:
                tensor[i][: data.shape[0]] = get_local_timestamps(data)
            else:
                tensor[i][: data.shape[0]] = data

        return tensor

    return _tensor_from_file


def make_event_outcome_tensor_from_file(visit_tm, double):
    def _tensor_from_file(tm, hd5, **kwargs):
        try:
            visit = visit_tm.tensor_from_file(visit_tm, hd5, **kwargs)[0]
            path = tm.path_prefix.replace("*", visit)
            if hd5[path].shape[0] > 0:
                if double:
                    tensor = np.array([[0, 1], [1, 0]])
                else:
                    tensor = np.array([[0, 1]])
                return tensor
        except KeyError:
            pass
        if double:
            raise ValueError(f"No data for {tm.name}.")
        if not double:
            tensor = np.array([[1, 0]])
        return tensor

    return _tensor_from_file


def make_general_event_tensor_from_subevents(events_names_list):
    def _tensor_from_file(tm, hd5, **kwargs):
        visits = get_visits(tm, hd5, **kwargs)

        flag_dates = kwargs.get("readable_dates")

        sizes = []
        for visit in visits:
            size = 0
            for event_name in events_names_list:
                try:
                    size += hd5[
                        tm.path_prefix.replace("*", visit).replace("{}", event_name)
                    ].size
                except KeyError:
                    continue
            sizes.append(size)
        max_size = max(sizes)
        shape = (len(visits), max_size)
        if flag_dates:
            tensor = np.zeros(shape, dtype=object)
        else:
            tensor = np.zeros(shape)
        for i, visit in enumerate(visits):
            subpath = tm.path_prefix.replace("*", visit)
            data = []
            for event_name in events_names_list:
                try:
                    path = subpath.replace("{}", event_name)
                    data = np.append(data, hd5[path][()])
                except KeyError:
                    continue
            data = np.sort(data)
            if flag_dates:
                tensor[i][: data.shape[0]] = get_local_timestamps(data)
            else:
                tensor[i][: data.shape[0]] = data

        return tensor

    return _tensor_from_file


def make_general_event_outcome_tensor_from_subevents(
    visit_tm,
    events_names_list,
    double,
):
    def _tensor_from_file(tm, hd5, **kwargs):
        try:
            visit = visit_tm.tensor_from_file(visit_tm, hd5, **kwargs)[0]
            subpath = tm.path_prefix.replace("*", visit)
            for event_name in events_names_list:
                path = subpath.replace("{}", event_name)
                if path in hd5:
                    if hd5[path].shape[0] > 0:
                        if double:
                            tensor = np.array([[0, 1], [1, 0]])
                        else:
                            tensor = np.array([[0, 1]])
                        return tensor
        except KeyError:
            pass
        if double:
            raise ValueError(f"No data for {tm.name}.")
        if not double:
            tensor = np.array([[1, 0]])
        return tensor

    return _tensor_from_file


def create_event_tmap(tm_name: str, event_name: str, event_type: str):
    tm = None
    name = event_name.replace("|_", "")
    if tm_name == f"{name}_start_date":
        tm = TensorMap(
            name=tm_name,
            shape=(None, None),  # type: ignore
            interpretation=Interpretation.EVENT,
            tensor_from_file=make_event_tensor_from_file(),
            path_prefix=f"edw/*/{event_type}/{event_name}/start_date",
        )
    elif tm_name == f"{name}_end_date":
        tm = TensorMap(
            name=tm_name,
            shape=(None, None),  # type: ignore
            interpretation=Interpretation.EVENT,
            tensor_from_file=make_event_tensor_from_file(),
            path_prefix=f"edw/*/{event_type}/{event_name}/end_date",
        )
    elif tm_name == f"{name}_double":
        tm = TensorMap(
            name=tm_name,
            shape=(2,),
            interpretation=Interpretation.CATEGORICAL,
            tensor_from_file=make_event_outcome_tensor_from_file(
                visit_tm=get_visit_tmap(f"{name}_first_visit"),
                double=True,
            ),
            channel_map={f"no_{name}": 0, name: 1},
            path_prefix=f"edw/*/{event_type}/{event_name}/start_date",
            time_series_limit=2,
        )
    elif tm_name == f"{name}_single":
        tm = TensorMap(
            name=tm_name,
            shape=(1,),
            interpretation=Interpretation.CATEGORICAL,
            tensor_from_file=make_event_outcome_tensor_from_file(
                visit_tm=get_visit_tmap(f"{name}_first_visit"),
                double=False,
            ),
            channel_map={f"no_{name}": 0, name: 1},
            path_prefix=f"edw/*/{event_type}/{event_name}/start_date",
            time_series_limit=2,
        )
    return tm


def create_arrest_tmap(tm_name: str):
    arrest_list = ["code_start", "rapid_response_start"]
    tm = None

    if tm_name == "arrest_start_date":
        tm = TensorMap(
            name=tm_name,
            shape=(None, None),  # type: ignore
            interpretation=Interpretation.EVENT,
            tensor_from_file=make_general_event_tensor_from_subevents(arrest_list),
            path_prefix="edw/*/events/{}/start_date",
        )
    if tm_name == "arrest_double":
        tm = TensorMap(
            name=tm_name,
            shape=(2,),
            interpretation=Interpretation.CATEGORICAL,
            tensor_from_file=make_general_event_outcome_tensor_from_subevents(
                visit_tm=get_visit_tmap("arrest_first_visit"),
                events_names_list=arrest_list,
                double=True,
            ),
            channel_map={"no_arrest": 0, "arrest": 1},
            path_prefix="edw/*/events/{}/start_date",
            time_series_limit=2,
        )
    if tm_name == "arrest_single":
        tm = TensorMap(
            name=tm_name,
            shape=(1,),
            interpretation=Interpretation.CATEGORICAL,
            tensor_from_file=make_general_event_outcome_tensor_from_subevents(
                visit_tm=get_visit_tmap("arrest_first_visit"),
                events_names_list=arrest_list,
                double=False,
            ),
            channel_map={"no_arrest": 0, "arrest": 1},
            path_prefix="edw/*/events/{}/start_date",
            time_series_limit=2,
        )

    return tm


def make_measurement_array_tensor_from_file():
    def _tensor_from_file(tm, hd5, **kwargs):
        visits = get_visits(tm, hd5, **kwargs)

        flag_dates = kwargs.get("readable_dates")

        if tm.is_timeseries:
            max_size = max(
                hd5[f"{tm.path_prefix}/value".replace("*", v)].size for v in visits
            )
            shape = (len(visits), 2, max_size)
            if flag_dates:
                tensor = np.zeros(shape, dtype=object)
            else:
                tensor = np.zeros(shape)
            for i, visit in enumerate(visits):
                path = tm.path_prefix.replace("*", visit)
                data = hd5[f"{path}/value"][()]
                tensor[i][0][: data.shape[0]] = data
                unix_array = hd5[f"{path}/time"][()]
                if flag_dates:
                    tensor[i][1][: unix_array.shape[0]] = get_local_timestamps(
                        unix_array,
                    )
                else:
                    tensor[i][1][: unix_array.shape[0]] = unix_array
        elif tm.is_event or tm.is_continuous:
            max_size = max(hd5[tm.path_prefix.replace("*", v)].size for v in visits)
            shape = (len(visits), max_size)
            tensor = np.zeros(shape)
            for i, visit in enumerate(visits):
                path = tm.path_prefix.replace("*", visit)
                data = hd5[path][()]
                if "|S" in str(data.dtype):
                    data = data.astype("<U32")
                    for element in data:
                        try:
                            float(element)
                        except ValueError:
                            if element.startswith("<") or element.startswith(">"):
                                try:
                                    float(element[1:])
                                    data[data == element] = element[1:]
                                except ValueError:
                                    data[data == element] = np.nan
                            else:
                                data[data == element] = np.nan
                    data = data.astype(float)

                tensor[i][: data.shape[0]] = data
        else:
            raise ValueError(
                f"Incorrect interpretation '{tm.interpretation}' for measurement tmap.",
            )

        return tensor

    return _tensor_from_file


def make_bp_array_tensor_from_file(position: int = None):
    def _tensor_from_file(tm, hd5, **kwargs):
        visits = get_visits(tm, hd5, **kwargs)

        flag_dates = kwargs.get("readable_dates")

        if tm.interpretation == Interpretation.TIMESERIES:
            max_size = max(
                hd5[f"{tm.path_prefix}/value".replace("*", v)].size for v in visits
            )
            shape = (len(visits), 2, max_size)
            if flag_dates:
                tensor = np.zeros(shape, dtype=object)
            else:
                tensor = np.zeros(shape)
            for i, visit in enumerate(visits):
                path = tm.path_prefix.replace("*", visit)
                values = hd5[f"{path}/value"][()].astype(str)
                for index, value in enumerate(values):
                    if value == "nan":
                        tensor[i][0][index] = value
                    elif position is not None:
                        pressures = [float(pressure) for pressure in value.split("/")]
                        tensor[i][0][index] = pressures[position]
                    else:
                        pressures = [float(pressure) for pressure in value.split("/")]
                        tensor[i][0][index] = (
                            pressures[0] - pressures[-1]
                        ) / pressures[0]
                unix_array = hd5[f"{path}/time"][()]
                if flag_dates:
                    tensor[i][1][: unix_array.shape[0]] = get_local_timestamps(
                        unix_array,
                    )
                else:
                    tensor[i][1][: unix_array.shape[0]] = unix_array
        elif (
            tm.interpretation == Interpretation.EVENT
            or tm.interpretation == Interpretation.CONTINUOUS
        ):
            max_size = max(hd5[tm.path_prefix.replace("*", v)].size for v in visits)
            shape = (len(visits), max_size)
            tensor = np.zeros(shape)
            for i, visit in enumerate(visits):
                path = tm.path_prefix.replace("*", visit)
                values = hd5[path][()].astype(str)
                for index, value in enumerate(values):
                    if value == "nan" or tm.interpretation == Interpretation.EVENT:
                        tensor[i][index] = value
                    elif position is not None:
                        pressures = [float(pressure) for pressure in value.split("/")]
                        tensor[i][index] = pressures[position]
                    else:
                        pressures = [float(pressure) for pressure in value.split("/")]
                        tensor[i][index] = (pressures[0] - pressures[-1]) / pressures[0]
        else:
            raise ValueError(
                f"Incorrect interpretation '{tm.interpretation}' for measurement tmap.",
            )

        return tensor

    return _tensor_from_file


def make_measurement_attribute_tensor_from_file(key: str):
    def _tensor_from_file(tm, hd5, **kwargs):
        visits = get_visits(tm, hd5, **kwargs)

        shape = (len(visits),) + tm.shape[1:]
        tensor = np.full(shape, "", object)

        for i, visit in enumerate(visits):
            path = tm.path_prefix.replace("*", visit)
            tensor[i] = hd5[path].attrs[key]

        tensor.astype("S")

        return tensor

    return _tensor_from_file


def create_measurement_tmap(tm_name: str, measurement_name: str, measurement_type: str):
    tm = None

    if (
        "blood_pressure" in measurement_name
        or "femoral_pressure" in measurement_name
        or "pulmonary_artery_pressure" in measurement_name
    ):
        if "systolic" in tm_name:
            key = "systolic"
            position = 0
        elif "diastolic" in tm_name:
            key = "diastolic"
            position = -1
        else:
            return None

        if tm_name == f"{measurement_name}_{key}_timeseries":
            tm = TensorMap(
                name=tm_name,
                shape=(None, 2, None),
                interpretation=Interpretation.TIMESERIES,
                tensor_from_file=make_bp_array_tensor_from_file(position),
                path_prefix=f"edw/*/{measurement_type}/{measurement_name}",
            )
        elif tm_name == f"{measurement_name}_{key}_value":
            tm = TensorMap(
                name=tm_name,
                shape=(None, None),
                interpretation=Interpretation.CONTINUOUS,
                tensor_from_file=make_bp_array_tensor_from_file(position),
                path_prefix=f"edw/*/{measurement_type}/{measurement_name}/value",
            )
        elif tm_name == f"{measurement_name}_{key}_time":
            tm = TensorMap(
                name=tm_name,
                shape=(None, None),
                interpretation=Interpretation.EVENT,
                tensor_from_file=make_bp_array_tensor_from_file(position),
                path_prefix=f"edw/*/{measurement_type}/{measurement_name}/time",
            )
        elif tm_name == f"{measurement_name}_{key}_units":
            tm = TensorMap(
                name=tm_name,
                shape=(None,),
                interpretation=Interpretation.LANGUAGE,
                tensor_from_file=make_measurement_attribute_tensor_from_file("units"),
                path_prefix=f"edw/*/{measurement_type}/{measurement_name}",
            )
    elif "ppi" in measurement_name:
        if tm_name == f"{measurement_name}_timeseries":
            tm = TensorMap(
                name=tm_name,
                shape=(None, 2, None),
                interpretation=Interpretation.TIMESERIES,
                tensor_from_file=make_bp_array_tensor_from_file(),
                path_prefix=f"edw/*/{measurement_type}/blood_pressure",
            )
        elif tm_name == f"{measurement_name}_value":
            tm = TensorMap(
                name=tm_name,
                shape=(None, None),
                interpretation=Interpretation.CONTINUOUS,
                tensor_from_file=make_bp_array_tensor_from_file(),
                path_prefix=f"edw/*/{measurement_type}/blood_pressure/value",
            )
        elif tm_name == f"{measurement_name}_time":
            tm = TensorMap(
                name=tm_name,
                shape=(None, None),
                interpretation=Interpretation.EVENT,
                tensor_from_file=make_bp_array_tensor_from_file(),
                path_prefix=f"edw/*/{measurement_type}/blood_pressure/time",
            )
    else:
        if tm_name == f"{measurement_name}_timeseries":
            tm = TensorMap(
                name=tm_name,
                shape=(None, 2, None),
                interpretation=Interpretation.TIMESERIES,
                tensor_from_file=make_measurement_array_tensor_from_file(),
                path_prefix=f"edw/*/{measurement_type}/{measurement_name}",
            )
        elif tm_name == f"{measurement_name}_value":
            tm = TensorMap(
                name=tm_name,
                shape=(None, None),
                interpretation=Interpretation.CONTINUOUS,
                tensor_from_file=make_measurement_array_tensor_from_file(),
                path_prefix=f"edw/*/{measurement_type}/{measurement_name}/value",
            )
        elif tm_name == f"{measurement_name}_time":
            tm = TensorMap(
                name=tm_name,
                shape=(None, None),
                interpretation=Interpretation.EVENT,
                tensor_from_file=make_measurement_array_tensor_from_file(),
                path_prefix=f"edw/*/{measurement_type}/{measurement_name}/time",
            )
        elif tm_name == f"{measurement_name}_units":
            tm = TensorMap(
                name=tm_name,
                shape=(None,),
                interpretation=Interpretation.LANGUAGE,
                tensor_from_file=make_measurement_attribute_tensor_from_file("units"),
                path_prefix=f"edw/*/{measurement_type}/{measurement_name}",
            )

    return tm


def make_med_array_tensor_from_file():
    def _tensor_from_file(tm, hd5, **kwargs):
        visits = get_visits(tm, hd5, **kwargs)

        flag_dates = kwargs.get("readable_dates")

        if tm.is_timeseries:
            max_size = max(
                hd5[f"{tm.path_prefix}/dose".replace("*", v)].size for v in visits
            )
            shape = (len(visits), 2, max_size)
            if flag_dates:
                tensor = np.zeros(shape, dtype=object)
            else:
                tensor = np.zeros(shape)
            for i, visit in enumerate(visits):
                path = tm.path_prefix.replace("*", visit)
                data = hd5[f"{path}/dose"][()]
                tensor[i][0][: data.shape[0]] = data
                unix_array = hd5[f"{path}/start_date"][()]
                if flag_dates:
                    tensor[i][1][: unix_array.shape[0]] = get_local_timestamps(
                        unix_array,
                    )
                else:
                    tensor[i][1][: unix_array.shape[0]] = unix_array
        elif tm.is_event or tm.is_continuous:
            max_size = max(hd5[tm.path_prefix.replace("*", v)].size for v in visits)
            shape = (len(visits), max_size)
            tensor = np.zeros(shape)
            for i, visit in enumerate(visits):
                path = tm.path_prefix.replace("*", visit)
                data = hd5[path][()]
                tensor[i][: data.shape[0]] = data
        else:
            raise ValueError(
                f"Incorrect interpretation '{tm.interpretation}' "
                "for medication tmap {tm.name}.",
            )

        return tensor

    return _tensor_from_file


def make_med_attribute_tensor_from_file(key: str):
    def _tensor_from_file(tm, hd5, **kwargs):
        visits = get_visits(tm, hd5, **kwargs)

        shape = (len(visits),) + tm.shape[1:]
        tensor = np.full(shape, "", object)

        for i, visit in enumerate(visits):
            path = tm.path_prefix.replace("*", visit)
            tensor[i] = hd5[path].attrs[key]

        tensor.astype("S")

        return tensor

    return _tensor_from_file


def create_med_tmap(tm_name: str, med_name: str):
    tm = None

    if tm_name == f"{med_name}_timeseries":
        tm = TensorMap(
            name=tm_name,
            shape=(None, 2, None),  # type: ignore
            interpretation=Interpretation.TIMESERIES,
            tensor_from_file=make_med_array_tensor_from_file(),
            path_prefix=f"edw/*/med/{med_name}",
        )
    elif tm_name == f"{med_name}_dose":
        tm = TensorMap(
            name=tm_name,
            shape=(None, None),  # type: ignore
            interpretation=Interpretation.CONTINUOUS,
            tensor_from_file=make_med_array_tensor_from_file(),
            path_prefix=f"edw/*/med/{med_name}/dose",
        )
    elif tm_name == f"{med_name}_time":
        tm = TensorMap(
            name=tm_name,
            shape=(None, None),  # type: ignore
            interpretation=Interpretation.EVENT,
            tensor_from_file=make_med_array_tensor_from_file(),
            path_prefix=f"edw/*/med/{med_name}/start_date",
        )
    elif tm_name == f"{med_name}_units":
        tm = TensorMap(
            name=tm_name,
            shape=(None,),  # type: ignore
            interpretation=Interpretation.LANGUAGE,
            tensor_from_file=make_med_attribute_tensor_from_file("units"),
            path_prefix=f"edw/*/med/{med_name}",
        )
    elif tm_name == f"{med_name}_route":
        tm = TensorMap(
            name=tm_name,
            shape=(None,),  # type: ignore
            interpretation=Interpretation.LANGUAGE,
            tensor_from_file=make_med_attribute_tensor_from_file("route"),
            path_prefix=f"edw/*/med/{med_name}",
        )

    return tm


def visit_tensor_from_file(tm: TensorMap, data: PatientData) -> np.ndarray:
    return np.array(tm.time_series_filter(data))[:, None]


def create_visits_tmap():
    tmap = TensorMap(
        name="visits",
        shape=(1,),
        interpretation=Interpretation.LANGUAGE,
        tensor_from_file=visit_tensor_from_file,
        path_prefix=EDW_PREFIX,
        time_series_limit=0,
        validators=validator_no_empty,
    )
    return tmap


def mrn_tensor_from_file(tm: TensorMap, data: PatientData) -> np.ndarray:
    mrn = str(data.id)
    return np.array([mrn] * len(tm.time_series_filter(data)))[:, None]


def create_mrn_tmap():
    tmap = TensorMap(
        name="mrn",
        shape=(1,),
        interpretation=Interpretation.LANGUAGE,
        tensor_from_file=mrn_tensor_from_file,
        path_prefix=EDW_PREFIX,
        time_series_limit=0,
        validators=validator_no_empty,
    )
    return tmap


def length_of_stay_tensor_from_file(tm: TensorMap, data: PatientData) -> np.ndarray:
    visits = tm.time_series_filter(data)
    shape = (len(visits),) + tm.shape
    tensor = np.zeros(shape)

    for i, visit in enumerate(visits):
        try:
            path = f"{tm.path_prefix}/{visit}"
            end_date = get_unix_timestamps(data[path].attrs["end_date"])
            start_date = get_unix_timestamps(data[path].attrs["admin_date"])
            tensor[i] = (end_date - start_date) / 60 / 60
        except (ValueError, KeyError) as e:
            logging.debug(f"Could not get length of stay from {data.id}/{visit}")
            logging.debug(e)

    return tensor


def create_length_of_stay_tmap():
    tmap = TensorMap(
        name="length_of_stay",
        shape=(1,),
        interpretation=Interpretation.CONTINUOUS,
        tensor_from_file=length_of_stay_tensor_from_file,
        path_prefix=EDW_PREFIX,
        time_series_limit=0,
        validators=validator_no_negative,
    )
    return tmap


def admin_age_tensor_from_file(
    tm: TensorMap, data: PatientData, **kwargs
) -> np.ndarray:

    if "visits" in kwargs:
        visits = kwargs["visits"]
        if isinstance(visits, str):
            visits = [visits]
    else:
        visits = tm.time_series_filter(data)

    shape = (len(visits),) + tm.shape
    tensor = np.zeros(shape)

    for i, visit in enumerate(visits):
        try:
            path = f"{tm.path_prefix}/{visit}"
            admit_date = get_unix_timestamps(data[path].attrs["admin_date"])
            birth_date = get_unix_timestamps(data[path].attrs["birth_date"])
            age = admit_date - birth_date
            tensor[i] = age / 60 / 60 / 24 / 365
        except (ValueError, KeyError) as e:
            logging.debug(f"Could not get age from {data.id}/{visit}")
            logging.debug(e)

    return tensor


def create_age_tmap():
    tmap = TensorMap(
        name="age",
        shape=(1,),
        interpretation=Interpretation.CONTINUOUS,
        tensor_from_file=admin_age_tensor_from_file,
        path_prefix=EDW_PREFIX,
        time_series_limit=0,
        validators=validator_no_negative,
    )
    return tmap


def sex_double_tensor_from_file(tm: TensorMap, data: PatientData) -> np.ndarray:
    visit = tm.time_series_filter(data)[0]
    shape = (2,) + tm.shape
    tensor = np.zeros(shape)
    path = f"{tm.path_prefix}/{visit}"
    value = data[path].attrs["sex"]
    tensor[:, tm.channel_map[value.lower()]] = np.array([1, 1])

    return tensor


def create_sex_double_tmap():
    tmap = TensorMap(
        name="sex_double",
        interpretation=Interpretation.CATEGORICAL,
        tensor_from_file=sex_double_tensor_from_file,
        channel_map={"male": 0, "female": 1},
        path_prefix=EDW_PREFIX,
        time_series_limit=2,
        validators=validator_not_all_zero,
    )
    return tmap


def make_static_tensor_from_file(key: str) -> Callable:
    def _tensor_from_file(tm: TensorMap, data: PatientData, **kwargs) -> np.ndarray:
        unix_dates = kwargs.get("unix_dates")
        if "visits" in kwargs:
            visits = kwargs["visits"]
            if isinstance(visits, str):
                visits = [visits]
        else:
            visits = tm.time_series_filter(data)

        temp = None
        finalize = False
        if tm.is_timeseries:
            temp = [data[f"{tm.path_prefix}/{v}"].attrs[key] for v in visits]
            max_len = max(map(len, temp))
            shape = (len(visits), max_len)
        else:
            shape = (len(visits),) + tm.shape

        if tm.is_categorical or tm.is_continuous or (tm.is_event and unix_dates):
            tensor = np.zeros(shape)
        elif tm.is_language or (tm.is_event and not unix_dates):
            tensor = np.full(shape, "", object)
            finalize = True
        elif tm.is_timeseries and temp is not None:
            if isinstance(temp[0][0], np.number):
                tensor = np.zeros(shape)
            else:
                tensor = np.full(shape, "", object)
                finalize = True
        else:
            raise ValueError("Unknown interpretation for static ICU data")

        for i, visit in enumerate(visits):
            try:
                path = f"{tm.path_prefix}/{visit}"
                value = data[path].attrs[key] if temp is None else temp[i]
                if tm.channel_map:
                    tensor[i, tm.channel_map[value.lower()]] = 1
                elif tm.is_event and unix_dates:
                    tensor[i] = get_unix_timestamps(value)
                else:
                    tensor[i] = value
            except (ValueError, KeyError) as e:
                logging.debug(f"Error getting {key} from {data.id}/{visit}")
                logging.debug(e)

        if finalize:
            tensor = np.array(tensor, dtype=str)
        return tensor

    return _tensor_from_file


def create_timeseries_tmap(key: str) -> TensorMap:
    tmap = TensorMap(
        name=key,
        shape=(None,),
        interpretation=Interpretation.TIMESERIES,
        tensor_from_file=make_static_tensor_from_file(key),
        path_prefix=EDW_PREFIX,
        time_series_limit=0,
    )
    return tmap


def create_language_tmap(key: str) -> TensorMap:
    tmap = TensorMap(
        name=key,
        shape=(1,),
        interpretation=Interpretation.LANGUAGE,
        tensor_from_file=make_static_tensor_from_file(key),
        path_prefix=EDW_PREFIX,
        time_series_limit=0,
    )
    return tmap


def create_static_event_tmap(key: str) -> TensorMap:
    tmap = TensorMap(
        name=key,
        shape=(1,),
        interpretation=Interpretation.EVENT,
        tensor_from_file=make_static_tensor_from_file(key),
        path_prefix=EDW_PREFIX,
        time_series_limit=0,
        validators=validator_no_empty,
    )
    return tmap


def create_continuous_tmap(key: str) -> TensorMap:
    tmap = TensorMap(
        name=key,
        shape=(1,),
        interpretation=Interpretation.CONTINUOUS,
        tensor_from_file=make_static_tensor_from_file(key),
        path_prefix=EDW_PREFIX,
        time_series_limit=0,
        validators=validator_no_negative,
    )
    return tmap


def create_categorical_tmap(key: str, channel_map: Dict[str, int]) -> TensorMap:
    tmap = TensorMap(
        name=key,
        interpretation=Interpretation.CATEGORICAL,
        tensor_from_file=make_static_tensor_from_file(key),
        channel_map=channel_map,
        path_prefix=EDW_PREFIX,
        time_series_limit=0,
        validators=validator_not_all_zero,
    )
    return tmap


def get_tmap(tm_name: str) -> Optional[TensorMap]:
    # Because we have so many tensor maps being made in this one function, we may
    # occasionally have partial matches for names (e.g. sodium in labs and medications)

    # So instead of always returning the result of "create" functions, first check if
    # the resulting object is None, if so, keep trying other methods to create the tm.

    # alarms
    for alarm_name in DEFINED_TMAPS["alarms"]:
        if tm_name.startswith(alarm_name):
            tm = create_alarm_tmap(tm_name, alarm_name)
            if tm is not None:
                return tm

    # bedmaster signals
    for signal_type in ["vitals", "waveform"]:
        for signal_name in DEFINED_TMAPS[signal_type]:
            if tm_name.startswith(f"{signal_name}_"):
                tm = get_bedmaster_signal_tmap(
                    tmap_name=tm_name,
                    signal_name=signal_name,
                    signal_type=signal_type,
                )
                if tm is not None:
                    return tm

    # events
    for data_type in ["surgery", "procedures", "transfusions", "events"]:
        for name in DEFINED_TMAPS[data_type]:
            if tm_name.startswith(name):
                tm = create_event_tmap(tm_name, name, data_type)
                if tm is not None:
                    return tm

    if tm_name.startswith("arrest"):
        tm = create_arrest_tmap(tm_name)
        if tm is not None:
            return tm

    # measurements
    for data_type in ["labs", "flowsheet"]:
        for name in DEFINED_TMAPS[data_type]:
            if tm_name.startswith(name):
                tm = create_measurement_tmap(tm_name, name, data_type)
                if tm is not None:
                    return tm

    # medications
    for name in DEFINED_TMAPS["med"]:
        if tm_name.startswith(name):
            tm = create_med_tmap(tm_name, name)
            if tm is not None:
                return tm

    # static
    tm = None
    source = "static"
    if tm_name in DEFINED_TMAPS[f"{source}_language"]:
        tm = create_language_tmap(tm_name)
    elif tm_name in DEFINED_TMAPS[f"{source}_continuous"]:
        tm = create_continuous_tmap(tm_name)
    elif tm_name in DEFINED_TMAPS[f"{source}_categorical"]:
        tm = create_categorical_tmap(
            tm_name,
            DEFINED_TMAPS[f"{source}_categorical"][tm_name],
        )
    elif tm_name in DEFINED_TMAPS[f"{source}_event"]:
        tm = create_static_event_tmap(tm_name)
    elif tm_name in DEFINED_TMAPS[f"{source}_timeseries"]:
        tm = create_timeseries_tmap(tm_name)
    elif tm_name == "mrn":
        tm = create_mrn_tmap()
    elif tm_name == "visits":
        tm = create_visits_tmap()
    elif tm_name == "length_of_stay":
        tm = create_length_of_stay_tmap()
    elif tm_name == "age":
        tm = create_age_tmap()
    elif tm_name == "sex_double":
        tm = create_sex_double_tmap()

    return tm
