# Imports: standard library
import logging
from typing import Tuple, Optional

# Imports: third party
import numpy as np

# Imports: first party
from ml4c3.definitions.icu import ICU_SCALE_UNITS
from ml4c3.tensormap.TensorMap import (
    TensorMap,
    Interpretation,
    get_visits,
    get_local_timestamps,
)
from ml4c3.definitions.icu_tmap_list import DEFINED_TMAPS


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


def get_tmap(tmap_name: str) -> Optional[TensorMap]:
    for signal_type in ["vitals", "waveform"]:
        for signal_name in DEFINED_TMAPS[signal_type]:
            if tmap_name.startswith(f"{signal_name}_"):
                return get_bedmaster_signal_tmap(
                    tmap_name=tmap_name,
                    signal_name=signal_name,
                    signal_type=signal_type,
                )
    return None
