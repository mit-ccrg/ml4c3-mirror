# Imports: standard library
import re
from typing import Optional

# Imports: third party
import numpy as np

# Imports: first party
from ml4c3.tensormap.TensorMap import (
    TensorMap,
    Interpretation,
    get_visits,
    get_local_timestamps,
)
from ml4c3.definitions.icu_tmap_list import DEFINED_TMAPS
from ml4c3.tensormap.icu_bedmaster_signals import get_tmap as get_bedmaster_tmap


def make_ecg_peak_tensor_from_file(lead):
    def _tensor_from_file(tm, hd5, **kwargs):
        visits = get_visits(tm, hd5, **kwargs)
        max_size = max(hd5[tm.path_prefix.replace("*", v)].size for v in visits)

        shape = (len(visits), max_size)
        flag_dates = kwargs.get("readable_dates")
        if flag_dates:
            tensor = np.zeros(shape, dtype=object)
        else:
            tensor = np.zeros(shape)

        for i, visit in enumerate(visits):
            path = tm.path_prefix.replace("*", visit)
            indices = hd5[path][()]
            is_nan_indices = np.isnan(indices)
            no_nan_indices = indices[~is_nan_indices].astype(int)
            ecg_tm = get_bedmaster_tmap(f"{lead}_time")
            data = ecg_tm.tensor_from_file(ecg_tm, hd5, visit=visit)[0][no_nan_indices]
            nan_indices = np.where(is_nan_indices)[0]
            nan_indices -= np.array(range(nan_indices.shape[0]))
            data = np.append(data, np.array([np.nan] * nan_indices.shape[0]))
            data = np.insert(data, nan_indices, np.nan)[: indices.shape[0]]
            if flag_dates:
                tensor[i][: data.shape[0]] = get_local_timestamps(data)
            else:
                tensor[i][: data.shape[0]] = data

        return tensor

    return _tensor_from_file


def make_ecg_feature_tensor_from_file(ref_peak=None):
    def _tensor_from_file(tm, hd5, **kwargs):
        visits = get_visits(tm, hd5, **kwargs)
        max_size = max(hd5[tm.path_prefix.replace("*", v)].size for v in visits)
        if tm.is_timeseries:
            shape = (len(visits), 2, max_size)
            flag_dates = kwargs.get("readable_dates")
            if flag_dates:
                tensor = np.zeros(shape, dtype=object)
            else:
                tensor = np.zeros(shape)

            for i, visit in enumerate(visits):
                path = tm.path_prefix.replace("*", visit)
                time = get_tmap(ref_peak).tensor_from_file(
                    get_tmap(ref_peak),
                    hd5,
                    visit=visit,
                )[0]
                data = hd5[path][()]
                tensor[i][0][: data.shape[0]] = data
                if flag_dates:
                    tensor[i][1][: time.shape[0]] = get_local_timestamps(time)
                else:
                    tensor[i][1][: time.shape[0]] = time
        elif tm.is_continuous:
            shape = (len(visits), max_size)
            tensor = np.zeros(shape)
            for i, visit in enumerate(visits):
                path = tm.path_prefix.replace("*", visit)
                data = hd5[path][()]
                tensor[i][: data.shape[0]] = data
        else:
            raise ValueError(
                "Incorrect interpretation "
                f"'{tm.interpretation}' for ecg_feature tmap {tm.name}.",
            )

        return tensor

    return _tensor_from_file


def create_ecg_feature_tmap(tm_name: str):
    tm = None
    match = None

    if not match:
        pattern = re.compile(r"(.*)_(i|ii|iii|v)$")
        match = pattern.findall(tm_name)
        if match:
            peak_name, lead = match[0]
            tm = TensorMap(
                name=tm_name,
                shape=(None, None),
                interpretation=Interpretation.EVENT,
                tensor_from_file=make_ecg_peak_tensor_from_file(lead),
                path_prefix=f"bedmaster/*/ecg_features/{lead}/ecg_{peak_name}",
            )
    if not match:
        pattern = re.compile(r"(.*)_(i|ii|iii|v)_(timeseries|value|time)$")
        match = pattern.findall(tm_name)
        if match:
            feature_name, lead, tm_type = match[0]
            if (
                feature_name.startswith("r")
                or feature_name.startswith("q")
                or feature_name.startswith("pr")
                or feature_name.startswith("s")
            ):
                ref_peak = "r_peak"
            elif feature_name.startswith("p"):
                ref_peak = "p_peak"
            elif feature_name.startswith("t"):
                ref_peak = "t_peak"
            if tm_type == "timeseries":
                tm = TensorMap(
                    name=tm_name,
                    shape=(None, None),
                    interpretation=Interpretation.TIMESERIES,
                    tensor_from_file=make_ecg_feature_tensor_from_file(
                        f"{lead}_{ref_peak}",
                    ),
                    path_prefix=f"bedmaster/*/ecg_features/{lead}/{feature_name}",
                )
            elif tm_type == "value":
                tm = TensorMap(
                    name=tm_name,
                    shape=(None, None),
                    interpretation=Interpretation.CONTINUOUS,
                    tensor_from_file=make_ecg_feature_tensor_from_file(),
                    path_prefix=f"bedmaster/*/ecg_features/{lead}/{feature_name}",
                )
            elif tm_type == "time":
                tm = TensorMap(
                    name=tm_name,
                    shape=(None, None),
                    interpretation=Interpretation.EVENT,
                    tensor_from_file=make_ecg_peak_tensor_from_file(lead),
                    path_prefix=f"bedmaster/*/ecg_features/{lead}/ecg_{ref_peak}",
                )
    return tm


def get_tmap(tm_name: str) -> Optional[TensorMap]:
    tm = None
    for name in DEFINED_TMAPS["ecg_features"]:
        if tm_name.startswith(name):
            tm = create_ecg_feature_tmap(tm_name)
            break
    return tm
