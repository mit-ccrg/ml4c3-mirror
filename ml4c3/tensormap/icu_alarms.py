# Imports: standard library
from typing import Optional

# Imports: third party
import numpy as np

# Imports: first party
from ml4c3.definitions.icu import ALARMS_FILES
from ml4c3.tensormap.TensorMap import (
    TensorMap,
    Interpretation,
    get_visits,
    get_local_timestamps,
)
from ml4c3.definitions.icu_tmap_list import DEFINED_TMAPS


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


def get_tmap(tm_name: str) -> Optional[TensorMap]:
    tm = None
    for alarm_name in DEFINED_TMAPS["alarms"]:
        if tm_name.startswith(alarm_name):
            tm = create_alarm_tmap(tm_name, alarm_name)
    return tm
