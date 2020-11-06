# Imports: standard library
from typing import Optional

# Imports: third party
import numpy as np

# Imports: first party
from ml4c3.definitions.icu import EDW_FILES
from ml4c3.tensormap.TensorMap import (
    TensorMap,
    Interpretation,
    get_visits,
    get_local_timestamps,
)
from ml4c3.definitions.icu_tmap_list import DEFINED_TMAPS


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


def get_tmap(tm_name: str) -> Optional[TensorMap]:
    tm = None
    for name in DEFINED_TMAPS["med"]:
        if tm_name.startswith(name):
            tm = create_med_tmap(tm_name, name)
            break
    return tm
