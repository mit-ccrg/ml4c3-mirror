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


def make_bp_array_tensor_from_file(position):
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
                    else:
                        tensor[i][0][index] = float(value.split("/")[position])
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
                    if value == "nan":
                        tensor[i][index] = value
                    else:
                        tensor[i][index] = int(value.split("/")[position])
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


def get_tmap(tm_name: str) -> Optional[TensorMap]:
    for data_type in ["labs", "flowsheet"]:
        for name in DEFINED_TMAPS[data_type]:
            if tm_name.startswith(name):
                return create_measurement_tmap(tm_name, name, data_type)
    return None
