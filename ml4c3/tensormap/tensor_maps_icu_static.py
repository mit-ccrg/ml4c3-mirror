# Imports: standard library
from typing import Any, Dict, Optional

# Imports: third party
import numpy as np

# Imports: first party
from ml4c3.utils import get_unix_timestamps
from ml4c3.definitions.icu import EDW_FILES
from ml4c3.tensormap.TensorMap import TensorMap, Interpretation, get_visits
from ml4c3.definitions.icu_tmap_list import DEFINED_TMAPS

# pylint: disable=too-many-return-statements


def visit_tensor_from_file(tm, hd5, **kwargs):
    base_path = tm.path_prefix
    visits = get_visits(tm, hd5, **kwargs)
    for visit in visits:
        path = base_path.replace("*", visit)
        if path not in hd5:
            raise KeyError(f"Visit {visit} not found in hd5")
    tensor = np.array(visits)
    return tensor


def length_of_stay_tensor_from_file(tm, hd5, **kwargs):
    base_path = tm.path_prefix
    visits = get_visits(tm, hd5, **kwargs)

    shape = (len(visits),) + tm.shape[1:]
    tensor = np.full(shape, "", object)

    for i, visit in enumerate(visits):
        path = base_path.replace("*", visit)
        end_date = kwargs.get("end_date", hd5[path].attrs["end_date"])
        tensor[i] = get_unix_timestamps(end_date) - get_unix_timestamps(
            hd5[path].attrs["admin_date"],
        )

    return tensor


def admin_age_tensor_from_file(tm, hd5, **kwargs):
    base_path = tm.path_prefix
    visits = get_visits(tm, hd5, **kwargs)

    shape = (len(visits),) + tm.shape[1:]
    tensor = np.full(shape, "", object)

    for i, visit in enumerate(visits):
        path = base_path.replace("*", visit)
        age = get_unix_timestamps(hd5[path].attrs["admin_date"]) - get_unix_timestamps(
            hd5[path].attrs["birth_date"],
        )
        tensor[i] = round(age / 60 / 60 / 24 / 365)

    return tensor


def make_sex_double_tensor_from_file():
    def _tensor_from_file(tm, hd5, **kwargs):
        base_path = tm.path_prefix
        visit = get_visits(tm, hd5, **kwargs)[0]

        shape = (2,) + tm.shape[1:]
        tensor = np.zeros(shape, dtype=int)
        path = base_path.replace("*", visit)
        data = hd5[path].attrs["sex"]
        tensor[:, tm.channel_map[data.lower()]] = np.array([1, 1])

        return tensor

    return _tensor_from_file


def make_static_tensor_from_file(key):
    def _tensor_from_file(tm, hd5, **kwargs):
        base_path = tm.path_prefix
        visits = get_visits(tm, hd5, **kwargs)

        shape = (len(visits),) + tm.shape[1:]
        tensor = None
        if tm.is_categorical or key == "height" or key == "weight":
            tensor = np.zeros(shape)
        elif tm.is_continuous:
            max_size = max(
                hd5[base_path.replace("*", v)].attrs[key].size for v in visits
            )
            shape = (len(visits), max_size)
            if isinstance(
                hd5[base_path.replace("*", visits[0])].attrs[key][0],
                (int, np.int64, float),
            ):
                tensor = np.zeros(shape)
            else:
                tensor = np.full(shape, "", object)
        elif tm.is_language or tm.is_event:
            tensor = np.full(shape, "", object)

        for i, visit in enumerate(visits):
            path = base_path.replace("*", visit)
            data = hd5[path].attrs[key]
            if tm.channel_map:
                tensor[i, tm.channel_map[data.lower()]] = 1
            else:
                if tm.is_continuous and not key == "height" and not key == "weight":
                    tensor[i][: data.shape[0]] = data
                else:
                    tensor[i] = data

        if tm.is_language:
            tensor = np.array(tensor, dtype=str)
        return tensor

    return _tensor_from_file


def create_visits_tmap():
    tmap = TensorMap(
        name="visits",
        shape=(None,),
        interpretation=Interpretation.LANGUAGE,
        tensor_from_file=visit_tensor_from_file,
        path_prefix="edw/*",
    )
    return tmap


def create_length_of_stay_tmap():
    tmap = TensorMap(
        name="length_of_stay",
        shape=(None,),
        interpretation=Interpretation.CONTINUOUS,
        tensor_from_file=length_of_stay_tensor_from_file,
        path_prefix="edw/*",
    )
    return tmap


def create_age_tmap():
    tmap = TensorMap(
        name="age",
        shape=(None,),
        interpretation=Interpretation.CONTINUOUS,
        tensor_from_file=admin_age_tensor_from_file,
        path_prefix="edw/*",
    )
    return tmap


def create_sex_double_tmap():
    tmap = TensorMap(
        name="sex_double",
        shape=(2,),
        interpretation=Interpretation.CATEGORICAL,
        tensor_from_file=make_sex_double_tensor_from_file(),
        channel_map={"male": 0, "female": 1},
        path_prefix="edw/*",
        time_series_limit=2,
    )
    return tmap


def create_language_tmap(key: str):
    tmap = TensorMap(
        name=key,
        shape=(None,),
        interpretation=Interpretation.LANGUAGE,
        tensor_from_file=make_static_tensor_from_file(key),
        path_prefix="edw/*",
    )
    return tmap


def create_event_tmap(key: str):
    tmap = TensorMap(
        name=key,
        shape=(None,),
        interpretation=Interpretation.EVENT,
        tensor_from_file=make_static_tensor_from_file(key),
        path_prefix="edw/*",
    )
    return tmap


def create_continuous_tmap(key: str):
    tmap = TensorMap(
        name=key,
        shape=(None, None),
        interpretation=Interpretation.CONTINUOUS,
        tensor_from_file=make_static_tensor_from_file(key),
        path_prefix="edw/*",
    )
    return tmap


def create_height_weight_tmap(key: str):
    tmap = TensorMap(
        name=key,
        shape=(None,),
        interpretation=Interpretation.CONTINUOUS,
        tensor_from_file=make_static_tensor_from_file(key),
        path_prefix="edw/*",
    )
    return tmap


def create_categorical_tmap(key: str, channel_map: Dict[str, Any]):
    tmap = TensorMap(
        name=key,
        shape=(None, 2),
        interpretation=Interpretation.CATEGORICAL,
        tensor_from_file=make_static_tensor_from_file(key),
        channel_map=channel_map,
        path_prefix="edw/*",
    )
    return tmap


def get_tmap(tm_name: str) -> Optional[TensorMap]:
    for language in DEFINED_TMAPS[f"{EDW_FILES['demo_file']['source'][4:]}_language"]:
        if tm_name.startswith(language):
            return create_language_tmap(language)
    for continuous in DEFINED_TMAPS[
        f"{EDW_FILES['demo_file']['source'][4:]}_continuous"
    ]:
        if tm_name.startswith(continuous):
            return create_continuous_tmap(continuous)
    for event in DEFINED_TMAPS[f"{EDW_FILES['demo_file']['source'][4:]}_event"]:
        if tm_name.startswith(event):
            return create_event_tmap(event)
    for categorical in DEFINED_TMAPS[
        f"{EDW_FILES['demo_file']['source'][4:]}_categorical"
    ]:
        if tm_name == categorical:
            return create_categorical_tmap(
                categorical,
                DEFINED_TMAPS[f"{EDW_FILES['demo_file']['source'][4:]}_categorical"][
                    categorical
                ],
            )
    if tm_name in ("height", "weight"):
        return create_height_weight_tmap(tm_name)
    if tm_name == "visits":
        return create_visits_tmap()
    if tm_name == "length_of_stay":
        return create_length_of_stay_tmap()
    if tm_name == "age":
        return create_age_tmap()
    if tm_name == "sex_double":
        return create_sex_double_tmap()

    return None
