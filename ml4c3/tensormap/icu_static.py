# Imports: standard library
import logging
from typing import Dict, Callable, Optional

# Imports: third party
import h5py
import numpy as np

# Imports: first party
from ml4c3.utils import get_unix_timestamps
from ml4c3.validators import (
    validator_no_empty,
    validator_no_negative,
    validator_not_all_zero,
)
from ml4c3.definitions.icu import EDW_FILES, EDW_PREFIX
from ml4c3.tensormap.TensorMap import TensorMap, Interpretation, id_from_filename
from ml4c3.definitions.icu_tmap_list import DEFINED_TMAPS

# pylint: disable=too-many-return-statements


def visit_tensor_from_file(tm: TensorMap, hd5: h5py.File) -> np.ndarray:
    return np.array(tm.time_series_filter(hd5))[:, None]


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


def mrn_tensor_from_file(tm: TensorMap, hd5: h5py.File) -> np.ndarray:
    mrn = str(id_from_filename(hd5.filename))
    return np.array([mrn] * len(tm.time_series_filter(hd5)))[:, None]


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


def length_of_stay_tensor_from_file(tm: TensorMap, hd5: h5py.File) -> np.ndarray:
    visits = tm.time_series_filter(hd5)
    shape = (len(visits),) + tm.shape
    tensor = np.zeros(shape)

    for i, visit in enumerate(visits):
        try:
            path = f"{tm.path_prefix}/{visit}"
            end_date = get_unix_timestamps(hd5[path].attrs["end_date"])
            start_date = get_unix_timestamps(hd5[path].attrs["admin_date"])
            tensor[i] = (end_date - start_date) / 60 / 60
        except (ValueError, KeyError) as e:
            logging.debug(f"Could not get length of stay from {hd5.filename}/{visit}")
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


def admin_age_tensor_from_file(tm: TensorMap, hd5: h5py.File) -> np.ndarray:
    visits = tm.time_series_filter(hd5)
    shape = (len(visits),) + tm.shape
    tensor = np.zeros(shape)

    for i, visit in enumerate(visits):
        try:
            path = f"{tm.path_prefix}/{visit}"
            admit_date = get_unix_timestamps(hd5[path].attrs["admin_date"])
            birth_date = get_unix_timestamps(hd5[path].attrs["birth_date"])
            age = admit_date - birth_date
            tensor[i] = age / 60 / 60 / 24 / 365
        except (ValueError, KeyError) as e:
            logging.debug(f"Could not get age from {hd5.filename}/{visit}")
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


def sex_double_tensor_from_file(tm: TensorMap, hd5: h5py.File) -> np.ndarray:
    visit = tm.time_series_filter(hd5)[0]
    shape = (2,) + tm.shape
    tensor = np.zeros(shape)
    path = f"{tm.path_prefix}/{visit}"
    data = hd5[path].attrs["sex"]
    tensor[:, tm.channel_map[data.lower()]] = np.array([1, 1])

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
    def _tensor_from_file(tm: TensorMap, hd5: h5py.File) -> np.ndarray:
        visits = tm.time_series_filter(hd5)
        temp = None
        finalize = False
        if tm.is_timeseries:
            temp = [hd5[f"{tm.path_prefix}/{v}"].attrs[key] for v in visits]
            max_len = max(map(len, temp))
            shape = (len(visits), max_len)
        else:
            shape = (len(visits),) + tm.shape

        if tm.is_categorical or tm.is_continuous:
            tensor = np.zeros(shape)
        elif tm.is_language or tm.is_event:
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
                data = hd5[path].attrs[key] if temp is None else temp[i]
                if tm.channel_map:
                    tensor[i, tm.channel_map[data.lower()]] = 1
                else:
                    tensor[i] = data
            except (ValueError, KeyError) as e:
                logging.debug(f"Error getting {key} from {hd5.filename}/{visit}")
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


def create_event_tmap(key: str) -> TensorMap:
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
    source = "static"

    if tm_name in DEFINED_TMAPS[f"{source}_language"]:
        return create_language_tmap(tm_name)
    elif tm_name in DEFINED_TMAPS[f"{source}_continuous"]:
        return create_continuous_tmap(tm_name)
    elif tm_name in DEFINED_TMAPS[f"{source}_categorical"]:
        return create_categorical_tmap(
            tm_name,
            DEFINED_TMAPS[f"{source}_categorical"][tm_name],
        )
    elif tm_name in DEFINED_TMAPS[f"{source}_event"]:
        return create_event_tmap(tm_name)
    elif tm_name in DEFINED_TMAPS[f"{source}_timeseries"]:
        return create_timeseries_tmap(tm_name)
    elif tm_name == "mrn":
        return create_mrn_tmap()
    elif tm_name == "visits":
        return create_visits_tmap()
    elif tm_name == "length_of_stay":
        return create_length_of_stay_tmap()
    elif tm_name == "age":
        return create_age_tmap()
    elif tm_name == "sex_double":
        return create_sex_double_tmap()

    return None
