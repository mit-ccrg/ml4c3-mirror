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
from ml4c3.tensormap.icu_first_visit_with_signal import get_tmap as get_visit_tmap

# pylint: disable=pointless-statement


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


def get_tmap(tm_name: str) -> Optional[TensorMap]:
    for data_type in ["surgery", "procedures", "transfusions", "events"]:
        for name in DEFINED_TMAPS[data_type]:
            if tm_name.startswith(name):
                return create_event_tmap(tm_name, name, data_type)
    if tm_name.startswith("arrest"):
        return create_arrest_tmap(tm_name)
    return None
