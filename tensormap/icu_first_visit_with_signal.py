# Imports: standard library
from typing import Optional

# Imports: third party
import numpy as np

# Imports: first party
from definitions.icu import EDW_FILES
from tensormap.TensorMap import TensorMap, Interpretation, get_visits
from definitions.icu_tmap_list import DEFINED_TMAPS


def make_first_visit_tensor_from_file():
    def _tensor_from_file(tm, hd5, **kwargs):
        visits = get_visits(tm, hd5, **kwargs)
        for visit in visits:
            path = tm.path_prefix.replace("*", visit)
            if path in hd5:
                return np.array([visit])
        raise KeyError(f"No visit with {tm.name}.")

    return _tensor_from_file


def create_first_visit_tmap(tm_name: str, signal_name: str, data_type: str):
    tm = None
    name = signal_name.replace("|_", "")
    if tm_name == f"{name}_first_visit":
        tm = TensorMap(
            name=tm_name,
            shape=(1,),
            interpretation=Interpretation.CONTINUOUS,
            tensor_from_file=make_first_visit_tensor_from_file(),
            path_prefix=f"edw/*/{data_type}/{signal_name}/start_date",
        )
    return tm


def make_general_first_visit_tensor_from_subset(subset):
    def _tensor_from_file(tm, hd5, **kwargs):
        visits = get_visits(tm, hd5, **kwargs)

        for visit in visits:
            subpath = tm.path_prefix.replace("*", visit)
            for signal_name in subset:
                path = subpath.replace("{}", signal_name)
                if path in hd5:
                    return np.array([visit])
        raise KeyError(f"No visit with {tm.name}.")

    return _tensor_from_file


def create_arrest_first_visit_tmap(tm_name: str):
    events_names_list = ["code_start", "rapid_response_start"]
    tm = None
    if tm_name == "arrest_first_visit":
        tm = TensorMap(
            name=tm_name,
            shape=(1,),
            interpretation=Interpretation.CONTINUOUS,
            tensor_from_file=make_general_first_visit_tensor_from_subset(
                events_names_list,
            ),
            path_prefix="edw/*/events/{}/start_date",
        )
    return tm


def get_tmap(tm_name: str) -> Optional[TensorMap]:
    for data_type in ["events", "surgery", "procedures", "transfusions"]:
        for name in DEFINED_TMAPS[data_type]:
            if tm_name.startswith(name):
                return create_first_visit_tmap(tm_name, name, data_type)

    if tm_name.startswith("arrest_first_visit"):
        return create_arrest_first_visit_tmap(tm_name)

    return None
