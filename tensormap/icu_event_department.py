# Imports: standard library
from typing import Optional

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from tensormap.TensorMap import (
    TensorMap,
    Interpretation,
    get_visits,
    get_local_timestamps,
)
from definitions.icu_tmap_list import DEFINED_TMAPS


def has_data(hd5, visit, data):
    return len(hd5[f"{data}/{visit}"].keys()) != 0


def make_event_department_tensor_from_file(both_sources: bool = False):
    def _tensor_from_file(tm, hd5):
        visits = get_visits(tm, hd5)
        shape = tm.shape
        tensor = np.zeros(shape)
        for visit in visits:
            path = tm.path_prefix.replace("*", visit)
            dates = hd5[path][()]
            if dates.shape[0] > 0:
                dates = get_local_timestamps(dates)
                for date in dates:
                    date = str(pd.to_datetime(date))
                    move_in = np.where(
                        hd5[f"edw/{visit}"].attrs["move_time"].astype(str) < date,
                    )[0][-1]
                    department = (
                        hd5[f"edw/{visit}"].attrs["department_nm"].astype(str)[move_in]
                    )
                    if both_sources and not has_data(hd5, visit, "bedmaster"):
                        continue
                    try:
                        tensor[tm.channel_map[department.lower()]] += 1
                    except KeyError:
                        tensor[tm.channel_map["other"]] += 1
        return tensor

    return _tensor_from_file


def create_event_department_tmap(tm_name: str, signal_name: str, data_type: str):
    tm = None
    name = signal_name.replace("|_", "")
    if tm_name == f"{name}_departments":
        tm = TensorMap(
            name=tm_name,
            interpretation=Interpretation.CONTINUOUS,
            tensor_from_file=make_event_department_tensor_from_file(),
            path_prefix=f"edw/*/{data_type}/{signal_name}/start_date",
            channel_map={
                "mgh blake 8 card sicu": 0,
                "mgh ellison 8 cardsurg": 1,
                "mgh ellison 9 med\\ccu": 2,
                "mgh ellison 10 stp dwn": 3,
                "mgh ellison11 card\\int": 4,
                "other": 5,
            },
        )
    elif tm_name == f"{name}_departments_with_bm":
        tm = TensorMap(
            name=tm_name,
            interpretation=Interpretation.CONTINUOUS,
            tensor_from_file=make_event_department_tensor_from_file(True),
            path_prefix=f"edw/*/{data_type}/{signal_name}/start_date",
            channel_map={
                "mgh blake 8 card sicu": 0,
                "mgh ellison 8 cardsurg": 1,
                "mgh ellison 9 med\\ccu": 2,
                "mgh ellison 10 stp dwn": 3,
                "mgh ellison11 card\\int": 4,
                "other": 5,
            },
        )
    return tm


def get_tmap(tm_name: str) -> Optional[TensorMap]:
    for data_type in ["surgery", "procedures", "transfusions", "events"]:
        for name in DEFINED_TMAPS[data_type]:
            if tm_name.startswith(name):
                return create_event_department_tmap(tm_name, name, data_type)
    return None
