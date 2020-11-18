# Imports: standard library
import re
from typing import Optional

# Imports: third party
import numpy as np

# Imports: first party
from ml4c3.tensormap.TensorMap import TensorMap, Interpretation
from ml4c3.tensormap.icu_around_event import get_tmap as get_around_tmap


def make_around_event_explore_tensor_from_file(
    around_tm_name,
):
    def _tensor_from_file(tm, hd5, **kwargs):
        tensor = np.array([])
        for feature in tm.channel_map:
            name = f"{around_tm_name}_{feature}"
            if feature == "mean_slope":
                name = name.replace("_value", "_timeseries")
            around_tm = get_around_tmap(name)
            try:
                value = around_tm.tensor_from_file(around_tm, hd5, **kwargs)
            except:
                value = np.nan
            tensor = np.append(tensor, value)
        return tensor

    return _tensor_from_file


def get_tmap(tmap_name: str) -> Optional[TensorMap]:
    match = None

    pattern = re.compile(
        r"^(.*)_(\d+)_to_(\d+)_hrs_(pre|post)_(.*)_explore$",
    )
    match = pattern.findall(tmap_name)
    if match:
        make_tensor_from_file = make_around_event_explore_tensor_from_file(
            tmap_name.replace("_explore", ""),
        )
        channel_map = {
            "min": 0,
            "max": 1,
            "mean": 2,
            "std": 3,
            "first": 4,
            "last": 5,
            "count": 6,
            # "mean_slope": 7,
            # "mean_crossing_rate": 8,
        }
        path_prefix = get_around_tmap(tmap_name.replace("_explore", "")).path_prefix
    if not match:
        return None

    return TensorMap(
        name=tmap_name,
        tensor_from_file=make_tensor_from_file,
        channel_map=channel_map,
        path_prefix=path_prefix,
        interpretation=Interpretation.CONTINUOUS,
    )
