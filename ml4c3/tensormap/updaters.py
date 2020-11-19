# Imports: standard library
import copy
from typing import Dict, List, Union, Optional

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from ml4c3.metrics import weighted_crossentropy
from ml4c3.tensormap.TensorMap import Dates, TensorMap, PatientData


def update_tmaps_weighted_loss(
    tmap_name: str,
    tmaps: Dict[str, TensorMap],
) -> Dict[str, TensorMap]:
    """Make new tmap from base name, modifying loss weight"""
    if "_weighted_loss_" not in tmap_name:
        return tmaps
    base_name, weight = tmap_name.split("_weighted_loss_")
    if base_name not in tmaps:
        raise ValueError(
            f"Base tmap {base_name} not in existing tmaps. "
            f"Cannot modify weighted loss.",
        )
    weight = weight.split("_")[0]
    tmap = copy.deepcopy(tmaps[base_name])
    new_tmap_name = f"{base_name}_weighted_loss_{weight}"
    tmap.name = new_tmap_name
    tmap.loss = weighted_crossentropy([1.0, float(weight)], new_tmap_name)
    tmaps[new_tmap_name] = tmap
    return tmaps


random_date_selections: Dict[str, Union[List[str], pd.Series]] = dict()


def update_tmaps_time_series(
    tmap_name: str,
    tmaps: Dict[str, TensorMap],
    time_series_limit: Optional[int] = None,
) -> Dict[str, TensorMap]:
    """Given the name of a needed tensor maps, e.g. ["ecg_age_newest"], and its base
    TMap, e.g. tmaps["ecg_age"], this function creates new tmap with the name of the
    needed tmap and the correct shape, but otherwise inherits properties from the base
    tmap. Next, updates new tmap to tmaps dict.
    """
    if "_newest" in tmap_name:
        base_split = "_newest"
    elif "_oldest" in tmap_name:
        base_split = "_oldest"
    elif "_random" in tmap_name:
        base_split = "_random"
    else:
        return tmaps
    base_name, _ = tmap_name.split(base_split)
    if base_name not in tmaps:
        raise ValueError(
            f"Base tmap {base_name} not in existing tmaps. Cannot modify time series.",
        )
    base_tmap = tmaps[base_name]

    def updated_time_series_filter(data: PatientData) -> Dates:
        _dates = base_tmap.time_series_filter(data)
        _dates = (
            _dates.sort_values() if isinstance(_dates, pd.Series) else sorted(_dates)
        )
        tsl = 1 if time_series_limit is None else time_series_limit
        if "_random" in tmap_name:
            if data.id in random_date_selections:
                return random_date_selections[data.id]
            if len(_dates) < tsl:
                tsl = len(_dates)
            _dates = (
                _dates.sample(tsl, replace=False)
                if isinstance(_dates, pd.Series)
                else np.random.choice(_dates, tsl, replace=False)
            )
            random_date_selections[data.id] = _dates
            return _dates
        elif "_oldest" in tmap_name:
            return _dates[:tsl]
        elif "_newest" in tmap_name:
            return _dates[-tsl:]
        else:
            raise ValueError(f"Unknown time series ordering: {tmap_name}")

    new_tmap = copy.deepcopy(base_tmap)
    new_tmap_name = f"{base_name}{base_split}"
    new_tmap.name = new_tmap_name
    new_tmap.time_series_limit = time_series_limit
    new_tmap.time_series_filter = updated_time_series_filter
    tmaps[new_tmap_name] = new_tmap
    return tmaps
