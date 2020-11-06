# Imports: standard library
import os
import copy
from typing import Dict, Optional

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from ml4c3.metrics import weighted_crossentropy
from ml4c3.definitions.sts import STS_PREDICTION_DIR
from ml4c3.tensormap.TensorMap import (
    TensorMap,
    Interpretation,
    TimeSeriesOrder,
    id_from_filename,
    find_negative_label_and_channel,
)


def update_tmaps_model_predictions(
    tmap_name: str,
    tmaps: Dict[str, TensorMap],
) -> Dict[str, TensorMap]:
    """
    Create tensor maps to load predictions saved by model inference.
    Predictions should be generated using infer mode.

    For example, to load the death predictions from model v14 on bootstrap 0, use the tensor map:
        sts_death_predictions_v14_bootstrap_0
    """
    if "_predictions_" not in tmap_name:
        return tmaps

    base_name, model_name = tmap_name.split("_predictions_")
    model_name, bootstrap = model_name.split("_bootstrap_")
    bootstrap = bootstrap.split("_")[0]

    if base_name not in tmaps:
        raise ValueError(
            f"Base tmap {base_name} not in existing tmaps, cannot interpret predictions.",
        )

    base_tmap = tmaps[base_name]
    original_name = base_tmap.name
    if base_tmap.interpretation != Interpretation.CATEGORICAL:
        raise NotImplementedError(
            f"{base_tmap.interpretation} predictions cannot yet be loaded via a tmap.",
        )
    elif len(base_tmap.channel_map) != 2:
        raise NotImplementedError(f"Cannot load predictions for non binary task.")

    prediction_dir = os.path.join(STS_PREDICTION_DIR, model_name, bootstrap)

    prediction_files = [
        os.path.join(prediction_dir, prediction_file)
        for prediction_file in os.listdir(prediction_dir)
        if prediction_file.endswith(".csv") and "prediction" in prediction_file
    ]
    prediction_dfs = [
        pd.read_csv(os.path.join(prediction_dir, prediction_file))
        for prediction_file in prediction_files
    ]
    prediction_df = pd.concat(prediction_dfs).set_index("sample_id")
    predictions = prediction_df.to_dict(orient="index")

    negative_label, _ = find_negative_label_and_channel(base_tmap.channel_map)
    positive_label = [
        channel for channel in base_tmap.channel_map if channel != negative_label
    ][0]

    def tff(tm, hd5):
        mrn = id_from_filename(hd5.filename)
        tensor = np.array(
            [predictions[mrn][f"{original_name}_{positive_label}_predicted"]],
        )
        return tensor

    new_name = f"{base_name}_predictions_{model_name}_bootstrap_{bootstrap}"
    tmaps[new_name] = TensorMap(
        new_name,
        interpretation=Interpretation.CONTINUOUS,
        tensor_from_file=tff,
        shape=(1,),
    )
    return tmaps


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
            f"Base tmap {base_name} not in existing tmaps. Cannot modify weighted loss.",
        )
    weight = weight.split("_")[0]
    tmap = copy.deepcopy(tmaps[base_name])
    new_tmap_name = f"{base_name}_weighted_loss_{weight}"
    tmap.name = new_tmap_name
    tmap.loss = weighted_crossentropy([1.0, float(weight)], new_tmap_name)
    tmaps[new_tmap_name] = tmap
    return tmaps


random_date_selections = dict()


def update_tmaps_time_series(
    tmap_name: str,
    tmaps: Dict[str, TensorMap],
    time_series_limit: Optional[int] = None,
) -> Dict[str, TensorMap]:
    """Given the name of a needed tensor maps, e.g. ["ecg_age_newest"], and its base
    TMap, e.g. tmaps["ecg_age"], this function creates new tmap with the name of the
    needed tmap and the correct time_series_order and shape, but otherwise inherits
    properties from the base tmap. Next, updates new tmap to tmaps dict.
    """
    if "_newest" in tmap_name:
        base_split = "_newest"
        time_series_order = TimeSeriesOrder.NEWEST
    elif "_oldest" in tmap_name:
        base_split = "_oldest"
        time_series_order = TimeSeriesOrder.OLDEST
    elif "_random" in tmap_name:
        base_split = "_random"
        time_series_order = TimeSeriesOrder.RANDOM
    else:
        return tmaps
    base_name, _ = tmap_name.split(base_split)
    if base_name not in tmaps:
        raise ValueError(
            f"Base tmap {base_name} not in existing tmaps. Cannot modify time series.",
        )
    base_tmap = tmaps[base_name]

    def time_series_filter(hd5):
        _dates = base_tmap.time_series_filter(hd5)
        _dates = sorted(_dates)
        tsl = 1 if time_series_limit is None else time_series_limit
        if time_series_order == TimeSeriesOrder.RANDOM:
            if hd5.filename in random_date_selections:
                return random_date_selections[hd5.filename]
            if len(_dates) < tsl:
                tsl = len(_dates)
            _dates = np.random.choice(_dates, tsl, replace=False)
            random_date_selections[hd5.filename] = _dates
            return _dates
        elif time_series_order == TimeSeriesOrder.OLDEST:
            return _dates[:tsl]
        elif time_series_order == TimeSeriesOrder.NEWEST:
            return _dates[-tsl:]
        else:
            raise ValueError(f"Unknown time series ordering: {time_series_order}")

    new_tmap = copy.deepcopy(base_tmap)
    new_tmap_name = f"{base_name}{base_split}"
    new_tmap.name = new_tmap_name
    new_tmap.time_series_limit = time_series_limit
    new_tmap.time_series_filter = time_series_filter
    tmaps[new_tmap_name] = new_tmap
    return tmaps
