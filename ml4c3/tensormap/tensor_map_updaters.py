# Imports: standard library
import os
import re
import copy
from typing import Dict, Optional

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from ml4c3.metrics import weighted_crossentropy
from ml4c3.normalizer import Standardize
from ml4c3.validators import validator_voltage_no_zero_padding
from ml4c3.definitions.ecg import (
    ECG_PREFIX,
    ECG_REST_LEADS_ALL,
    ECG_REST_LEADS_INDEPENDENT,
)
from ml4c3.definitions.sts import STS_PREDICTION_DIR
from ml4c3.tensormap.TensorMap import (
    TensorMap,
    Interpretation,
    TimeSeriesOrder,
    id_from_filename,
    find_negative_label_and_channel,
)
from ml4c3.tensormap.tensor_maps_ecg import (
    get_ecg_dates,
    name2augmenters,
    make_voltage_tff,
)


def update_tmaps_ecg_voltage(
    tmap_name: str, tmaps: Dict[str, TensorMap],
) -> Dict[str, TensorMap]:
    """
    Generates ECG voltage TMaps that are given by the name format:
        [12_lead_]ecg_{length}[_exact][_std][_augmentations]

    Required:
        length: the number of samples present in each lead.

    Optional:
        12_lead: use the 12 clinical leads.
        exact: only return voltages when raw data has exactly {length} samples in each lead.
        std: standardize voltages using mean = 0, std = 2000.
        augmentations: apply crop, noise, and warp transformations to voltages.

    Examples:
        valid: ecg_2500_exact_std
        valid: 12_lead_ecg_625_crop_warp
        invalid: ecg_2500_noise_std

    Note: if additional modifiers are present after the name format, e.g.
        ecg_2500_std_newest_sts, the matched part of the tmap name, e.g.
        ecg_2500_std, will be used to construct a tmap and save it to the dict.
        Later, a function will find the tmap name ending in _newest, and modify the
        tmap appropriately.
    """
    voltage_tm_pattern = re.compile(
        r"^(12_lead_)?ecg_\d+(_exact)?(_std)?(_warp|_crop|_noise)*",
    )
    match = voltage_tm_pattern.match(tmap_name)
    if match is None:
        return tmaps

    # Isolate matching components of tmap name and build it
    match_tmap_name = match[0]
    leads = ECG_REST_LEADS_ALL if "12_lead" in tmap_name else ECG_REST_LEADS_INDEPENDENT
    length = int(tmap_name.split("ecg_")[1].split("_")[0])
    exact = "exact" in tmap_name
    normalizer = Standardize(mean=0, std=2000) if "std" in tmap_name else None
    augmenters = [
        augment_function
        for augment_option, augment_function in name2augmenters.items()
        if augment_option in tmap_name
    ]
    tmap = TensorMap(
        name=match_tmap_name,
        shape=(length, len(leads)),
        path_prefix=ECG_PREFIX,
        tensor_from_file=make_voltage_tff(exact_length=exact),
        normalizers=normalizer,
        channel_map=leads,
        time_series_limit=0,
        validators=validator_voltage_no_zero_padding,
        augmenters=augmenters,
    )
    tmaps[match_tmap_name] = tmap
    return tmaps


def update_tmaps_model_predictions(
    tmap_name: str, tmaps: Dict[str, TensorMap],
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
    tmap_name: str, tmaps: Dict[str, TensorMap],
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


def update_tmaps_sts_window(
    tmap_name: str, tmaps: Dict[str, TensorMap],
) -> Dict[str, TensorMap]:
    """Make new tmap from base name, making conditional on surgery date"""

    suffixes = ["preop", "postop"]
    for suffix in suffixes:
        if suffix in tmap_name:
            # fmt: off
            from ml4c3.tensormap.tensor_maps_sts import date_interval_lookup  # isort:skip
            # fmt: on

            base_name, _ = tmap_name.split(f"_{suffix}")
            if base_name not in tmaps:
                raise ValueError(
                    f"Base tmap {base_name} not in existing tmaps. Cannot modify STS {suffix} window.",
                )
            tmap = copy.deepcopy(tmaps[base_name])
            new_tmap_name = f"{base_name}_{suffix}"
            tmap.name = new_tmap_name
            tmap.time_series_lookup = date_interval_lookup[suffix]
            tmaps[new_tmap_name] = tmap
            break
    return tmaps


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
    tmap = copy.deepcopy(tmaps[base_name])
    new_tmap_name = f"{base_name}{base_split}"
    tmap.name = new_tmap_name
    tmap.time_series_limit = time_series_limit
    tmap.time_series_order = time_series_order
    tmap.metrics = None
    tmaps[new_tmap_name] = tmap
    return tmaps
