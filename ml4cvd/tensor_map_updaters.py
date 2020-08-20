# Imports: standard library
import re
import copy
from typing import Dict, Optional

# Imports: third party
import numpy as np

# Imports: first party
from ml4cvd.metrics import weighted_crossentropy
from ml4cvd.TensorMap import TensorMap, TimeSeriesOrder, id_from_filename
from ml4cvd.normalizer import Standardize
from ml4cvd.validators import validator_voltage_no_zero_padding
from ml4cvd.definitions import (
    ECG_PREFIX,
    ECG_REST_AMP_LEADS,
    ECG_REST_INDEPENDENT_LEADS,
)
from ml4cvd.tensor_maps_ecg import get_ecg_dates, make_voltage_tff, name2augmentations


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
    leads = ECG_REST_AMP_LEADS if "12_lead" in tmap_name else ECG_REST_INDEPENDENT_LEADS
    length = int(tmap_name.split("ecg_")[1].split("_")[0])
    exact = "exact" in tmap_name
    normalization = Standardize(mean=0, std=2000) if "std" in tmap_name else None
    augmentations = [
        augment_function
        for augment_option, augment_function in name2augmentations.items()
        if augment_option in tmap_name
    ]
    tmap = TensorMap(
        name=match_tmap_name,
        shape=(None, length, len(leads)),
        path_prefix=ECG_PREFIX,
        tensor_from_file=make_voltage_tff(exact_length=exact),
        normalization=normalization,
        channel_map=leads,
        time_series_limit=0,
        validator=validator_voltage_no_zero_padding,
        augmentations=augmentations,
    )
    tmaps[match_tmap_name] = tmap
    return tmaps


def update_tmaps_sts_match_ecg(
    tmap_name: str, tmaps: Dict[str, TensorMap],
) -> Dict[str, TensorMap]:
    """
    Modify STS tensor map's tensor_from_file function to repeat the tensor to match number of ECGs found from ECG tensor map(s).

    An ECG TMap must appear as an earlier argument to input or output tensors
    than the first "sts_match_ecg" TMap, e.g.

    --input_tensors
        ecg_age
        sts_death_match_ecg
    """
    if not tmap_name.startswith("sts_"):
        return tmaps
    elif "_match_ecg" not in tmap_name:
        return tmaps
    base_name = tmap_name.split("_match_ecg")[0]
    if base_name not in tmaps:
        raise ValueError(
            f"Base STS TMap {base_name} not in tmaps, cannot match ECG tmap.",
        )
    tmap = copy.deepcopy(tmaps[base_name])
    new_tmap_name = f"{base_name}_match_ecg"
    tmap.name = new_tmap_name
    tmap.shape = (None,) + tmap.shape
    old_tm = tmaps[base_name]

    def tff(tm, hd5, dependents={}):
        tensor = old_tm.tensor_from_file(old_tm, hd5, dependents)
        ecg_dates = get_ecg_dates.mrn_lookup[id_from_filename(hd5.filename)]
        return np.repeat(tensor[np.newaxis, ...], len(ecg_dates), axis=0)

    tmap.tensor_from_file = tff
    tmaps[new_tmap_name] = tmap
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
    from ml4cvd.tensor_maps_sts import date_interval_lookup  # isort:skip

    if "_sts" not in tmap_name:
        return tmaps
    base_name, _ = tmap_name.split("_sts")
    if base_name not in tmaps:
        raise ValueError(
            f"Base tmap {base_name} not in existing tmaps. Cannot modify STS window.",
        )
    tmap = copy.deepcopy(tmaps[base_name])
    new_tmap_name = f"{base_name}_sts"
    tmap.name = new_tmap_name
    tmap.time_series_lookup = date_interval_lookup
    tmaps[new_tmap_name] = tmap
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
    if time_series_limit is None:
        tmap.shape = tmap.static_shape
    tmap.time_series_limit = time_series_limit
    tmap.time_series_order = time_series_order
    tmap.metrics = None
    tmap.infer_metrics()
    tmaps[new_tmap_name] = tmap
    return tmaps
