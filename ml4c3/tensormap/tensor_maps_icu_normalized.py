# Imports: standard library
from typing import Optional

# Imports: third party
import pandas as pd

# Imports: first party
from ml4c3.normalizer import MinMax, Standardize, RobustScaler, ZeroMeanStd1
from ml4c3.definitions.icu import ICU_TMAPS_SUMMARY_PATH
from ml4c3.tensormap.TensorMap import TensorMap
from ml4c3.tensormap.tensor_maps_icu_around_event import get_tmap as get_around_tmap
from ml4c3.tensormap.tensor_maps_icu_static_around_event import (
    get_tmap as get_static_around_tmap,
)


def get_tmap(tmap_name: str) -> Optional[TensorMap]:
    tm = None

    def _get_tmap(_tmap_name):
        _tm = None
        for _get in [
            get_around_tmap,
            get_static_around_tmap,
        ]:
            _tm = _get(_tmap_name)
            if _tm is not None:
                break
        return _tm

    if tmap_name.endswith("_standardized"):
        tm = _get_tmap(tmap_name.replace("_standardized", ""))
        if tm:
            feature_name = tm.name.replace("_mean_imputation", "")
            tm.name = tm.name + "_standardized"
            metadata = pd.read_csv(ICU_TMAPS_SUMMARY_PATH, index_col=0)
            normalizer = Standardize(
                mean=metadata.loc[feature_name]["mean"],
                std=metadata.loc[feature_name]["std"],
            )
            tm.normalizers = [normalizer]
    elif tmap_name.endswith("_robustscale"):
        tm = _get_tmap(tmap_name.replace("_robustscaler", ""))
        if tm:
            feature_name = tm.name.replace("_mean_imputation", "")
            tm.name = tm.name + "_robustscaler"
            metadata = pd.read_csv(ICU_TMAPS_SUMMARY_PATH, index_col=0)
            normalizer = RobustScaler(
                median=metadata.loc[feature_name]["median"],
                iqr=metadata.loc[feature_name]["iqr"],
            )
            tm.normalizers = [normalizer]
    elif tmap_name.endswith("_zeromean"):
        tm = _get_tmap(tmap_name.replace("_zeromean", ""))
        if tm:
            feature_name = tm.name.replace("_mean_imputation", "")
            tm.name = tm.name + "_zeromean"
            metadata = pd.read_csv(ICU_TMAPS_SUMMARY_PATH, index_col=0)
            normalizer = ZeroMeanStd1()
            tm.normalizers = [normalizer]
    elif tmap_name.endswith("_minmax"):
        tm = _get_tmap(tmap_name.replace("_minmax", ""))
        if tm:
            feature_name = tm.name.replace("_mean_imputation", "")
            tm.name = tm.name + "_minmax"
            metadata = pd.read_csv(ICU_TMAPS_SUMMARY_PATH, index_col=0)
            normalizer = MinMax(
                min=metadata.loc[feature_name]["min"],
                max=metadata.loc[feature_name]["max"],
            )
            tm.normalizers = [normalizer]

    return tm
