# type: ignore
# Imports: standard library
from typing import Optional

# Imports: first party
from definitions.icu import ICU_TMAPS_METADATA
from ml4c3.normalizer import MinMax, ZScore, ZScorePopulation, RobustScalePopulation
from tensormap.TensorMap import TensorMap
from tensormap.icu_around_event import get_tmap as get_around_tmap
from tensormap.icu_static_around_event import get_tmap as get_static_around_tmap


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
            normalizer = ZScorePopulation(
                mean=ICU_TMAPS_METADATA[feature_name]["mean"],
                std=ICU_TMAPS_METADATA[feature_name]["std"],
            )
            tm.normalizers = [normalizer]
    elif tmap_name.endswith("_robustscale"):
        tm = _get_tmap(tmap_name.replace("_robustscaler", ""))
        if tm:
            feature_name = tm.name.replace("_mean_imputation", "")
            tm.name = tm.name + "_robustscaler"
            normalizer = RobustScalePopulation(
                median=ICU_TMAPS_METADATA[feature_name]["median"],
                iqr=ICU_TMAPS_METADATA[feature_name]["iqr"],
            )
            tm.normalizers = [normalizer]
    elif tmap_name.endswith("_zeromean"):
        tm = _get_tmap(tmap_name.replace("_zeromean", ""))
        if tm:
            feature_name = tm.name.replace("_mean_imputation", "")
            tm.name = tm.name + "_zeromean"
            normalizer = ZScore()
            tm.normalizers = [normalizer]
    elif tmap_name.endswith("_minmax"):
        tm = _get_tmap(tmap_name.replace("_minmax", ""))
        if tm:
            feature_name = tm.name.replace("_mean_imputation", "")
            tm.name = tm.name + "_minmax"
            normalizer = MinMax(
                min=ICU_TMAPS_METADATA[feature_name]["min"],
                max=ICU_TMAPS_METADATA[feature_name]["max"],
            )
            tm.normalizers = [normalizer]

    return tm