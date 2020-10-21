# Imports: standard library
from typing import Optional

# Imports: third party
import numpy as np

# Imports: first party
from ml4c3.tensormap.TensorMap import TensorMap, Interpretation, get_visits
from ml4c3.definitions.icu_tmap_list import DEFINED_TMAPS


def make_list_signal_tensor_from_file(sig_type: str):
    def _tensor_from_file(tm, hd5, **kwargs):
        visits = get_visits(tm, hd5, **kwargs)

        # Get flag if it exists in kwargs, otherwise it's None
        flag_filtered = kwargs.get("filtered")

        if flag_filtered:
            max_size = max(
                len(
                    set(hd5[tm.path_prefix.replace("*", v)].keys()).intersection(
                        set(DEFINED_TMAPS[sig_type]),
                    ),
                )
                for v in visits
            )
        else:
            max_size = max(
                len(hd5[tm.path_prefix.replace("*", v)].keys()) for v in visits
            )

        shape = (len(visits), max_size)
        tensor = np.full(shape, "", object)
        for i, visit in enumerate(visits):
            path = tm.path_prefix.replace("*", visit)
            list_signals = list(hd5[path].keys())
            if flag_filtered:
                desired_signals = set(DEFINED_TMAPS[sig_type])
                common = set(list_signals).intersection(desired_signals)
                list_signals = sorted(common)
            tensor[i][: len(list_signals)] = list_signals
        return tensor

    return _tensor_from_file


def create_list_signals_tmap(sig_name: str, sig_type: str, root: str):
    tm = TensorMap(
        name=sig_name,
        shape=(None, None),
        interpretation=Interpretation.LANGUAGE,
        tensor_from_file=make_list_signal_tensor_from_file(sig_type),
        path_prefix=f"{root}/*/{sig_type}",
    )
    return tm


def get_tmap(tm_name: str) -> Optional[TensorMap]:
    for signal in DEFINED_TMAPS["bm_signals"]:
        signal_name = f"bm_{signal}_signals"
        if tm_name.startswith(signal_name):
            return create_list_signals_tmap(signal_name, signal, "bedmaster")

    for signal in DEFINED_TMAPS["edw_signals"]:
        signal_name = f"edw_{signal}_signals"
        if tm_name.startswith(signal_name):
            return create_list_signals_tmap(signal_name, signal, "edw")

    return None
