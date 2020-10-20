# Imports: standard library
from datetime import datetime

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from ml4c3.definitions.globals import TIMEZONE


def get_unix_timestamps(time_stamp: np.ndarray) -> np.ndarray:
    """
    Convert readable time stamps to unix time stamps.

    :param time_stamp: <np.ndarray> Array with all readable time stamps.
    :return: <np.ndarray> Array with Unix time stamps.
    """
    try:
        arr_timestamps = pd.to_datetime(time_stamp)
    except pd.errors.ParserError as error:
        raise ValueError("Array contains non datetime values") from error

    # Convert readable local timestamps in local seconds timestamps
    local_timestamps = (
        np.array(arr_timestamps, dtype=np.datetime64)
        - np.datetime64("1970-01-01T00:00:00")
    ) / np.timedelta64(1, "s")

    # Compute unix timestamp by checking local time shift
    def _get_time_shift(t_span):
        dt_span = datetime.utcfromtimestamp(t_span)
        offset = TIMEZONE.utcoffset(  # type: ignore
            dt_span, is_dst=True,
        ).total_seconds()
        return t_span - offset

    apply_time_shift = np.vectorize(_get_time_shift)

    if np.size(local_timestamps) > 0:
        unix_timestamps = apply_time_shift(local_timestamps)
    else:
        unix_timestamps = local_timestamps

    return unix_timestamps
