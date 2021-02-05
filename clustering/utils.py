# Imports: standard library
import datetime

# Imports: third party
import numpy as np
from pytz import timezone

# Imports: first party
from definitions.icu_tmaps import DEFINED_TMAPS


class IncorrectSignalError(ValueError):
    """ Error raised when the signal has an unacceptable form"""


def get_signal_type(signal):
    for sig_type in DEFINED_TMAPS:
        if signal in DEFINED_TMAPS[sig_type]:
            return sig_type
    raise ValueError(f"Signal {signal} not recognized!")


def format_time(time, asunix=False):
    if isinstance(time, bytes):
        time = time.decode("utf-8")

    if isinstance(time, str):
        if time.replace(".", "").isnumeric():
            time = float(time)
        else:
            extra_decs = len(time.split(".")[-1]) - 6
            if extra_decs > 0:
                time = time[:-extra_decs]
            time = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")
    if isinstance(time, (int, float)):
        if np.isnan(time):
            return time

        time = datetime.datetime.fromtimestamp(time)

    time = timezone("America/New_York").localize(time)
    if asunix:
        time = time.timestamp()
    return time
