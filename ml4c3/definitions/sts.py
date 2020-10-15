# Imports: standard library
import os
import socket


def _get_sts_data_path() -> str:
    """Get path to STS data depending on the machine hostname"""
    if "anduril" == socket.gethostname():
        path = "~/dropbox/sts-data"
    elif "mithril" == socket.gethostname():
        path = "~/dropbox/sts-data"
    elif "stultzlab" in socket.gethostname():
        path = "/storage/shared/sts-data-deid"
    else:
        path = "~/dropbox/sts-data"
    return os.path.expanduser(path)


STS_DATE_FORMAT = "%Y-%m-%d"
STS_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

STS_DATA_CSV = os.path.join(_get_sts_data_path(), "sts-mgh.csv")
STS_PREDICTION_DIR = os.path.expanduser("~/dropbox/sts-ecg/predictions")
