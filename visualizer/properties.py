# Imports: standard library
import os

# Imports: third party
import yaml

# Imports: first party
from definitions.icu import VISUALIZER_PATH

LAYOUT = {}

# pylint: disable=global-statement


def load_config(user_files, defaults_file=None):
    if defaults_file is None:
        defaults_file = os.path.join(VISUALIZER_PATH, "graphs.yml")

    with open(defaults_file) as file:
        defaults = yaml.load(file, Loader=yaml.FullLoader)

    if user_files:
        with open(user_files) as file:
            user_def = yaml.load(file, Loader=yaml.FullLoader)
    else:
        user_def = {}

    def merge(user_opts, def_opts):
        for key in def_opts:
            if key in user_opts:
                def_opts[key] = user_opts[key]
        return def_opts

    global LAYOUT
    LAYOUT = {
        "graphs": user_def["graphs"] if "graphs" in user_def else defaults["graphs"],
        "statics": user_def["statics"]
        if "statics" in user_def
        else defaults["statics"],
        "options": merge(user_def["options"], defaults["options"])
        if "options" in user_def
        else defaults["options"],
    }

    dd_count = 0
    for graph in LAYOUT["graphs"].values():
        top_dd_num = len(graph["top_dropdowns"])
        side_dd_num = len(graph["side_dropdowns"])
        dd_count += top_dd_num
        dd_count += side_dd_num
    LAYOUT["options"]["dropdown_num"] = dd_count


MESSAGES = {
    "signal_too_large": "The signal that you want to plot  seems "
    "very  large.\n Consider cropping or "
    "downsampling  it. \n Do you want to plot it "
    "anyway?  Loading the signal might take some "
    "time.",
}

SIGNAL_INTERPRETATION = {
    "bedmaster_waveform": {
        "source": "bedmaster",
        "timeseries_key": "timeseries",
        "value_key": "value",
        "name": "name",
        "time_key": "time",
        "interpretation": "signal",
    },
    "bedmaster_vitals": {
        "source": "bedmaster",
        "name": "name",
        "timeseries_key": "timeseries",
        "value_key": "value",
        "time_key": "time",
        "interpretation": "signal",
    },
    "bedmaster_alarms": {
        "source": "bedmaster",
        "name": "name",
        "timeseries_key": ["name", "level"],
        "value_key": ["name", "level"],
        "time_key": ["init_date", "duration"],
        "interpretation": "event",
    },
    "edw_flowsheet": {
        "source": "edw",
        "name": "name",
        "timeseries_key": "timeseries",
        "value_key": "value",
        "time_key": "time",
        "interpretation": "signal",
    },
    "edw_labs": {
        "source": "edw",
        "name": "name",
        "timeseries_key": "timeseries",
        "value_key": "value",
        "time_key": "time",
        "interpretation": "signal",
    },
    "edw_med": {
        "source": "edw",
        "name": ["name", "route"],
        "timeseries_key": "timeseries",
        "value_key": "dose",
        "time_key": "time",
        "interpretation": "signal",
    },
    "edw_events": {
        "source": "edw",
        "name": "name",
        "timeseries_key": None,
        "value_key": None,
        "time_key": ["start_date"],
        "interpretation": "event",
    },
    "edw_surgery": {
        "source": "edw",
        "name": "name",
        "timeseries_key": None,
        "value_key": None,
        "time_key": ["start_date", "end_date"],
        "interpretation": "event",
    },
    "edw_transfusions": {
        "source": "edw",
        "name": "name",
        "timeseries_key": None,
        "value_key": None,
        "time_key": ["start_date", "end_date"],
        "interpretation": "event",
    },
    "edw_other_procs": {
        "source": "edw",
        "name": "name",
        "timeseries_key": None,
        "value_key": None,
        "time_key": ["start_date", "end_date"],
        "interpretation": "event",
    },
    "edw_procedures": {
        "source": "edw",
        "name": "name",
        "timeseries_key": None,
        "value_key": None,
        "time_key": ["start_date", "end_date"],
        "interpretation": "event",
    },
    "static": {
        "source": "edw",
        "name": "name",
        "timeseries_key": "department_nm",
        "value_key": "department_nm",
        "time_key": "move_time",
        "interpretation": "static",
        "tmap_generator": "memory",
    },
}
