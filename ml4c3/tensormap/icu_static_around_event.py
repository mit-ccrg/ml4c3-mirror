# Imports: standard library
import re

# Imports: third party
import numpy as np

# Imports: first party
from ml4c3.utils import get_unix_timestamps
from ml4c3.tensormap.TensorMap import TensorMap, Interpretation
from ml4c3.tensormap.icu_events import get_tmap as GET_EVENT_TMAP
from ml4c3.tensormap.icu_static import admin_age_tensor_from_file
from ml4c3.tensormap.icu_first_visit_with_signal import get_tmap as GET_FIRST_VISIT_TMAP


def length_of_stay_event_tensor_from_file(visit_tm, event_tm, hrs_to_event, period):
    def _tensor_from_file(tm, hd5, **kwargs):
        base_path = tm.path_prefix
        visit = visit_tm.tensor_from_file(visit_tm, hd5, **kwargs)[0]
        event_time = event_tm.tensor_from_file(event_tm, hd5, visits=visit, **kwargs)[
            0
        ][0]
        path = base_path.replace("*", visit)
        admin_date = get_unix_timestamps(hd5[path].attrs["admin_date"])
        sign = -1 if period == "pre" else 1
        tensor = np.array(
            [event_time + sign * int(hrs_to_event) * 60 * 60 - admin_date],
        )
        return tensor

    return _tensor_from_file


def length_of_stay_event_double_tensor_from_file(
    visit_tm,
    event_tm,
    hrs_to_event_1,
    hrs_to_event_2,
    period,
):
    def _tensor_from_file(tm, hd5, **kwargs):
        base_path = tm.path_prefix
        visit = visit_tm.tensor_from_file(visit_tm, hd5, **kwargs)[0]
        event_time = event_tm.tensor_from_file(event_tm, hd5, visits=visit, **kwargs)[
            0
        ][0]
        path = base_path.replace("*", visit)
        admin_date = get_unix_timestamps(hd5[path].attrs["admin_date"])
        sign = -1 if period == "pre" else 1
        tensor = np.array(
            [
                np.array(
                    [event_time + sign * int(hrs_to_event_1) * 60 * 60 - admin_date],
                ),
                np.array(
                    [event_time + sign * int(hrs_to_event_2) * 60 * 60 - admin_date],
                ),
            ],
        )
        return tensor

    return _tensor_from_file


def admin_age_event_visit_tensor_from_file(visit_tm):
    def _tensor_from_file(tm, hd5, **kwargs):
        visit = visit_tm.tensor_from_file(visit_tm, hd5, **kwargs)[0]
        return admin_age_tensor_from_file(tm, hd5, visits=visit, **kwargs)

    return _tensor_from_file


def admin_age_event_visit_double_tensor_from_file(visit_tm):
    def _tensor_from_file(tm, hd5, **kwargs):
        visit = visit_tm.tensor_from_file(visit_tm, hd5, **kwargs)[0]
        tensor = admin_age_tensor_from_file(tm, hd5, visits=visit, **kwargs)
        return np.array([tensor, tensor])

    return _tensor_from_file


def get_tmap(tm_name: str):
    tm = None
    match = None

    if not match:
        pattern = re.compile(r"age_first_visit_(.*)_single$")
        match = pattern.findall(tm_name)
        if match:
            event_procedure = match[0]
            visit_tm = GET_FIRST_VISIT_TMAP(
                event_procedure.replace("end_date", "first_visit").replace(
                    "start_date",
                    "first_visit",
                ),
            )
            tm = TensorMap(
                name=tm_name,
                shape=(1,),
                interpretation=Interpretation.CONTINUOUS,
                tensor_from_file=admin_age_event_visit_tensor_from_file(visit_tm),
                path_prefix="edw/*",
            )
    if not match:
        pattern = re.compile(r"age_first_visit_(.*)_double$")
        match = pattern.findall(tm_name)
        if match:
            event_procedure = match[0]
            visit_tm = GET_FIRST_VISIT_TMAP(
                event_procedure.replace("end_date", "first_visit").replace(
                    "start_date",
                    "first_visit",
                ),
            )
            tm = TensorMap(
                name=tm_name,
                shape=(1,),
                interpretation=Interpretation.CONTINUOUS,
                tensor_from_file=admin_age_event_visit_double_tensor_from_file(
                    visit_tm,
                ),
                path_prefix="edw/*",
                time_series_limit=2,
            )
    if not match:
        pattern = re.compile(r"^length_of_stay_(\d+)_hrs_(pre|post)_(.*)$")
        match = pattern.findall(tm_name)
        if match:
            time, period, event_procedure = match[0]
            event_procedure_tm = GET_EVENT_TMAP(event_procedure)
            visit_tm = GET_FIRST_VISIT_TMAP(
                event_procedure.replace("end_date", "first_visit").replace(
                    "start_date",
                    "first_visit",
                ),
            )
            tm = TensorMap(
                name=tm_name,
                shape=(1,),
                interpretation=Interpretation.CONTINUOUS,
                tensor_from_file=length_of_stay_event_tensor_from_file(
                    visit_tm,
                    event_procedure_tm,
                    time,
                    period,
                ),
                path_prefix="edw/*",
            )

    if not match:
        pattern = re.compile(r"^length_of_stay_(\d+)_and_(\d+)_hrs_(pre|post)_(.*)$")
        match = pattern.findall(tm_name)
        if match:
            time_1, time_2, period, event_procedure = match[0]
            event_procedure_tm = GET_EVENT_TMAP(event_procedure)
            visit_tm = GET_FIRST_VISIT_TMAP(
                event_procedure.replace("end_date", "first_visit").replace(
                    "start_date",
                    "first_visit",
                ),
            )
            tm = TensorMap(
                name=tm_name,
                shape=(1,),
                interpretation=Interpretation.CONTINUOUS,
                tensor_from_file=length_of_stay_event_double_tensor_from_file(
                    visit_tm,
                    event_procedure_tm,
                    time_1,
                    time_2,
                    period,
                ),
                path_prefix="edw/*",
                time_series_limit=2,
            )
    return tm
