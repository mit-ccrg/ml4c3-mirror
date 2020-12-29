# Imports: standard library
from typing import Dict
from datetime import timedelta
from dataclasses import dataclass

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
# pylint: disable=unused-import, global-statement, too-many-function-args
from tensormap.TensorMap import create_tmap, update_tmaps
from visualizer.properties import SIGNAL_INTERPRETATION

TMAPS: Dict = {}


class TMapHelper:
    """
    Join information coming from tmaps in a more convenient format.
    """

    @staticmethod
    def list_signal(hd5, sig_id: str, visit_id: str, filtered: bool = False):
        signals_list_tmap = f"{sig_id}_signals"
        update_tmaps(signals_list_tmap, TMAPS)
        tensor = TMAPS[signals_list_tmap]
        return tensor.tensor_from_file(tensor, hd5, visit=visit_id, filtered=filtered)[
            0
        ]

    @staticmethod
    def get_signal(
        hd5,
        visit_id,
        signal_source,
        signal_name,
        no_time=False,
    ):

        if no_time:  # Plot by index
            value_key = SIGNAL_INTERPRETATION[signal_source]["value_key"]
            tmap = TMapHelper._get_tmap(signal_name, signal_source, value_key)
            tensor = tmap.tensor_from_file(tmap, hd5, visits=visit_id)
            values = tensor[0]
            time = None
        else:  # Plot by time
            value_key = SIGNAL_INTERPRETATION[signal_source]["timeseries_key"]
            tmap = TMapHelper._get_tmap(signal_name, signal_source, value_key)
            tensor = tmap.tensor_from_file(
                tmap,
                hd5,
                visits=visit_id,
                readable_dates=True,
            )
            values, time = tensor[0]

        # Units
        units_tmap = TMapHelper._get_tmap(signal_name, signal_source, "units")
        units = units_tmap.tensor_from_file(units_tmap, hd5, visits=visit_id)[0]

        # Name
        name_keys = SIGNAL_INTERPRETATION[signal_source]["name"]
        name = TMapHelper._get_name(
            name_keys,
            signal_name,
            signal_source,
            hd5,
            visit_id,
        )

        return Signal(name=name, values=values, time=time, units=units)

    @staticmethod
    def _get_tmap(signal_name, signal_source, value_key):
        signal_type = signal_source.split("_")[1]
        global TMAPS
        tensor_id = f"{signal_name}_{value_key}"
        try:
            TMAPS = update_tmaps(tensor_id, TMAPS)
            tmap = TMAPS[tensor_id]
        except ValueError:
            tmap = create_tmap(signal_name, signal_type, value_key)

        return tmap

    @staticmethod
    def _get_name(name_keys, signal_name, signal_source, hd5, visit_id):
        if not isinstance(name_keys, list):
            name_keys = [name_keys]
        name_list = []
        for key in name_keys:
            if key == "name":
                name_list.append(signal_name)
            else:
                global TMAPS
                name_tmap = TMapHelper._get_tmap(signal_name, signal_source, key)
                name = name_tmap.tensor_from_file(name_tmap, hd5, visits=visit_id)[0]
                name_list.append(name)

        name = " - ".join(name_list)
        return name

    @staticmethod
    def get_event(hd5, visit_id, event_source, event_name):
        time_keys = SIGNAL_INTERPRETATION[event_source]["time_key"]

        if not isinstance(time_keys, list):
            time_keys = [time_keys]

        # Name
        name_keys = SIGNAL_INTERPRETATION[event_source]["name"]
        name = TMapHelper._get_name(name_keys, event_name, event_source, hd5, visit_id)

        # Start date
        start_date_key = time_keys[0]
        st_tmap = TMapHelper._get_tmap(event_name, event_source, start_date_key)
        st_dates = st_tmap.tensor_from_file(
            st_tmap,
            hd5,
            visits=visit_id,
            readable_dates=True,
        )[0]

        # End date
        end_date_key = time_keys[1] if len(time_keys) == 2 else None
        if end_date_key == "end_date":
            end_tmap = TMapHelper._get_tmap(event_name, event_source, end_date_key)
            end_dates = end_tmap.tensor_from_file(
                end_tmap,
                hd5,
                visits=visit_id,
                readable_dates=True,
            )[0]
        elif end_date_key == "duration":
            duration_tmap = TMapHelper._get_tmap(event_name, event_source, end_date_key)
            durations = duration_tmap.tensor_from_file(
                duration_tmap,
                hd5,
                visits=visit_id,
                readable_dates=True,
            )[0]
            end_dates = np.array(
                [
                    st_date + timedelta(seconds=durations[idx])
                    for idx, st_date in enumerate(st_dates)
                ],
            )
        elif not end_date_key:
            end_dates = None
        else:
            raise ValueError(f"Unknown end date key {end_date_key}")

        return Event(name=name, start_dates=st_dates, end_dates=end_dates)

    @staticmethod
    def get_static_data(tmap_name: str, hd5):
        """
        Extracts data from the HD5 using tmaps. Then it formats it into a
        serializable format:

            * np.arrays -> Lists
            * bytes -> str
        Cleans the values (See _decode and _decode_categorical):
            * floats -> rounded to 3 decimals
            * datetimes -> rounded to seconds
            * Numeric categorical -> label

        :param tmap_name: <str> name of the tmap (and thus the data) to extract.
        :param hd5: <h5py.File> the file to extract data from.
        :param visit_id: <str> visit ID to look into the file
        :return: formatted value for the specified tmap
        """
        # Get the raw value
        update_tmaps(tmap_name, TMAPS)
        tmap = TMAPS[tmap_name]
        value = tmap.tensor_from_file(tmap, hd5)[0]

        # Decode categorical
        if tmap.is_categorical:
            value = TMapHelper.decode_categorical(value, tmap.channel_map)

        # Decode others
        is_date = "time" in tmap_name or "date" in tmap_name
        if isinstance(value, np.ndarray):
            value = list(map(lambda x: TMapHelper._decode(x, is_date), value))
        else:
            value = TMapHelper._decode(value, is_date)

        return value

    @staticmethod
    def _decode(element, to_datetime=False):
        """
        Clean a value:

        * bytes -> str
        * floats -> rounded to 3 decimals
        * datetimes -> rounded to seconds
        """
        if isinstance(element, bytes):
            element = element.decode("utf-8")

        if isinstance(element, np.float64):
            element = round(element, 3)

        if to_datetime:
            try:
                date = pd.to_datetime(element)
                element = date.strftime("%m/%d/%Y, %H:%M:%S")
            except ValueError:
                pass

        return element

    @staticmethod
    def decode_categorical(numeric_value, labels):
        """
        Convert a numeric categorical data into its label.
        """
        value = None

        for label, idx in labels.items():
            if numeric_value[idx]:
                value = label
                break

        if not value:
            raise ValueError(f"Couldn't decode categorical data with labels {labels}")

        return value

    @staticmethod
    def from_movement_to_event(mvmnt, static_data):
        mvmnt_dict = static_data["Movements"]["fields"]
        mvmnt_idx = mvmnt_dict["department_nm"].index(mvmnt)

        st_date = mvmnt_dict["move_time"][mvmnt_idx]

        if mvmnt_idx + 1 == len(mvmnt_dict["move_time"]):
            end_date = static_data["Admission"]["fields"]["end_date"]
            if isinstance(end_date, list):
                end_date = end_date[0]
        else:
            end_date = mvmnt_dict["move_time"][mvmnt_idx + 1]

        event = Event(
            name=mvmnt,
            start_dates=np.array([pd.to_datetime(st_date)]),
            end_dates=np.array([pd.to_datetime(end_date)]),
        )

        return event


@dataclass
class Signal:
    """
    Holds the information needed to plot a signal.
    """

    name: str
    values: np.ndarray
    time: np.ndarray
    units: str


@dataclass
class Event:
    """
    Holds the information needed to plot an event.
    """

    name: str
    start_dates: np.ndarray
    end_dates: np.ndarray
