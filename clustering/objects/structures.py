# Imports: standard library
import os
import pickle
import pprint
import random
import string
import logging
import datetime
from typing import Dict, List, Union
from collections import OrderedDict

# Imports: third party
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as log_progress

# Imports: first party
from clustering.utils import IncorrectSignalError, format_time, get_signal_type
from clustering.globals import METADATA_MEDS
from tensormap.TensorMap import Interpretation, update_tmaps
from clustering.objects.modifiers import (
    Padder,
    Cluster,
    Normalizer,
    Downsampler,
    DistanceMetric,
    OutlierRemover,
    FeatureExtractor,
    DimensionsReducer,
)

# pylint: disable=global-statement

TMAPS: Dict = {}


def _get_tmap(key):
    global TMAPS
    if key not in TMAPS:
        TMAPS = update_tmaps(key, TMAPS)
    return TMAPS[key]


class HD5File(h5py.File):
    """
    Wrapper wround an h5py File containing methods specific to get information
    from tensorized hd5 files.
    """

    def __init__(self, path):
        super().__init__(path, "r")

    @staticmethod
    def get_full_type_path(sig_type, visit=None):
        """
        Returns the full path of a signal type.
        >>> HD5File.get_full_type_path("flowsheet", visit="123")
        edw/123/flowsheet
        """
        if sig_type in ["flowsheet", "labs", "med", "surgery", "transfusions"]:
            path = f"edw/*/{sig_type}"
        elif sig_type in ["vitals", "waveform"]:
            path = f"bedmaster/*/{sig_type}"
        else:
            raise ValueError(f"Invalid signal type: {sig_type}")

        if visit:
            path = path.replace("*", visit)

        return path

    @staticmethod
    def get_full_signal_path(signal, visit=None):
        """
        Returns the full path of a signal.
        >>> HD5File.get_full_signal_path("pa2s", visit="123")
        bm/123/vitals/pa2s
        """
        _, sig_type = get_signal_type(signal)
        type_path = HD5File.get_full_type_path(sig_type, visit)
        sig_path = f"{type_path}/{signal}"
        return sig_path

    @property
    def visits(self):
        """Returns all the visits on the file"""
        visits = set()
        for source in self.sources:
            visits = visits | set(self[source])
        return visits

    @property
    def sources(self):
        """Returns all the sources containing at least a visit"""
        sources = []
        for source in self:
            if self.has_source(source):
                sources.append(source)
        return sources

    @property
    def mrn(self):
        """Returns the cleaned MRN of the file"""
        file_name = self.filename.split("/")[-1]
        file_name = file_name.split(".")[0]
        return file_name

    def signal_types(self, visit=None):
        """
        Returns all the signal types on the hd5. If `visit` is set, it returns
        only the signal types for that visit.
        """
        sig_types = set()
        visits = self.visits if not visit else [visit]
        for source in self.sources:
            for visit_id in visits:
                sig_types = sig_types | set(self[source][visit_id])
        return sig_types

    def has_source(self, source, visit=None):
        """
        Evaluates if the file contains a source. If `visit` is set,
        it evaluates if the visit contains that source of data
        """
        has_source = source in self
        if has_source:
            try:
                visits = list(self[source]) if not visit else [visit]
            except KeyError:
                logging.info(
                    f"File's {self.filename} {source} directory "
                    f"is corrupted and can't be accessed.",
                )
                return False
            for visit_id in visits:
                has_visit = visit_id in self[source]
                if has_visit:
                    has_signals = bool(list(self[source][visit_id]))
                    has_attrs = bool(list(self[source][visit_id]))
                    source_is_full = has_attrs or has_signals
                    if source_is_full:
                        return True
        return False

    def has_sig_type(self, sig_type, visit=None):
        """
        Evaluates if the file contains a signal type. If `visit` is set,
        it evaluates only if the visit contains that signal type.
        """
        type_path = self.get_full_type_path(sig_type)
        visits = [visit] if visit else self.visits
        has_sig = any([type_path.replace("*", visit) in self for visit in visits])
        return has_sig

    def has_signal(self, signal, visit=None):
        """
        Evaluates if the file contains a signal. If `visit` is set,
        it evaluates only if the visit contains that signal type.
        """
        type_path = self.get_full_signal_path(signal)
        visits = [visit] if visit else self.visits
        has_sig = any([type_path.replace("*", visit) in self for visit in visits])
        return has_sig

    def get_blk08_duration(
        self,
        visit,
        max_len=None,
        asunix=False,
        only_first=True,
    ) -> List[Dict]:
        """
        Returns a list with all the blake08 stay of a patient. Stays are represented
        as a dictionary in the form {"start_date": start_date, "end_date": end_date}.
        Dates are datetimes in UTC time zone or ints that represent the unix timestamp.

        Note:
        -   if multiple blk08 entrances are concatenated, the algorithm authomatically
            merges them into one. For example: [ELS09, BLK08, BlK08, ELS09, BLK08]
            becomes [ELS09, BLK08, ELS09, BK08].
        -   the patient may return an empty dict. There are three reasons for that:
            1. The patient has not gone through blake08
            2. The patient has gone through blk08 but the start time or the end time
            is unknown (time on the movement list is nan)
            3. The patient has gone through blk08 but the end time is unknown because
            it's blk08 is the last movement and the discharge time is unknown (a nan).

        :param visit: <str> visit to get the BLK08 duration from.
        :param max_len: <int> If set, the BLK08 stay will be cropped to the first
                        <max_len> hours. If <max_len> is longer than the BLK08 stay,
                        the return value will be  limited to the blk08 stay duration.
        :param asunix: <bool> if True, the start and end times are represented as
                       unix timestamps ints.
        :param only_first: <bool> if True, only the first BLK08 stay will be returned
        """

        def _get_end_time(start_idx):
            for rel_idx, _ in enumerate(movements_list[start_idx:]):
                abs_idx = rel_idx + start_idx
                if abs_idx not in blk08_idx:
                    end_time = movements_list[abs_idx]
                    return end_time
            tmap = _get_tmap("end_date")
            end_time = tmap.tensor_from_file(tmap, self, visits=visit)[0][0]
            if end_time == "nan":
                end_time = np.nan
            return end_time

        tmap = _get_tmap("department_nm")
        department_list = tmap.tensor_from_file(tmap, self, visits=visit)[0]
        blk08_idx = np.where(department_list == "MGH BLAKE 8 CARD SICU")[0]
        tmap = _get_tmap("move_time")
        movements_list = tmap.tensor_from_file(tmap, self, visits=visit)[0]

        blk08_stays = []
        blk08_starts = [
            blk08_idx[i]
            for i, _ in enumerate(blk08_idx)
            if blk08_idx[i - 1] != blk08_idx[i] - 1
        ]
        for idx in blk08_starts:
            start_time = movements_list[idx]
            end_time = _get_end_time(start_idx=idx)

            if pd.isna(start_time) or pd.isna(end_time):
                logging.info(
                    f"File {self.mrn} on visit {visit} has unknown start or"
                    f"discharge from BLK08",
                )
                continue

            start_time = format_time(start_time)
            end_time = format_time(end_time)

            if max_len:
                end_time = min(
                    start_time + datetime.timedelta(hours=max_len),
                    end_time,
                )

            if asunix:
                start_time = start_time.timestamp()
                end_time = end_time.timestamp()

            blk08_stays.append({"start": start_time, "end": end_time})
            if only_first:
                return blk08_stays

        return blk08_stays

    def find_signal(self, signal, visit, max_length=None):
        """
        Looks for a signal on the given file and visit and during the BLK08 stay
        (cropped by max_length if set).

        :param signal: signal to search
        :param visit: visit to search
        :param output: 3 options:
            - existence: output is a boolean that confirms existence of the signal
            - length: output is the duration of the signal in seconds
            - percentage: output is the percentage of existence of the signal with
                          respect to the BLK08 stay.
        :param max_length: crop the BLK08 stay to a maximum of <max_lenth> hours.
        """
        sig_path = HD5File.get_full_signal_path(signal=signal, visit=visit)
        if sig_path not in self:
            logging.info(f"File {self.mrn} has no signal {signal}")
            return {"_existence": False}

        duration = self.get_blk08_duration(
            visit,
            max_length,
            asunix=True,
            only_first=True,
        )[0]
        blk08_start, blk08_end = duration["start"], duration["end"]

        time = self[sig_path]["time"]
        if not time.size:
            logging.info(f"Signal {signal} exists on {self.mrn} but it's empty.")
            return {"_existence": False}
        valid_idx = np.where((time[()] >= blk08_start) & (time[()] <= blk08_end))[0]
        if not valid_idx.size:
            logging.info(
                f"Signal {signal} exists on {self.mrn} but is out of BLK08 range.",
            )
            return {"_existence": False}

        start_time, end_time = time[valid_idx[0]], time[valid_idx[-1]]
        unfilled = np.diff(
            np.concatenate(([blk08_start], time[valid_idx], [blk08_end])),
        )
        max_unfilled = unfilled.max()

        valid_time = time[valid_idx]
        time_diff = np.diff(valid_time)

        is_nm = time_diff < 0
        non_monotonicities = time_diff[is_nm]

        overlap_length = end_time - start_time
        overlap_percent = (end_time - start_time) / (blk08_end - blk08_start) * 100
        non_monotonicities_num = non_monotonicities.size
        if non_monotonicities_num:
            max_non_monotonicity = abs(non_monotonicities).max()
            non_monotonicities_sum = abs(non_monotonicities).sum()
            mask = np.concatenate(([False], is_nm, [False]))
            changes = np.nonzero(mask[1:] != mask[:-1])[0]
            durations = changes[1::2] - changes[::2]
            longest_non_mon = durations.max()
        else:
            max_non_monotonicity = 0
            non_monotonicities_sum = 0
            longest_non_mon = 0

        results = {
            "_existence": True,
            "_overlap_length": overlap_length,
            "_overlap_percent": overlap_percent,
            "_start_time": start_time,
            "_end_time": end_time,
            "max_unfilled_(s)": max_unfilled,
            "non_monotonicities_(#)": non_monotonicities_num,
            "max_non_monotonicity_(s)": max_non_monotonicity,
            "non_monotonicities_sum_(s)": non_monotonicities_sum,
            "longest_consecutive_non_mono_(#)": longest_non_mon,
        }
        return results

    def get_signal(
        self,
        signal_name,
        visit,
        start_time=0,
        end_time=np.inf,
        allow_empty=False,
        **kwargs,
    ):
        """
        Extracts the given signal from the file on a Signal object.
        The signal will be cropped to the start_time-end_time range. If no range is
        set, the full signal will be returned. If `allow_empty` is set and the signal
        is not found, instead of raising an error, an empty signal
        (value=0 and time=<start_time>) will be returned.
        """
        _, sig_type = get_signal_type(signal_name)

        valid_idx = np.array([])
        if self.has_signal(signal_name, visit=visit):
            try:
                tmap = _get_tmap(f"{signal_name}_timeseries")
                tmap.interpretation = Interpretation.TIMESERIES
                values, time = tmap.tensor_from_file(
                    tmap, self, visits=visit, **kwargs
                )[0]

                blk08_idx = np.where((time >= start_time) & (time <= end_time))[0]
                time = time[blk08_idx]
                values = values[blk08_idx]

                valid_idx = np.where(~np.isnan(values))[0]

            except ValueError as err:
                logging.warning(
                    f"Unexpected format on signal {signal_name} of "
                    f"file {self.mrn}: \n {err}",
                )

        if valid_idx.size == 0:
            if allow_empty:
                signal = Signal(
                    name=signal_name,
                    values=np.array([0]),
                    time=np.array([start_time]),
                    stype=sig_type,
                )
                return signal
            logging.info(f"Signal {signal_name} not found on BLK08 on {self.mrn}")
            raise ValueError(f"Signal {signal_name} not found!")

        tmap = _get_tmap(f"{signal_name}_units")
        units = tmap.tensor_from_file(tmap, self, visits=visit)[0]

        time = time[valid_idx]
        values = values[valid_idx]
        signal = Signal(
            name=signal_name,
            values=values,
            time=time,
            stype=sig_type,
            units=units,
        )

        try:
            tmap = _get_tmap(f"{signal}_sample_freq")
            sf = tmap.tensor_from_file(tmap, self, visits=visit)[0]
            if sf.size > 1:
                logging.warning(
                    f"Signal {signal} has multiple sample freqs "
                    f"but only the first will be written",
                )
            signal.sample_freq = sf[0][0]
        except:
            signal.sample_freq = None

        return signal

    def has_reoperation(self, visit, start_time, end_time):
        """
        Boolean determining if the patient had a reoperation between both timestamps.
        """
        if "surgery" not in self["edw"][visit]:
            return False
        for surgery_name in self["edw"][visit]["surgery"].keys():
            surgery = self["edw"][visit]["surgery"][surgery_name]
            surg_start = surgery["start_date"][0]
            if start_time < surg_start <= end_time:
                return True
        return False

    def get_patient_metadata(self, visit, max_len=None):
        """
        Extracts metadata from the patient.
        """
        static_dir = self["edw"][visit].attrs
        patient_info = {}

        def _row_to_dict(string_row):
            """
            Convert string table rows in dicts for easier conversion to df.
            """
            decoded_row = {}
            columns = string_row.split(";")
            for column in columns:
                split_column = column.split(":")
                if len(split_column) == 2:
                    col_name, col_value = split_column
                else:
                    col_name = split_column[0]
                    col_value = "".join(split_column[1:])
                decoded_row[col_name] = col_value
            return decoded_row

        discharge_date = format_time(static_dir["end_date"], asunix=True)
        admin_date = format_time(static_dir["end_date"], asunix=True)
        birth_date = format_time(static_dir["birth_date"], asunix=True)

        patient_info["hospital_stay (days)"] = (discharge_date - admin_date) / (
            3600 * 24
        )
        patient_info["age (y)"] = (admin_date - birth_date) / (3600 * 24 * 325.25)
        patient_info["sex (%)"] = static_dir["sex"]
        patient_info["race"] = static_dir["race"]
        patient_info["outcome"] = static_dir["end_stay_type"]
        patient_info["height (m)"] = static_dir["height"]
        patient_info["weight (lbs)"] = static_dir["weight"]

        if "tobacco_hist" in static_dir:
            tobacco = _row_to_dict(static_dir["tobacco_hist"])
            patient_info["tobacco"] = tobacco["STATUS"].strip()
        if "alcohol_hist" in static_dir:
            tobacco = _row_to_dict(static_dir["alcohol_hist"])
            patient_info["alcohol"] = tobacco["STATUS"].strip()

        blk08 = self.get_blk08_duration(visit, None, only_first=True, asunix=True)[0]
        blk08_start, blk08_end = blk08["start"], blk08["end"]
        blk08_duration = (blk08_end - blk08_start) / (3600 * 24)

        patient_info["reoperation (%)"] = self.has_reoperation(
            visit,
            blk08_start,
            blk08_end,
        )
        patient_info["BLK08 stay (d)"] = blk08_duration
        patient_info["BLK08 start"] = format_time(blk08_start)
        patient_info["BLK08 end"] = format_time(blk08_end)

        duration = self.get_blk08_duration(
            visit,
            max_len,
            only_first=True,
            asunix=True,
        )[0]
        sig_start, sig_end = duration["start"], duration["end"]
        patient_info["Start time"] = format_time(sig_start)
        patient_info["End time"] = format_time(sig_end)
        patient_info["Period studied (h)"] = max_len if max_len else blk08_duration * 24

        for nickname, signal_name in METADATA_MEDS.items():
            signal = self.get_signal(
                signal_name,
                visit,
                start_time=sig_start,
                end_time=sig_end,
                allow_empty=True,
            )
            local_dose = signal.values.sum()
            signal = self.get_signal(
                signal_name,
                visit,
                start_time=blk08_start,
                end_time=blk08_end,
                allow_empty=True,
            )
            blk08_dose = signal.values.sum()

            patient_info[f"{nickname} {max_len}h period"] = local_dose
            patient_info[f"{nickname} BLK08 total"] = blk08_dose

        return patient_info

    def extract_patient(self, visit, signals, max_length=None):
        """
        Extracts the input signals from a visit along with metadata from that visit,
        and returns a Patient instance with all the information.

        Only the BLK08 part of the signals will be included. Signals can be further
        cropped to <max_length> if this parameter is set.
        """
        blk08stays = self.get_blk08_duration(
            visit,
            max_length,
            asunix=True,
            only_first=True,
        )[0]
        blk08_start, blk08_end = blk08stays["start"], blk08stays["end"]

        extracted_signals = []
        for signal_name in signals:
            _, signal_type = get_signal_type(signal_name)
            if signal_type == "waveform":
                kwargs = {"interpolation": "complete_no_nans"}
            else:
                kwargs = {}
            allow_empty = signal_type == "meds"
            try:
                signal = self.get_signal(
                    signal_name,
                    visit=visit,
                    start_time=blk08_start,
                    end_time=blk08_end,
                    allow_empty=allow_empty,
                    **kwargs,
                )
                extracted_signals.append(signal)
            except ValueError:
                logging.warning(
                    f"Signal {signal_name} could not be extracted"
                    f"from file {self.mrn}. Patient won't be extracted.",
                )
                return None

        metadata = self.get_patient_metadata(visit, max_len=max_length)
        patient = Patient(
            extracted_signals,
            metadata,
            mrn=self.mrn,
            visit=visit,
            file=self.filename,
        )
        return patient


class Signal:
    """ Represents a signal with timestamps, values and methods associated """

    def __init__(
        self,
        name=None,
        values=None,
        time=None,
        sample_freq=None,
        stype=None,
        units=None,
    ):
        self.name = name
        self.values = values
        self.time = time
        self.sample_freq = sample_freq
        self.units = units

        self.stype = stype
        self.summary = {"original_size": self.values.size}
        self.scale_factor = (1, 1)
        self.verify()

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Signal({self.name})"

    def verify(self):
        if self.time is None or self.values is None or self.time.size == 0:
            raise ValueError(f"Signal {self.name} does not have either time or values")
        if self.time.size != self.values.size:
            raise ValueError(
                f"Signal {self.name} has different len for time ({self.time.size}) "
                f"and values ({self.values.size})",
            )
        return True

    def remove_outliers(self, method, **kwargs):
        signal_limits = kwargs.pop("signal_limits", (-np.inf, np.inf))

        if method != "min_max_values":
            low_th, up_th = OutlierRemover.remove_outliers(
                struct=self, method=method, **kwargs
            )
        else:
            low_th, up_th = (-np.inf, np.inf)

        min_value = kwargs.pop("min_value", -np.inf)
        max_value = kwargs.pop("max_value", np.inf)

        min_value = max(min_value, signal_limits[0], low_th)
        max_value = min(max_value, signal_limits[1], up_th)

        outs, outs_pc = OutlierRemover.remove_outliers(
            struct=self,
            method="min_max_values",
            min_value=min_value,
            max_value=max_value,
            **kwargs,
        )
        self.summary["outliers_removed"] = outs
        self.summary["outliers_removed (%)"] = outs_pc

    def downsample(self, method="linear interpolation", **kwargs):
        ds_percent = Downsampler.downsample(signal=self, method=method, **kwargs)
        self.summary["downsample%"] = ds_percent

    def crop(self, min_time, max_time):
        valid_idx = np.where((self.time >= min_time) & (self.time <= max_time))
        self.summary["points cropped"] = self.values.size - valid_idx.size
        self.time = self.time[valid_idx]
        self.values = self.values[valid_idx]

    def pad(self, filling, min_time, max_time, **kwargs):
        padding = Padder.pad(
            _signal=self, method=filling, min_time=min_time, max_time=max_time, **kwargs
        )
        self.summary["values padded"] = padding
        self.summary["values padded (total)"] = sum(padding)

    def normalize(self, min_val, max_val):
        self.values = (self.values - min_val) / (max_val - min_val)
        self.time = self.time - self.time[0]

    def rescale(self):
        minv, maxv = self.scale_factor
        self.values = self.values * (maxv - minv) + minv
        self.scale_factor = (1, 1)

    def plot(self):
        time = (self.time - self.time[0]) / 3600
        plt.plot(time, self.values)
        plt.show()


class Patient:
    """ A patient containing a set of signals a methods associated """

    def __init__(
        self,
        signals: Union[Dict[str, Signal], List[Signal]],
        meta,
        mrn=None,
        visit=None,
        file=None,
    ):
        if isinstance(signals, dict):
            self.signals = OrderedDict(signals)
        elif isinstance(signals, list):
            self.signals = OrderedDict({signal.name: signal for signal in signals})
        else:
            raise TypeError("signals should be either a list or a dict")
        self.meta = meta
        self.mrn = mrn
        self.visit = visit
        self.name = self.set_name()
        self.file = file
        self.out_padded = False
        self.verify()

    def __contains__(self, item):
        if isinstance(item, Signal):
            return item in self.signals.values()
        return item in self.signals.keys()

    def __iter__(self):
        return iter(sorted(self.signals.items(), key=lambda x: x[0]))

    def __getitem__(self, item):
        l = []
        for signal in sorted(self.signals.values(), key=lambda x: x.name):
            l.append(self.signals[signal.name].values[item])
        return np.array(l)

    def __setitem__(self, key, value):
        raise ValueError("Bundles shouldn't be manually modified")

    def __len__(self):
        length = np.inf
        for signal in self.signals.values():
            if signal.values.size < length:
                length = signal.values.size
        return length

    def __repr__(self):
        return f"Patient({self.name})"

    def set_name(self):
        ids = [ident for ident in (self.mrn, self.visit) if ident is not None]
        if len(ids) == 2:
            return f"{self.mrn} - {self.visit}"
        if len(ids) == 1:
            return ids[0]
        rand_id = "".join(random.choice(string.digits) for _ in range(8))
        return f"Unknown {rand_id}"

    def time_range(self):
        any_signal = next(iter(self.signals.values()))
        max_time = any_signal.time.max()
        min_time = any_signal.time.min()
        for signal in self.signals.values():
            signal_max = signal.time.max()
            signal_min = signal.time.min()
            if signal_max > max_time:
                max_time = signal_max
            if signal_min < min_time:
                min_time = signal_min
        return min_time, max_time

    def get_signal(self, signal) -> Signal:
        return self.signals[signal]

    def downsample(self, method, **kwargs):
        for signal in self.signals.values():
            signal.downsample(method, **kwargs)

    def pad(
        self, min_time=None, max_time=None, filling="zero", out_pad=False, **kwargs
    ):
        if not min_time:
            min_time = self.meta["Start time"].timestamp()
        if not max_time:
            max_time = (
                self.meta["End time"].timestamp()
                if not out_pad
                else min_time + self.meta["Period studied (h)"] * 3600
            )
        for signal in self.signals.values():
            if signal.stype == "meds":
                signal.pad(
                    filling="zero", min_time=min_time, max_time=max_time, **kwargs
                )
            else:
                signal.pad(
                    filling=filling, min_time=min_time, max_time=max_time, **kwargs
                )

        self.out_padded = out_pad

    def verify(self):
        for signal in self.signals.values():
            signal.verify()
            if signal.time[0] < self.meta["Start time"].timestamp():
                raise ValueError(f"Signal {signal.name} starts before blake08")
            if self.out_padded:
                if (
                    signal.time[-1]
                    > self.meta["Start time"].timestamp()
                    + self.meta["Period studied (h)"] * 3600
                ):
                    raise ValueError(f"Signal {signal.name} ends after what's allowed")
            else:
                if (
                    signal.time[-1] > self.meta["End time"].timestamp()
                    and not self.out_padded
                ):
                    raise ValueError(f"Signal {signal.name} ends after what's allowed")
            if self.meta["BLK08 start"] != self.meta["Start time"]:
                raise ValueError(
                    f"Patient {self}'s analyzed period does not start at"
                    f"the beginning of BLK08.",
                )
            real_span = (
                self.meta["End time"].timestamp() - self.meta["Start time"].timestamp()
            )
            teoric_max_span = self.meta["Period studied (h)"] * 3600
            if real_span > teoric_max_span:
                raise ValueError(
                    f"Patient {self} is being analyzed for a longer time range: "
                    f"{real_span / 3600}h than it should (max: {teoric_max_span}h)",
                )
            if real_span < teoric_max_span:
                if self.meta["BLK08 end"] != self.meta["End time"]:
                    blk08_stay = self.meta["BLK08 end"] - self.meta["BLK08 start"]
                    raise ValueError(
                        f"Patient {self} has a shorter analyzed timespan ({real_span}) "
                        f"than the max: ({teoric_max_span / 3600}h) but it does not "
                        f"come from a shorter BLK08 stay, which is {blk08_stay}",
                    )
            t_min, t_max = self.time_range()
            signals_range = t_max - t_min
            if not self.out_padded:
                if signals_range > real_span:
                    raise ValueError(
                        f"Patient {self}'s signals take larger time range than they "
                        f"should: {signals_range / 3600}h which are larger than the "
                        f"time span {real_span}h",
                    )
            else:
                if signals_range != teoric_max_span:
                    raise ValueError(
                        f"Patient {self} should be out-padded to "
                        f"{teoric_max_span / 3600}h duration, "
                        f"but it has {signals_range}h",
                    )

    def has_equal_len(self):
        """ Returns True if all signals on the patient have the same length. """
        size = next(iter(self.signals.values())).values.size
        for signal in self.signals.values():
            if signal.values.size != size:
                return False
        return True

    def remove_outliers(self, method, **kwargs):
        """ Removes outliers from the signals. """
        signal_limits = kwargs.pop("signal_limits", None)
        for signal in self.signals.values():
            if signal.stype == "meds":
                continue
            limits = signal_limits[signal.name] if signal_limits else (-np.inf, np.inf)
            signal.remove_outliers(method, signal_limits=limits, **kwargs)

    def plot(self, signals=None):
        """ Plots signals of the patient. """
        if not signals:
            signals = self.signals.keys()

        for signal_name in signals:
            signal = self.signals[signal_name]
            s_time = (signal.time - signal.time[0]) / 3600
            plt.plot(s_time, signal.values, label=signal_name)

        plt.title(self.name)
        if self.meta:
            start = self.meta["Start time"].timestamp()

            def substract(ts):
                return (ts.timestamp() - start) / 3600

            plt.axvline(x=substract(self.meta["BLK08 start"]))
            plt.axvline(x=substract(self.meta["BLK08 end"]))
            plt.axvline(x=substract(self.meta["Start time"]), color="red")
            plt.axvline(x=substract(self.meta["End time"]), color="red")

        plt.legend()
        plt.show()

    def force_overlap(self):
        min_time = 0
        max_time = 10000000000000000
        for signal in self.signals.values():
            if signal.time[0] > min_time:
                min_time = signal.time[0]
            if signal.time[-1] < max_time:
                max_time = signal.time[-1]
        for signal in self.signals.values():
            signal.crop(min_time, max_time)

    def normalize_signal(self, signal, min_val, max_val):
        self.get_signal(signal).normalize(min_val, max_val)

    def get_curation_summary(self):
        summary = {}
        for signal_name, signal in self.signals.items():
            summary[signal_name] = signal.summary
        return summary

    def rescale(self):
        for signal in self.signals.values():
            signal.rescale()


class Bundle:
    """ Represents a set of patients with methods associated. """

    def __init__(self, patients: Dict[str, Patient], name=None):
        self.name = name
        self.patients = OrderedDict(patients)

        self.out_padded = False
        self.processes: OrderedDict = OrderedDict({"raw": {}})

        self.verify()

    def __getitem__(self, item):
        return self.patients[item]

    def __repr__(self):
        return f"Bundle({self.name})"

    def verify(self):
        time_span = self.time_range()
        signals = self.signal_list()
        for patient in self.patients.values():
            patient.verify()
            patient_signals = list(patient.signals)
            if not sorted(signals) == sorted(patient_signals):
                raise ValueError(f"Patient  {patient} has a different set of signals")
            if not patient.meta["Period studied (h)"] == time_span:
                raise ValueError(
                    f"Patient {patient} has a different teoric time range "
                    f"({patient.meta['teoric_max_span']}h) "
                    f"than the rest ({time_span}h)",
                )
            if not patient.out_padded == self.out_padded:
                raise ValueError(
                    f"Patient {patient}'s out-padding status is "
                    f"'{patient.out_padded}', while it should be "
                    f"'{self.out_padded}'",
                )

    def signal_list(self):
        return sorted(list(self.any_patient().signals))

    def time_range(self):
        return self.any_patient().meta["Period studied (h)"]

    def any_patient(self):
        return next(iter(self.patients.values()))

    def _signal_min_max(self, signal):
        total_max = -10000000
        total_min = 10000000
        for patient in self.patients.values():
            sig = patient.get_signal(signal)
            sig_max = sig.values.max()
            sig_min = sig.values.min()
            if sig_max > total_max:
                total_max = sig_max
            if sig_min < total_min:
                total_min = sig_min
        return total_min, total_max

    def patient_list(self, include_clean=False):
        if include_clean:
            patients = [(path, patient.name) for path, patient in self.patients.items()]
            return sorted(patients, key=lambda x: x[0])
        return sorted(list(self.patients))

    def remove_outliers(
        self, method=None, filter_method=None, list_methods=False, **kwargs
    ):
        """
        Removes outliers from the bundle. Use list_methods=True for the list of methods
        available to do so.
        """
        if list_methods:
            return OutlierRemover.print_methods()

        signal_limits = {}
        if filter_method:
            signal_limits = OutlierRemover.remove_outliers(
                self, filter_method, **kwargs
            )

        incorrect_patients = []
        for p_name, patient in log_progress(
            self.patients.items(),
            desc="Removing outliers...",
        ):
            try:
                patient.remove_outliers(method, signal_limits=signal_limits, **kwargs)
            except IncorrectSignalError as err:
                detailed_message = f"Patient {p_name} on {str(err)}"
                logging.warning(detailed_message)
                incorrect_patients.append(p_name)

        if incorrect_patients:
            logging.warning(f"Deleting {len(incorrect_patients)} patients...")
            for patient in incorrect_patients:
                del self.patients[patient]

        self.verify()
        self.processes["outliers_removal"] = [method, kwargs]
        return None

    def normalize(self, method=None, list_methods=False, **kwargs):
        """
        Normalizes all the signals on the bundle. Use list_methods=True for the
        list of methods available to do so.
        """
        if list_methods:
            return Normalizer.print_methods()
        Normalizer.normalize(method, bundle=self, **kwargs)
        self.verify()
        self.processes["normalize"] = [method, kwargs]
        return None

    def downsample(self, method=None, list_methods=False, **kwargs):
        """
        Downsamples all the signals on the bundle. Use list_methods=True for the
        list of methods available to do so.
        """
        if list_methods:
            return Downsampler.print_methods()
        for patient in log_progress(self.patients.values(), desc="Downsampling..."):
            patient.downsample(method, **kwargs)
        self.verify()
        self.processes["downsample"] = [method, kwargs]
        return None

    def pad(
        self,
        min_time=None,
        max_time=None,
        filling="zero",
        out_pad=False,
        list_methods=False,
        **kwargs,
    ):
        """
        Pads the signals from the bundle with the selected filling.

        * Use list_methods=True for the list of methods available to do so.
        * If out_pad=False, the padding is applied until the end of the signal with
          maximum length. If out_pad=True, the padding is applied until the end of
          the bundle period.
        * If min_time or max_time are set, the padding will be applied from or until
          those timestamps. Otherwise, the padding will be applied depending on the
          out_pad parameter.
        """
        if list_methods:
            return Padder.print_methods()

        process_name = "outpad" if out_pad else "inpad"

        for patient in log_progress(
            self.patients.values(),
            desc=f"{process_name}ing...",
        ):
            patient.pad(
                filling=filling,
                min_time=min_time,
                max_time=max_time,
                out_pad=out_pad,
                **kwargs,
            )
        self.out_padded = out_pad
        self.verify()
        self.processes[process_name] = [filling, kwargs]
        return None

    def plot_signal(self, signal_name, patients=None):
        """
        Plots the a signal on the bundle. Specific patients can be selected with the
        `patients` argument. If `patients` is `any`, a random patient will be chosen.
        If `patients` is None, all patients will be plotted.
        """
        if not patients:
            patients = self.patient_list()
        elif patients == "any":
            patients = [random.choice(self.patient_list())]

        for patient in patients:
            signal = self.patients[patient].get_signal(signal_name)
            s_time = (signal.time - signal.time[0]) / 3600
            plt.plot(s_time, signal.values)

        plt.title(signal_name)
        plt.xlabel("Time")
        plt.ylabel(signal.units)
        plt.show()

    def feature_matrix(self, method, store_path=None, verbose=False, **kwargs):
        """
        Creates a 2D matrix representation of the signal by concatenating all
        signals from a patient. Each patient will be a column with the signals
        concatenated one after the other. Therefore, from a 3D information
        n_patients x n_signals x n_timestamps it will return a 2D matrix of
        n_patients x (n_signals*n_timestamps).

        Use the `store_path` variable to store the matrix on a pickle.
        """
        dist_matrix = FeatureExtractor.get_feature(
            method, self, verbose=verbose, **kwargs
        )
        if store_path:
            with open(store_path, "wb") as out:
                pickle.dump(dist_matrix, out, pickle.HIGHEST_PROTOCOL)
        return dist_matrix

    def distance_matrix(
        self, method=None, store_path=None, list_methods=False, verbose=False, **kwargs
    ):
        """
        Calculates the distance matrix from the bundle. Use list_methods=True
        to return the list of methods.

        Use list_methods=True for the list of methods available to do so.

        Use the `store_path` variable to store the matrix on a pickle..
        """
        if list_methods:
            return DistanceMetric.print_methods()

        n_patients = len(self.patient_list())
        dist_matrix = np.zeros((n_patients, n_patients))
        for row, (_, patient1) in enumerate(sorted(self.patients.items())):
            for col, (_, patient2) in enumerate(sorted(self.patients.items())):
                if dist_matrix[row, col] != 0:
                    continue
                if row == col:
                    d = 0
                else:
                    d = DistanceMetric.get_distance(
                        method, patient1, patient2, verbose, **kwargs
                    )
                dist_matrix[row, col] = d
                dist_matrix[col, row] = d
            if verbose:
                print(f"\t row {row}/{len(self.patients)}")

        if store_path:
            with open(store_path, "wb") as out:
                pickle.dump(dist_matrix, out, pickle.HIGHEST_PROTOCOL)

        return dist_matrix

    def cluster(
        self,
        method=None,
        distances=None,
        distance_algo=None,
        cluster_algo=None,
        list_methods=False,
        optimize=False,
        **kwargs,
    ):
        """
        Clusters patients on the bundle.
        :param method: method for clustering. Use list_methods=True for a list of
                       available ones.
        :param distances: distance matrix to use for clustering.
        :param distance_algo: algorithm to treat the distances on the distance matrix.
                             Use list_methods=True for a list of the available ones.
        :param cluster_algo: algorithm to tweak the clustering process.
                             Use list_methods=True for a list of the available ones.
        :param list_methods: if True, the ouput will be a list of the available
                             clustering methods.
        :param optimize: if True, the clustering hyperparameters will try to be
                         optimized (if available on the chosen clustering method)
        :param kwargs: Additional params for the clustering method.
        :returns: a ClusterResult object that relates each patient with its cluster
        """
        if list_methods:
            return pprint.pprint(Cluster.ACCEPTED_DISTANCES)
        clusters = Cluster.cluster(
            method=method,
            distances=distances,
            distance_algo=distance_algo,
            cluster_algo=cluster_algo,
            optimize=optimize,
            **kwargs,
        )

        cluster_results = {}
        for idx, patient_name in enumerate(sorted(self.patients)):
            cluster_results[patient_name] = clusters.labels_[idx]
        cluster_results = ClusterResult(cluster_results)

        for p_name in self.patients.keys():
            if p_name not in cluster_results.patients:
                raise ValueError(f"Patient {p_name} does not have a cluster!")

        return cluster_results

    def rescale(self):
        """ Undoes normalization step  """
        for patient in self.patients.values():
            patient.rescale()

    def reduce_dimensions(self, method, features=None, list_methods=False, **kwargs):
        """
        Reduces dimensions of a feature matrix with the given method.
        Use list_methods=True for the list of methods available to do so.

        :param method: Method for dimensionality reduction.
        :param features: a matrix with the features. If None, you can specify
                         a method and params to calculate it as kwargs
        :param list_methods: return a list of available methods for dimensionality
                             retuction instead.
        :param kwargs: kwargs to use for calculation of the feature matrix if None
                       is input.
        :returns: tuple with a dataframe containing the reduced space and an float
                  containing the explained variance.
        """
        if list_methods:
            return DimensionsReducer.print_methods()

        if features is None:
            features = self.feature_matrix(**kwargs)

        space, exp_var = DimensionsReducer.reduce(method, features)

        df = pd.DataFrame(space, columns=[f"ax{i + 1}" for i in range(space.shape[1])])
        df["cluster"] = [p.cluster for _, p in sorted(self.patients.items())]
        df["patient"] = self.patient_list()

        return df, exp_var

    def cluster_stats(self, cluster_results):
        """ Returns a ClusterReport object with summary statistics of the clusters """
        cluster_report = ClusterReport()
        for patient in self.patients.values():
            cluster = cluster_results[patient.name]
            cluster_report.add_patient(patient, cluster=cluster)
        return cluster_report

    def get_cluster_trajectories(self, cluster_results):
        """ Returns the average trajectory for each signal on each cluster """
        trajectories = {}
        signals = self.signal_list()
        npoints = len(self.any_patient())

        for cluster, cluster_patients in cluster_results.patients_per_cluster:
            cluster_traj = {}
            for signal in signals:
                stacked_values = []
                for i in range(npoints):
                    i_values = [
                        self.patients[p_name].signals[signal].values[i]
                        for p_name in cluster_patients
                    ]
                    stacked_values.append(i_values)
                values = np.array(stacked_values).mean(axis=1)
                cluster_traj[signal] = Signal(
                    name=f"{signal}-mean",
                    values=values,
                    time=np.arange(0, values.size),
                    sample_freq=self.any_patient().signals[signal].sample_freq,
                    stype=self.any_patient().signals[signal].stype,
                )
            trajectories[cluster] = cluster_traj

        return trajectories

    def get_curation_stats(self):
        summary = {}
        for patient in self.patients.values():
            summary[patient.name] = patient.get_curation_summary()
        return summary

    def get_bundle_report(self):
        max_trange_patients = 0
        early_dischage_patients = 0

        max_time_range = self.time_range() * 3600
        early_dc_time = max_time_range * 0.2

        for patient in self.patients.values():
            p_tmin, p_tmax = patient.time_range()
            p_range = p_tmax - p_tmin
            if p_range == max_time_range:
                max_trange_patients += 1
            elif max_time_range - p_range > early_dc_time:
                early_dischage_patients += 1

        report = {
            "Total patients": len(self.patients),
            "Time range": f"{str(self.time_range())}h",
            "Patients with complete time range": max_trange_patients,
            "Patients with early discharge": len(self.patients) - max_trange_patients,
            f"Sooner than {int(early_dc_time / 3600)}h": early_dischage_patients,
        }
        return report

    def store(self, path, rename_subfiles=False):
        if rename_subfiles:
            folder, name_with_ext = os.path.split(path)
            folder = folder if folder else "."
            new_name, _ = os.path.splitext(name_with_ext)

            old_name = self.name
            other_bundles = [
                file for file in os.listdir(folder) if file.startswith(f"{old_name}-")
            ]
            for file_name in other_bundles:
                new_file_name = file_name.replace(old_name, new_name)
                os.rename(
                    os.path.join(folder, file_name),
                    os.path.join(folder, new_file_name),
                )
            self.name = new_name

        with open(path, "wb") as out:
            pickle.dump(self, out, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def from_pickle(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)


class ClusterResult:
    """ Class that associates every patient on a Bundle with a cluster"""

    def __init__(self, patient_cluster: Dict):
        self.patient_cluster = patient_cluster
        self.clusters = list(set(self.patient_cluster.values()))
        self.patients = list(self.patient_cluster.keys())

    def __getitem__(self, item):
        return self.patient_cluster[item]

    def __setitem__(self, key, value):
        self.patient_cluster[key] = value

    def patients_per_cluster(self):
        patients_per_cluster = {cluster: [] for cluster in self.clusters}
        for patient, cluster in self.patient_cluster:
            patients_per_cluster[cluster].append(patient)
        return patients_per_cluster

    def store(self, path):
        with open(path, "wb") as out:
            pickle.dump(self, out, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def from_pickle(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)


class ClusterReport:
    """
    Class that contains summary statistics about each cluster from a ClusterResult
    on a Bundle.
    """

    def __init__(self):
        self.cluster_info = {}
        self.clusters = list(self.cluster_info)

    def __getitem__(self, item):
        return self.cluster_info[item]

    def __setitem__(self, key, value):
        self.cluster_info[key] = value

    @staticmethod
    def _format_categorical(data):
        value_count = {}
        unique_values = set(data)
        for uval in unique_values:
            value_count[f"{uval} (%)"] = (data.count(uval) / len(data)) * 100
        return value_count

    @staticmethod
    def _format_list(data, percent=False, include_std=False):
        data = np.array(data)
        data = data[~np.isnan(data)]
        mean = np.mean(data)
        if percent:
            mean = mean * 100
        if include_std:
            std = np.std(data)
            return f"{mean:.2f} +/- {std:.2f}"
        return mean

    def add_patient(self, patient, cluster):
        if cluster not in self.cluster_info:
            self.cluster_info[cluster] = {"patients": 0, "patient_list": []}

        for key, value in patient.meta.items():
            if key not in self.cluster_info[cluster]:
                self.cluster_info[cluster][key] = []
            self.cluster_info[cluster][key].append(value)

        self.cluster_info[cluster]["patients"] += 1
        self.cluster_info[cluster]["patient_list"].append(patient.name)

    def cluster_patients(self):
        clust_patients = {
            cluster: self.cluster_info[cluster]["patient_list"]
            for cluster in self.clusters
        }
        return clust_patients

    def get_summary(self, include_std=False):
        cluster_report = {}
        for cluster, cluster_stats in sorted(
            self.cluster_info.items(),
            key=lambda x: x[0],
        ):
            cluster_name = f"Cluster {cluster}"
            cluster_report[cluster_name] = {}
            for key, value in cluster_stats.items():
                if key in ("tobacco", "alcohol", "race", "patient_list"):
                    continue
                if isinstance(value, list):
                    if isinstance(value[0], (int, float, np.integer)):
                        is_percent = "%" in key
                        formatted_value = self._format_list(
                            value,
                            percent=is_percent,
                            include_std=include_std,
                        )
                        cluster_report[cluster_name][key] = formatted_value
                    elif isinstance(value[0], str):
                        formatted_value = self._format_categorical(value)
                        for subkey in formatted_value:
                            cluster_report[cluster_name][subkey] = formatted_value[
                                subkey
                            ]
                else:
                    cluster_report[cluster_name][key] = value

        return pd.DataFrame(cluster_report)

    def plot_distribution(self, field):
        for cluster in sorted(self.cluster_info):
            data = self.cluster_info[cluster][field]
            if not isinstance(data[0], (int, float, np.integer)):
                raise ValueError(
                    "Can't make histograms of non numerical variables",
                )
            plt.hist(data, label=f"Cluster {cluster}", bins=100)
        plt.legend()
        plt.show()

    def store(self, path):
        with open(path, "wb") as out:
            pickle.dump(self, out, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def from_pickle(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)
