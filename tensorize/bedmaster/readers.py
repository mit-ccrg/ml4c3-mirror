# Imports: standard library
import os
import re
import logging
from abc import ABC
from typing import Any, Dict, List, Optional

# Imports: third party
import h5py
import numpy as np
import pandas as pd
import unidecode

# Imports: first party
from ml4c3.utils import get_unix_timestamps
from definitions.edw import EDW_FILES
from definitions.icu import ICU_SCALE_UNITS
from tensorize.bedmaster.data_objects import BedmasterSignal
from tensorize.bedmaster.bedmaster_stats import BedmasterStats
from tensorize.bedmaster.match_patient_bedmaster import PatientBedmasterMatcher

# pylint: disable=too-many-branches, dangerous-default-value


class Reader(ABC):
    """
    Parent class for our Readers class.

    As an abstract class, it can't be directly instanced. Its children
    should be used instead.
    """

    @staticmethod
    def _ensure_contiguous(data: np.ndarray) -> np.ndarray:
        if len(data) > 0:
            dtype = Any
            try:
                data = data.astype(float)
                if all(x.is_integer() for x in data):
                    dtype = int
                else:
                    dtype = float
            except ValueError:
                dtype = "S"
            try:
                data = np.ascontiguousarray(data, dtype=dtype)
            except (UnicodeEncodeError, SystemError):
                logging.info("Unknown character. Not ensuring contiguous array.")
                new_data = []
                for element in data:
                    new_data.append(unidecode.unidecode(str(element)))
                data = np.ascontiguousarray(new_data, dtype="S")
            except ValueError:
                logging.exception(
                    f"Unknown method to convert np.ndarray of "
                    f"{dtype} objects to numpy contiguous type.",
                )
                raise
        return data


class BedmasterReader(h5py.File, Reader):
    """
    Implementation of the Reader for Bedmaster data.

    Usage:
    >>> reader = BedmasterReader('file.mat')
    >>> hr = reader.get_vs('HR')
    """

    def __init__(
        self,
        file: str,
        scaling_and_units: Dict[str, Dict[str, Any]] = ICU_SCALE_UNITS,
        summary_stats: BedmasterStats = None,
    ):
        super().__init__(file, "r")
        self.max_segment = {
            "vs": {"segmentNo": 0, "maxTime": -1, "signalName": ""},
            "wv": {"segmentNo": 0, "maxTime": -1, "signalName": ""},
        }
        self.interbundle_corr: Dict[str, Optional[Dict]] = {
            "vs": None,
            "wv": None,
        }
        self.scaling_and_units: Dict[str, Dict[str, Any]] = scaling_and_units
        self.summary_stats = summary_stats
        if self.summary_stats:
            self.summary_stats.add_file_stats("total_files")

    def _update_max_segment(self, sig_name, sig_type, max_time):
        """
        Update the signal that holds the segment with the last timespan.

        Needed for inter-bundle correction.

        :param sig_name: <str> name of the new candidate signal
        :param sig_type: <str> wv or vs
        :param max_time: <int> latest timespan for that signal
        """
        packet = self["vs_packet"] if sig_type == "vs" else self["wv_time_original"]
        max_seg = self.max_segment[sig_type]
        max_seg["maxTime"] = max_time
        max_seg["segmentNo"] = packet[sig_name]["SegmentNo"][-1][0]
        max_seg["signalName"] = sig_name

    def get_interbundle_correction(self, previous_max):
        """
        Calculate interbundle correction parameters from previous bundle maxs.

        Based on the signal with maximum time from the previous bundle,
        it calculates the 'maxTime': the last timespan that is overlapped
        with the previous bundle, and 'timeCorr': the time shifting to be
        applied on this bundle.

        Parameters are stored on attribute 'interbundle_corr'.

        :param previous_max: <Dict> dict with the max timepans info from
            the previous bundle. Same format than 'max_sement' attribute.
        """

        def _ib_corr(previous_max, segments, time):
            ib_corr = None
            overlap_idx = np.where(segments[()] == previous_max["segmentNo"])[0]
            if overlap_idx.size > 0:  # Bundles overlap
                last_overlap_idx = overlap_idx[-1]

                if last_overlap_idx >= len(time):
                    last_overlap_idx = len(time) - 1

                last_overlap_time = time[last_overlap_idx][0]
                time_corr = previous_max["maxTime"] - last_overlap_time

                ib_corr = {"maxTime": last_overlap_time, "timeCorr": time_corr}
            return ib_corr

        vs_corr = None
        last_max_vs = previous_max["vs"]["signalName"]
        if self.contains_group("vs"):
            if last_max_vs in self["vs"].keys():
                vs_corr = _ib_corr(
                    previous_max=previous_max["vs"],
                    segments=self["vs_packet"][last_max_vs]["SegmentNo"],
                    time=self["vs_time_corrected"][last_max_vs]["res_vs"],
                )

        wv_corr = None
        last_max_wv = previous_max["wv"]["signalName"]
        if self.contains_group("wv"):
            if last_max_wv in self["wv"].keys():
                wv_corr = _ib_corr(
                    previous_max=previous_max["wv"],
                    segments=self["wv_time_original"][last_max_wv]["SegmentNo"],
                    time=self["wv_time_corrected"][last_max_wv]["res_wv"],
                )
        self.max_segment = previous_max
        self.interbundle_corr["vs"] = vs_corr
        self.interbundle_corr["wv"] = wv_corr

    def apply_ibcorr(self, signal: BedmasterSignal):
        """
        Apply inter-bundle correction on a given signal.

        The correction will be applied based on the 'interbundle_corr'
        attribute, which needs is updated using the method:
        'get_interbundle_correction'

        The correction will cut the overlapping values between this bundle
        and the previous one. In addition, it will shift the timespans so that
        the first timespan on this bundle is the continuation of the last
        timespan of the previouz value.

        Note that this shifting will occur until a dataevent 1 or 5 is found.

        :param signal: <BedmasterSignal> a Bedmaster signal.
        """
        source = "vs" if signal._source_type == "vitals" else "wv"
        if not self.interbundle_corr[source]:
            return

        overlap_idx = np.where(
            signal.time <= self.interbundle_corr[source]["maxTime"],  # type: ignore
        )[0]
        if overlap_idx.size > 0:
            first_non_ol_idx = overlap_idx[-1] + 1
            signal.time = signal.time[first_non_ol_idx:]
            signal.time_corr_arr = signal.time_corr_arr[first_non_ol_idx:]
            value_cut_idx = (
                first_non_ol_idx
                if source == "vs"
                else np.sum(signal.samples_per_ts[:first_non_ol_idx])
            )
            signal.value = signal.value[value_cut_idx:]
            signal.samples_per_ts = signal.samples_per_ts[first_non_ol_idx:]
            if signal.source == "waveform":
                signal.sample_freq = self.get_sample_freq_from_channel(
                    channel=signal.channel,
                    first_idx=first_non_ol_idx,
                )

        corr_to_apply = self.interbundle_corr[source]["timeCorr"]  # type: ignore
        if corr_to_apply:
            de_idx = np.where(signal.time_corr_arr == 1)[0]
            if de_idx.size > 0:  # Contains data events
                first_event = de_idx[0]
                signal.time[:first_event] = signal.time[:first_event] + corr_to_apply
            else:
                signal.time = signal.time + corr_to_apply

        if self.summary_stats and overlap_idx.size > 0:
            if signal.value.size > 0:
                self.summary_stats.add_signal_stats(
                    signal.name,
                    "overlapped_points",
                    first_non_ol_idx,
                    source=signal.source,
                )

    def contains_group(self, group_name: str) -> bool:
        """
        Check if the .mat file contains the given group.
        """
        has_group = False
        if group_name in self.keys():
            if isinstance(self[group_name], h5py.Group):
                has_group = True
        return has_group

    def list_vs(self) -> List[str]:
        """
        Get the JUST the names of vital signals contained on the .mat file.

        It doesn't return the value of the vital signs.
        :return: <list[str]> A list with the vital signals' names contained
                on the .mat file
        """
        if not self.contains_group("vs"):
            logging.warning(f"No BM vitalsign found on file {self.filename}.")
            if self.summary_stats:
                self.summary_stats.add_file_stats("missing_vs")
            return []
        return list(self["vs"].keys())

    def list_wv(self) -> Dict[str, str]:
        """
        Get the the names of waveform signals contained on the .mat file.

        The format is : {wv_name: channel}, where `channel` is the input
        channel where the the signal enters. If a channel contains
        no waveform or contains multiple waveforms, it will be ignored.

        :return: <Dict[str:str]> A dict with the wave form signals
                contained on the .mat file, along with their input channel.
        """
        wv_signals: Dict[str, str] = {}

        if not self.contains_group("wv"):
            logging.warning(f"No BM waveform found on file {self.filename}.")
            if self.summary_stats:
                self.summary_stats.add_file_stats("missing_wv")
            return wv_signals

        for ch_name in self["wv"].keys():
            signal_name = self.get_wv_from_channel(ch_name)
            if signal_name:
                wv_signals[signal_name] = ch_name

        return wv_signals

    def format_data(self, data) -> np.ndarray:
        """
        Format multidimensional data into 1D arrays.

        :param data: <np.array> Data to be formatted
        :return: <np.array> formatted data
        """
        # Pseudo 1D data to 1D data
        if data.shape[1] == 1:  # Case [[0],[1]]
            data = np.transpose(data)
        if data.shape[0] == 1:  # Case [[0, 1]]
            data = data[0]

        # 2D data unicode encoded to 1D decoded
        if data.ndim == 2:
            if data.shape[0] < data.shape[1]:
                data = np.transpose(data)
            data = self.decode_data(data)

        return data

    @staticmethod
    def decode_data(data: np.ndarray) -> np.ndarray:
        """
        Decodes data stored as unicode identifiers and returns a 1D array.

        Example:
        >>> data  # 3D array with unicode codes for '0','.','2'
        array([[48, 46, 50],
               [48, 46, 50],
               [48, 46, 50],
               [48, 46, 50]])

        >>> BedmasterReader.decode_data(data)
        array([0.2, 0.2, 0.2, 0.2])

        :param data: <np.ndarray> Data to decode
        :return: <np.ndarray> decoded data
        """

        def _decode(row):
            row = "".join([chr(code) for code in row]).strip()
            if row in ("X", "None"):
                return np.nan
            return row

        data = np.apply_along_axis(_decode, 1, data)
        try:
            data = data.astype(float)
            if all(x.is_integer() for x in data):
                dtype = int  # type: ignore
            else:
                dtype = float  # type: ignore
        except ValueError:
            dtype = "S"  # type: ignore

        data = data.astype(dtype)
        return data

    def get_vs(self, signal_name: str) -> Optional[BedmasterSignal]:
        """
        Get the corrected vs signal from the.mat file.

        2. Applies corrections on the signal
        3. Wraps the corrected signal and its metadata on a BedmasterDataObject

        :param signal_name: <string> name of the signal
        :return: <BedmasterSignal> wrapped corrected signal
        """
        if signal_name not in self["vs"].keys():
            raise ValueError(
                f"In bedmaster_file {self.filename}, the signal {signal_name} "
                "was not found.",
            )

        # Get values and time
        values = self["vs"][signal_name][()]

        if values.ndim == 2:
            values = self.format_data(values)

        if values.dtype.char == "S":
            logging.warning(
                f"{signal_name} on .mat file  {self.filename}, has unexpected "
                "string values.",
            )
            return None

        if values.ndim >= 2:
            raise ValueError(
                f"Signal {signal_name} on file: {self.filename}. The values"
                f"of the signal have higher dimension than expected (>1) after"
                f"being formatted. The signal is probably in a bad format so it "
                f"won't be written.",
            )

        time = np.transpose(self["vs_time_corrected"][signal_name]["res_vs"][:])[0]

        # Get the occurrence of event 1 and 5
        de_1 = self["vs_time_corrected"][signal_name]["data_event_1"]
        de_5 = self["vs_time_corrected"][signal_name]["data_event_5"]
        events = (de_1[:] | de_5[:]).astype(np.bool)

        # Get scaling factor and units
        if signal_name in self.scaling_and_units:
            scaling_factor = self.scaling_and_units[signal_name]["scaling_factor"]
            units = self.scaling_and_units[signal_name]["units"]
        else:
            scaling_factor = 1
            units = "UNKNOWN"

        # Samples per timespan
        samples_per_ts = np.array([1] * len(time))

        signal = BedmasterSignal(
            name=signal_name,
            source="vitals",
            channel=signal_name,
            value=self._ensure_contiguous(values),
            time=self._ensure_contiguous(time),
            units=units,
            sample_freq=np.array([(0.5, 0)], dtype="float,int"),
            scale_factor=scaling_factor,
            time_corr_arr=events,
            samples_per_ts=self._ensure_contiguous(samples_per_ts),
        )

        # Apply inter-bundle correction
        if self.interbundle_corr["vs"]:
            self.apply_ibcorr(signal)

        if signal.time.size == 0:
            logging.info(
                f"Signal {signal} on .mat file {self.filename} doesn't contain new "
                f"information (only contains overlapped values from previous bundles). "
                f"It won't be written.",
            )
            if self.summary_stats:
                self.summary_stats.add_signal_stats(
                    signal.name,
                    "total_overlap_bundles",
                    source=signal.source,
                )
            return None

        # Compress time_corr_arr
        signal.time_corr_arr = np.packbits(np.transpose(signal.time_corr_arr)[0])

        # Update the max segment time (for inter-bundle correction)
        max_time = time[-1]
        if max_time > self.max_segment["vs"]["maxTime"]:
            self._update_max_segment(signal_name, "vs", max_time)

        # Quality check on data
        if not signal.time.shape[0] == signal.value.shape[0]:
            logging.warning(
                f"Something went wrong with signal {signal.name} on file: "
                f"{self.filename}. Time vector doesn't have the same length than "
                f"values vector. The signal won't be written.",
            )
            if self.summary_stats:
                self.summary_stats.add_signal_stats(
                    signal.name,
                    "defective_signal",
                    source=signal.source,
                )
            return None

        if not signal.samples_per_ts.shape[0] == signal.time.shape[0]:
            logging.warning(
                f"Something went wrong with signal {signal.name} on file: "
                f"{self.filename}. Time vector doesn't have the same length than "
                f"values vector. The signal won't be written.",
            )
            if self.summary_stats:
                self.summary_stats.add_signal_stats(
                    signal.name,
                    "defective_signal",
                    source=signal.source,
                )
            return None

        if self.summary_stats:
            self.summary_stats.add_from_signal(signal)

        return signal

    def get_wv(
        self,
        channel_n: str,
        signal_name: str = None,
    ) -> Optional[BedmasterSignal]:
        """
        Get the corrected wv signal from the.mat file.

        1. Gets the signal and its metadata from the .mat file
        2. Applies corrections on the signal
        3. Wraps the corrected signal and its metadata on a BedmasterDataObject

        :param channel_n: <string> channel where the signal is
        :param signal_name: <string> name of the signal
        :return: <BedmasterSignal> wrapped corrected signal
        """
        if channel_n not in self["wv"].keys():
            raise ValueError(
                f"In bedmaster_file {self.filename}, the signal {channel_n} was "
                "not found.",
            )

        if not signal_name:
            signal_name = self.get_wv_from_channel(channel_n)
            if not signal_name:
                signal_name = "?"

        values = np.array(np.transpose(self["wv"][channel_n][:])[0])
        if values.ndim == 2:
            values = self.format_data(values)

        if values.ndim >= 2:
            raise ValueError(
                f"Something went wrong with signal {signal_name} "
                f"on file: {self.filename}. Dimension of values "
                f"formatted values is higher than expected (>1).",
            )

        time = np.transpose(self["wv_time_corrected"][channel_n]["res_wv"][:])[0]

        # Get scaling factor and units
        scaling_factor, units = self.get_scaling_and_units(channel_n, signal_name)

        # Get the occurrence of event 1 and 5
        de_1 = self["wv_time_corrected"][channel_n]["data_event_1"]
        de_5 = self["wv_time_corrected"][channel_n]["data_event_5"]
        time_reset_events = de_1[:] | de_5[:].astype(np.bool)

        # Get sample frequency
        sample_freq = self.get_sample_freq_from_channel(channel_n)

        # Get samples per timespan
        samples_per_ts = self["wv_time_original"][channel_n]["Samples"][()]
        if samples_per_ts.ndim == 2:
            samples_per_ts = self.format_data(samples_per_ts)

        signal = BedmasterSignal(
            name=signal_name,
            source="waveform",
            channel=channel_n,
            value=values[:],
            time=time[:],
            units=units,
            sample_freq=sample_freq,
            scale_factor=scaling_factor,
            time_corr_arr=time_reset_events,
            samples_per_ts=samples_per_ts,
        )

        # Apply inter-bundle correction
        if self.interbundle_corr["wv"]:
            self.apply_ibcorr(signal)

        if signal.time.size == 0:
            logging.info(
                f"In bedmaster_file {self.filename}, {signal} is completely "
                "overlapped, it won't be written.",
            )
            if self.summary_stats:
                self.summary_stats.add_signal_stats(
                    signal.name,
                    "total_overlap_bundles",
                    source=signal.source,
                )
            return None

        # Add the rest of events and compress the array
        tc_len = len(signal.time_corr_arr)
        de_2 = self["wv_time_corrected"][channel_n]["data_event_2"]
        de_3 = self["wv_time_corrected"][channel_n]["data_event_3"]
        de_4 = self["wv_time_corrected"][channel_n]["data_event_4"]

        events = signal.time_corr_arr | de_2[-tc_len:] | de_3[-tc_len:] | de_4[-tc_len:]
        events = np.packbits(np.transpose(events)[0])
        signal.time_corr_arr = events

        # Update the max segment time (for inter-bundle correction)
        max_time = time[-1]
        if max_time > self.max_segment["wv"]["maxTime"]:
            self._update_max_segment(channel_n, "wv", max_time)

        # Quality check on data
        if not signal.time.shape[0] == signal.samples_per_ts.shape[0]:
            logging.warning(
                f"Something went wrong with signal: "
                f"{signal.name} on file: {self.filename}. "
                f"Time vector doesn't have the same length than "
                f"'samples_per_ts' vector. The signal won't be written.",
            )
            if self.summary_stats:
                self.summary_stats.add_signal_stats(
                    signal.name,
                    "defective_signal",
                    source=signal.source,
                )
            return None

        if not signal.value.shape[0] == np.sum(signal.samples_per_ts):
            logging.warning(
                f"Something went wrong with signal: "
                f"{signal.name} on file: {self.filename} "
                f"'samples_per_ts' vector's sum isn't equal to "
                f"values vector's length. This seems an error on the primitive "
                f".stp file. The signal won't be written.",
            )
            if self.summary_stats:
                self.summary_stats.add_signal_stats(
                    signal.name,
                    "defective_signal",
                    source=signal.source,
                )
            return None

        if self.summary_stats:
            self.summary_stats.add_from_signal(signal)

        return signal

    def get_wv_from_channel(self, channel: str) -> Optional[str]:
        path = f"wv_time_original/{channel}/Label"
        length = self[path].shape[-1]
        if length < 10:
            signals = self[path][:]
        else:
            signals = self[path][..., range(0, length, length // 10)]
        signals = np.unique(signals.T, axis=0)
        signals = signals[(signals != 32) & (signals != 0)]

        if signals.ndim > 1:
            logging.warning(
                f"Channel {channel} on file {self.filename} "
                f"is a mix of different signals: {signals}. "
                f"This situation is not supported. "
                f"The channel will be ignored.",
            )
            if self.summary_stats:
                self.summary_stats.add_file_stats("multiple_label_signal")
            return None

        if signals.size == 0:
            logging.warning(
                f"The signal on channel {channel} on file {self.filename} "
                f"has no name. It is probably an empty signal or a badly "
                f"recorded one. It won't be written to the tensorized file.",
            )
            if self.summary_stats:
                self.summary_stats.add_file_stats("no_label_signal")
            return None

        name = "".join([chr(letter) for letter in signals])
        return name

    def get_sample_freq_from_channel(self, channel: str, first_idx=0):
        sf_arr = self["wv_time_original"][channel]["SampleRate"][first_idx:].T[0]
        if sf_arr.shape[0] <= 0:
            logging.info(
                f"The signal on channel {channel} on file {self.filename} has an "
                f"incorrect sample frequency format. Either it doesn't have sample "
                f"frequency or it has an incongruent one. Sample frequency will be set "
                f"to Nan for this signal.",
            )
            return np.array([(np.nan, 0)], dtype="float,int")
        changes = np.concatenate([[-1], np.where(sf_arr[:-1] != sf_arr[1:])[0]])
        return np.fromiter(
            ((sf_arr[index + 1], index + 1) for index in changes),
            dtype="float,int",
        )

    def get_scaling_and_units(self, channel_n, signal_name):
        if signal_name in self.scaling_and_units:
            scaling_factor = self.scaling_and_units[signal_name]["scaling_factor"]
            units = self.scaling_and_units[signal_name]["units"]
        else:
            try:
                calibration = self["wv_time_original"][channel_n]["Cal"][()]
                calibration = self.decode_data([calibration.T[0]])[0].decode("utf-8")
                calibration = [
                    part for part in re.split(r"(\d*\.?\d+)", calibration) if part
                ]
                if len(calibration) == 2:
                    scaling_factor, units = calibration
                else:
                    raise ValueError
            except (KeyError, ValueError):
                logging.warning(
                    f"Scaling factor or units not found "
                    f"for signal {signal_name} on file {self.filename}. They will "
                    f"be set to units: UNKNOWN, scaling_factor: 0.",
                )
                scaling_factor = 0
                units = "UNKNOWN"

        return float(scaling_factor), units


class CrossReferencer:
    """
    Class that cross-references Bedmaster and EDW data.

    Used to ensure correspondence between the data.
    """

    def __init__(
        self,
        bedmaster_dir: str,
        xref_file: str,
        adt: str,
    ):
        self.bedmaster_dir = bedmaster_dir
        self.xref_file = xref_file
        self.adt = adt
        self.crossref: Dict[str, Dict[str, List[str]]] = {}

    def get_xref_files(
        self,
        mrns: List[str] = None,
        starting_time: int = None,
        ending_time: int = None,
        overwrite_hd5: bool = True,
        n_patients: int = None,
        tensors: str = None,
        allow_one_source: bool = False,
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Get the cross-referenced Bedmaster files and EDW files.

        The output dictionary will have the format:

        {"MRN1": {"visitID":[bedmaster_files],
                  "visitID2": ...,
                  ...
                  }
         "MRN2: ...}

        :param mrns: <List[str]> list with the MRNs.
                    If None, it take all the existing ones
        :param starting_time: <int> starting time in Unix format.
                             If None, timestamps will be taken
                             from the first one.
        :param ending_time: <int> ending time in Unix format.
                            If None, timestamps will be taken
                            until the last one.
        :param overwrite_hd5: <bool> indicates if the mrns of the existing
                              hd5 files are excluded from the output dict.
        :param n_patients: <int> max number of patients to tensorize.
        :param tensors: <str> directory to check for existing hd5 files.
        :param allow_one_source: <bool> bool indicating whether a patient with
                                just one type of data will be tensorized or not.
        :return: <dict> a dictionary with the MRNs, visit ID and Bedmaster files.
        """
        self.crossref = {}
        if not os.path.exists(self.xref_file):
            bedmaster_matcher = PatientBedmasterMatcher(
                bedmaster=self.bedmaster_dir,
                adt=self.adt,
            )
            bedmaster_matcher.match_files(
                self.xref_file,
            )

        adt_df = pd.read_csv(self.adt)
        adt_columns = EDW_FILES["adt_file"]["columns"]
        adt_df = adt_df[adt_columns].drop_duplicates()

        xref = pd.read_csv(self.xref_file)
        xref = xref.drop_duplicates(subset=["MRN", "PatientEncounterID", "path"])
        xref["MRN"] = xref["MRN"].astype(str)

        if mrns:
            xref = xref[xref["MRN"].isin(mrns)]
            adt_df = adt_df[adt_df["MRN"].isin(mrns)]
        if starting_time:
            xref = xref[xref["TransferInDTS"] > starting_time]
            adt_df[adt_columns[4]] = get_unix_timestamps(adt_df[adt_columns[4]].values)
            adt_df = adt_df[adt_df[adt_columns[4]] > starting_time]
        if ending_time:
            xref = xref[xref["TransferOutDTS"] < ending_time]
            adt_df[adt_columns[3]] = get_unix_timestamps(adt_df[adt_columns[3]].values)
            adt_df = adt_df[adt_df[adt_columns[3]] < ending_time]

        edw_mrns = list(adt_df[adt_columns[0]].drop_duplicates().astype(str))

        if not overwrite_hd5 and tensors and os.path.isdir(tensors):
            existing_mrns = [
                hd5file[:-4]
                for hd5file in os.listdir(tensors)
                if hd5file.endswith(".hd5")
            ]
            xref = xref[~xref["MRN"].isin(existing_mrns)]
            edw_mrns = [ele for ele in edw_mrns if ele not in existing_mrns]
        elif not overwrite_hd5 and not tensors:
            logging.warning(
                "overwrite_hd5 is set to False, but output_dir option is "
                "not set, ignoring overwrite_hd5 option. HD5 files are "
                "going to be overwritten.",
            )

        self.add_bedmaster_elements(
            xref=xref,
            edw_mrns=edw_mrns,
            allow_one_source=allow_one_source,
        )

        # Get only the first n patients
        if (n_patients or 0) > len(self.crossref):
            logging.warning(
                f"Number of patients set to tensorize "
                f"exceeds the amount of patients stored. "
                f"Number of patients to tensorize will be changed to "
                f"{len(self.crossref)}.",
            )
        else:
            self.crossref = dict(list(self.crossref.items())[:n_patients])
        return self.crossref

    def add_bedmaster_elements(self, xref, edw_mrns, allow_one_source):
        # Add elements from xref.csv
        for _, row in xref.iterrows():
            mrn = str(row["MRN"])
            if not allow_one_source and mrn not in edw_mrns:
                continue
            try:
                csn = str(int(row["PatientEncounterID"]))
            except ValueError:
                csn = str(row["PatientEncounterID"])
            fname = os.path.split(row["path"])[1]
            bedmaster_path = os.path.join(self.bedmaster_dir, fname)
            if mrn not in self.crossref:
                self.crossref[mrn] = {csn: [bedmaster_path]}
            elif csn not in self.crossref[mrn]:
                self.crossref[mrn][csn] = [bedmaster_path]
            else:
                self.crossref[mrn][csn].append(bedmaster_path)

        for _mrn, visits in self.crossref.items():
            for _csn, bedmaster_files in visits.items():
                bedmaster_files.sort(
                    key=lambda x: (int(re.split("[_-]", x)[-3]), int(x.split("_")[-2])),
                )
