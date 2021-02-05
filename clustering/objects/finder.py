# Imports: standard library
import os
import pickle
import logging
from typing import Union
from collections import Counter, namedtuple

# Imports: third party
import numpy as np
import pandas as pd
from tqdm import tqdm as log_progress
from matplotlib import pyplot as plt

# Imports: first party
from definitions.icu_tmaps import DEFINED_TMAPS
from clustering.objects.structures import Bundle, HD5File

FILE_ID = namedtuple("FILE_ID", ["file", "visit"])


class Explorer:
    """
    Tool to find signals on a set od HD5 Files and get statistics about the
    signals they contain.
    """

    def __init__(self, path):
        self.files = [file for file in os.listdir(path) if file.endswith(".hd5")]
        self.path = path
        self.report = ExploreReport()

    def get_quality(self):
        """ Crates a report with statistics from the HD5 files"""
        for file in log_progress(self.files):
            file_path = os.path.join(self.path, file)
            try:
                with HD5File(file_path) as hd5:
                    for visit in hd5.visits:
                        if hd5.has_source("edw", visit):
                            blk08 = bool(
                                hd5.get_department_duration(
                                    department="BLK08",
                                    visit=visit,
                                    only_first=True,
                                ),
                            )
                        else:
                            blk08 = "-"
                        for sig_type in hd5.signal_types(visit):
                            self.report.add_row(
                                True,
                                hd5.sources,
                                sig_type,
                                file,
                                visit,
                                blk08,
                            )
                    if not hd5.visits:
                        logging.warning(f"File {file} has no visits")
                        self.report.add_row(
                            is_readable=True,
                            sources=hd5.sources,
                            file=file,
                        )
            except OSError:
                logging.warning(f"File {file} could not be read")
                self.report.add_row(is_readable=False)

        return self.report

    def guess_signal_name(self, keyword, signal_type=None):
        """
        Given a keyword, search between all the files signals that contain that keyword.
        If `signal_type` is set, the keyword will be searched only under that signal
        type.
        """
        proposed_sigs = Counter()
        print("Could it be...?")

        for file in log_progress(self.files):
            file_path = os.path.join(self.path, file)
            try:
                with HD5File(file_path) as hd5:
                    visits = hd5.visits
                    if not visits:
                        logging.warning(f"File {file} has no visits")
                        self.report.add_row(
                            is_readable=True,
                            sources=hd5.sources,
                            file=file,
                        )

                    possible_types = [signal_type] if signal_type else hd5.signal_types
                    for visit in visits:
                        for sig_type in possible_types:
                            if hd5.has_sig_type(sig_type, visit=visit):
                                full_path = HD5File.get_full_type_path(
                                    signal_type,
                                    visit,
                                )
                                for sig in hd5[full_path]:
                                    if keyword in sig:
                                        proposed_sigs[sig] += 1
                                        logging.info(f"Found new signal!: {sig}")
            except (OSError, KeyError):
                logging.warning(f"File {file} could not be read")
                self.report.add_row(is_readable=False)

        return proposed_sigs

    def find(self, signals=None, department=None, stay_length=0):
        """
        Create a table with the overlap time of the input signals during the BLK08
        stay of the patient. If stay length is set, the considered time will be the
        period between entry to BLK08 and the next <stay_length> hours.

        Note:
        * EDW data is needed to get the BLK08 stay of the patient, so any hd5 file
          without EDW data will be ignored.
        * Patients that haven't gone through BLK08 or whose BLK08 data is missing
          (admittance or discharge from BLK08 is unknown) are ignored.
        * If a patient has gone through BLK08 multiple times, only the first BLK08 stay
          will be used.

        :param signals: list with the signals to be searched.
        :param department: department where the signal will be looked at. If None,
                           the whole signal will be taken.
        :param stay_length: length of the stay (in hours) to consider the signal.
                            If set to 0 it will take the whole BLK08 stay.
        :return:
        """
        timeseries_types = ["vitals", "waveform", "flowsheet", "labs"]

        tmap_signals = []
        for stype in timeseries_types:
            tmap_signals.extend(DEFINED_TMAPS[stype])

        if signals:
            if all(sig in timeseries_types for sig in signals):
                logging.info(
                    "Signal types instead of individual signals detected."
                    "Will take all the tmaps for those types.",
                )
                tmap_signals = []
                for stype in signals:
                    tmap_signals.extend(DEFINED_TMAPS[stype])

                signals = tmap_signals
            else:
                logging.info("Individual signal detected")
                not_valid_signals = set(signals) - set(tmap_signals)
                if not_valid_signals:
                    logging.warning(
                        f"Signals {not_valid_signals} don't have an "
                        f"associated tmap. They won't be used",
                    )
                signals = set(signals) & set(tmap_signals)
        else:
            signals = tmap_signals

        results = {}
        for file in log_progress(self.files, desc="Finding files..."):
            file_path = os.path.join(self.path, file)
            logging.info(f"\t file: {file_path}")
            try:
                with HD5File(file_path) as hd5:
                    for visit_id in hd5.visits:

                        file_id = FILE_ID(file, visit_id)

                        if not hd5.has_source("edw", visit=visit_id):
                            logging.info(
                                f"CSN {visit_id} of MRN {hd5.mrn} "
                                f"does not have edw data. Ignorning it",
                            )
                            continue

                        dpmt_stays = hd5.get_department_duration(
                            visit_id,
                            department=department,
                            max_len=stay_length,
                            only_first=True,
                            asunix=False,
                        )
                        if not dpmt_stays:
                            continue

                        file_info = {}

                        dpmt_stay = dpmt_stays[0]
                        dpmt_duration = dpmt_stay["end"] - dpmt_stay["start"]
                        dpmt_duration = dpmt_duration.total_seconds() / 3600

                        file_info["_period_studied"] = dpmt_duration
                        file_info["_period_studied_start"] = dpmt_stay[
                            "start"
                        ].timestamp()
                        file_info["_period_studied_end"] = dpmt_stay["end"].timestamp()

                        for signal in signals:
                            signal_info = hd5.find_signal(
                                signal,
                                department=department,
                                visit=visit_id,
                                max_length=stay_length,
                            )

                            for info in signal_info:
                                if info not in file_info:
                                    file_info[info] = {}
                                file_info[info][signal] = signal_info[info]

                        results[file_id] = file_info

            except OSError:
                logging.warning(f"File {file} is invalid!")

        request_report = RequestReport(results, department=department)
        return request_report

    def extract_data(self, report):
        patients = {}
        for file, visit in log_progress(report.files_and_visits):
            file_path = os.path.join(self.path, file)
            logging.info(f"\t file: {file_path}")
            try:
                with HD5File(file_path) as hd5:
                    patient = hd5.extract_patient(
                        visit,
                        signals=report.current_signals,
                        department=report.department,
                        max_length=report.time_studied,
                    )
                if patient:
                    patients[patient.name] = patient

            except OSError:
                logging.warning(f"File {file} is invalid!")

        bundle = Bundle(patients)
        return bundle

    def plot_file_signals(self, file_name, visit, signals, max_len=None):
        """
        Plots all the input signals of the input file for the BLK08 duration.
        Signals can be cropped using the `max_len` parameter.
        """
        file_path = os.path.join(self.path, file_name)
        with HD5File(file_path) as hd5:
            for signal in signals:
                s = hd5.get_signal(signal, visit, max_length=max_len)
                s.time = (s.time - s.time[0]) / 3600
                plt.plot(s.time, s.values)
        plt.legend(signals)
        plt.show()

    def plot_signal_trajectory(self, signal, report=None, max_len=None):
        """
        Plots a given signal through all the targeted files. If a report is given
        (RequestReport), only the filtered files of the report will be used.
        Otherwise, all the files on the directory containing that signal will be used.
        """
        if report:
            files_and_visits = report.files_and_visits.sort(key=lambda x: x[0])
            all_files, all_visits = zip(*files_and_visits)
        else:
            all_files = self.files

        for idx, file in log_progress(enumerate(all_files)):
            file_path = os.path.join(self.path, file)
            with HD5File(file_path) as hd5:
                visits = hd5.visits if not report else [all_visits[idx]]
                for visit in visits:
                    if hd5.has_signal(signal, visit):
                        s = hd5.get_signal(signal, visit, max_length=max_len)
                        s.time = (s.time - s.time[0]) / 3600
                        plt.plot(s.time, s.values)
        plt.show()

    def count_incomplete_signals(self, report, max_len=None):
        """
        Counts the number of signals that aren't complete for the whole BLK08 duration.
        """
        counter = Counter()
        for file, visit in log_progress(report.files_and_visits):
            file_path = os.path.join(self.path, file)
            with HD5File(file_path) as hd5:
                for signal in report.signals:
                    s = hd5.get_signal(signal, visit, max_length=max_len)
                    s.time = (s.time - s.time[0]) / 3600
                    if s.time[-1] < max_len - 0.5:
                        counter[s.name] += 1
        return counter


class ExploreReport:
    """ Report containing summary statistics about the hd5 files stored in a path """

    def __init__(self, data=None):
        if not data:
            data = []
        self.data = data

    def add_row(
        self,
        is_readable: bool = False,
        sources: Union[list, str] = "-",
        sig_type: str = "-",
        file: str = "-",
        visit: str = "-",
        blk08="-",
    ):
        if not sources:
            source = "No source"
        else:
            source = " and ".join(sources)
            if "and" not in source:
                source = f"Only {source}"

        readable = "Readable" if is_readable else "Not readable"
        self.data.append([readable, source, sig_type, file, visit, blk08])

    @property
    def df(self):
        dataframe = pd.DataFrame(self.data)
        dataframe.columns = [
            "Readable",
            "Source",
            "Signal type",
            "File",
            "Visit",
            "In BLK08",
        ]
        return dataframe

    @property
    def files(self):
        return self.df["File"].unique().size

    @property
    def visits(self):
        return self.df["Visit"].unique().size

    @property
    def columns(self):
        return self.df.columns

    def by_sources(self):
        df = self.df.drop(columns=["In BLK08"])
        df = df.groupby(["Readable", "Source", "Signal type"]).nunique()
        return df

    def by_blk08(self):
        df = self.df.drop(columns=["Signal type"])
        df = df.groupby(["Readable", "Source", "In BLK08"]).nunique()
        return df

    def by(self, stack_columns, value_columns):
        df = self.df[stack_columns + value_columns]
        df = df.groupby(stack_columns).nunique()
        return df

    def to_csv(self, path):
        self.df.to_csv(path, index=False)

    @staticmethod
    def from_csv(path):
        df = pd.read_csv(path)
        return ExploreReport(df.to_dict())


class RequestReport:
    """ Report containing statistics about the signals. Generated by Explorer.find """

    def __init__(self, data, department=None):
        self.data = data
        self.df = self.generate_table()
        self.department = department
        self.current_signals = self.all_signals

    def generate_table(self, signals=None, blame_signal=False):
        if signals is None:
            signals = self.all_signals
        unknown_signals = set(signals) - set(self.all_signals)
        if len(unknown_signals) != 0:
            raise ValueError(f"Signals {unknown_signals} are not found in data")

        data = []
        for file_id, file_data in self.data.items():
            row = {"File": file_id.file, "Visit": file_id.visit}

            existing_signals = [
                signal
                for signal in file_data["_existence"]
                if file_data["_existence"][signal]
            ]
            available_signals = set(existing_signals) & set(signals)

            if len(available_signals) == 0:
                duration = np.nan
            else:
                start_time = max(
                    file_data["_start_time"][signal] for signal in available_signals
                )
                end_time = min(
                    file_data["_end_time"][signal] for signal in available_signals
                )
                duration = (end_time - start_time) / 3600

            period_studied = file_data["_period_studied"]
            duration_percent = (duration / period_studied) * 100

            row["Overlap"] = duration > 0 and len(available_signals) == len(signals)
            row["Signals found"] = f"{len(available_signals)}/{len(signals)}"
            row["Overlap percent (%)"] = (
                duration_percent if duration_percent > 0 else np.nan
            )
            row["Overlap length (h)"] = duration if duration > 0 else np.nan
            row["Time studied (h)"] = period_studied

            for info_key, signal_values in file_data.items():
                if not info_key.startswith("_"):
                    curated_key = info_key.replace("_", " ").capitalize()
                    valid_values = [
                        (value, signal)
                        for signal, value in signal_values.items()
                        if signal in signals
                    ]
                    max_value, max_signal = max(valid_values, key=lambda x: x[0])
                    row[curated_key] = max_value
                    if blame_signal:
                        row[f"{curated_key} signal"] = max_signal

            data.append(row)

        self.df = pd.DataFrame(data)
        self.df = self.df.round(2)
        self.current_signals = signals

        return self.df

    @property
    def time_studied(self):
        return self.df["Time studied (h)"].max()

    @property
    def all_signals(self):
        signals = list(next(iter(self.data.values()))["_existence"].keys())
        return signals

    @property
    def files_and_visits(self):
        file_visits = [
            (file, str(visit)) for file, visit in zip(self.df["File"], self.df["Visit"])
        ]
        return file_visits

    def to_csv(self, path):
        self.df.to_csv(path, index=False)

    @staticmethod
    def from_pickle(path):
        with open(path, "rb") as file:
            return pickle.load(file)

    def to_pickle(self, path):
        with open(path, "wb") as out:
            pickle.dump(self, out, pickle.HIGHEST_PROTOCOL)

    def plot_overlap_length(self):
        plt.hist(self.df["Overlap length (h)"], 24, cumulative=-1)
        plt.xlim(left=0)
        plt.show()

    def plot_signal_duration(self, signal):
        """
        Plots the histogram of the duration of the signal over the period studied
        """
        if signal not in self.df:
            raise ValueError(
                f"Signal '{signal} not found on dataframe. Available signals: "
                f"{self.df.columns}",
            )
        freq = self.df[signal].replace("False", 0).astype(float)
        freq = freq[freq > 0] / 3600
        freq.plot(kind="hist", cumulative=True, bins=24, density=1, title=signal)
        plt.xlabel("Recording duration")
        plt.show()

    def plot_signals_frequency(self):
        counts = []
        for signal in self.current_signals:
            count = np.count_nonzero(~np.isnan(self.df[signal]))
            counts.append(count)
        plt.bar(self.all_signals, counts)
        plt.legend()
        plt.show()

    def plot_signal_cells(self):
        for signal in self.current_signals:
            values = ~np.isnan(self.df[signal])
            plt.hist(values, label=signal)
        plt.legend()
        plt.show()

    def remove_signal(self):
        pass
