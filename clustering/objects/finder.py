# Imports: standard library
import os
import logging
from typing import Union
from collections import Counter

# Imports: third party
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm as log_progress

# Imports: first party
from clustering.objects.structures import Bundle, HD5File


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
                                hd5.get_blk08_duration(visit=visit, only_first=True),
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

    def find(self, signals, cell="existence", stay_length=0):
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

        :param path: path where the hd5 files are
        :param signals: list with the signals to be searched
        :param cell: 3 options:
            - existence: cells are boolean that confirms existence of the signal
            - length: cells are the seconds of duration of the signal
            - percentage: cells are the percentage of existence of the signal with respect
                          to the BLK08 stay.
        :param stay_length: length of the stay (in hours) to consider the signal.
                            If set to 0 it will take the whole BLK08 stay
        :return:
        """

        def _find_on_file(hd5, signals, visit_id, stay_length):
            if not hd5.has_source("edw", visit=visit_id):
                logging.info(
                    f"CSN {visit_id} of MRN {hd5.mrn} does not "
                    f"have edw data. Ignorning it",
                )
                return

            blk08_stays = hd5.get_blk08_duration(
                visit_id,
                max_len=stay_length,
                only_first=True,
            )
            if not blk08_stays:
                logging.info(
                    f"CSN {visit_id} of MRN {hd5.mrn} didn't go "
                    f"through BLK08. Ignoring it",
                )
                return

            row = {}

            blk08_stay = blk08_stays[0]
            blk08_duration = blk08_stay["end"] - blk08_stay["start"]
            blk08_duration = blk08_duration.total_seconds() / 3600

            min_value = 0
            max_value = np.inf
            local_max_gap = 0
            complete_overlap = True
            non_mono_num = 0
            max_non_mono = 0
            for signal in signals:
                signal_info = hd5.find_signal(
                    signal,
                    visit=visit_id,
                    max_length=stay_length,
                )
                if signal_info:
                    if cell not in signal_info:
                        raise ValueError(
                            f"Cell value {cell} not valid! "
                            f"Valid ones are: {list(signal_info.keys())}",
                        )
                    if signal_info["start_time"] >= min_value:
                        min_value = signal_info["start_time"]
                    if signal_info["end_time"] <= max_value:
                        max_value = signal_info["end_time"]
                    if signal_info["max_unfilled"] > local_max_gap:
                        local_max_gap = signal_info["max_unfilled"]
                    if signal_info["non_monotonicities"] > non_mono_num:
                        non_mono_num = signal_info["non_monotonicities"]
                    if signal_info["max_non_monotonicity"] > max_non_mono:
                        max_non_mono = signal_info["max_unfilled"]
                else:
                    complete_overlap = False

                if cell in signal_info:
                    row[signal] = signal_info[cell]
                else:
                    row[signal] = np.nan

            signals_overlap = min_value <= max_value and complete_overlap
            overlap_len = max_value - min_value if complete_overlap else 0
            overlap_percent = overlap_len / (3600 * blk08_duration) * 100

            row["File"] = file
            row["Visit"] = visit_id
            row["Time studied (h)"] = blk08_duration
            row["Max unfilled time (s)"] = local_max_gap
            row["Max non-monotonicities (#)"] = non_mono_num
            row["Max non-monotonicity (s)"] = max_non_mono
            row["Overlap"] = signals_overlap
            row["Overlap length (h)"] = overlap_len / 3600
            row["Overlap percent (%)"] = overlap_percent

            return row

        results = []
        for file in log_progress(self.files, desc="Finding files..."):
            file_path = os.path.join(self.path, file)
            logging.info(f"\t file: {file_path}")
            try:
                with HD5File(file_path) as hd5:
                    for visit_id in hd5.visits:
                        row = _find_on_file(hd5, signals, visit_id, stay_length)
                        if not row:
                            continue
                        results.append(row)

            except OSError:
                print(f"File {file} is invalid!")

        request_report = RequestReport(pd.DataFrame(results))
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
                        report.signals,
                        report.time_studied,
                    )

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
    def __init__(self, df):
        self.df = df
        self.not_signal_columns = [
            "File",
            "Visit",
            "Time studied (h)",
            "Overlap",
            "Overlap length (h)",
            "Overlap percent (%)",
            "Max unfilled time (s)",
            "Max non-monotonicities (#)",
            "Max non-monotonicity (s)",
        ]

    @property
    def time_studied(self):
        return self.df["Time studied (h)"].max()

    @property
    def signals(self):
        signals = [col for col in self.df.columns if col not in self.not_signal_columns]
        return signals

    @property
    def files_and_visits(self):
        file_visits = [
            (file, str(visit)) for file, visit in zip(self.df["File"], self.df["Visit"])
        ]
        return file_visits

    @staticmethod
    def from_csv(path="data/all_files.csv"):
        df = pd.read_csv(path, index_col=False)
        return RequestReport(df)

    def to_csv(self, path):
        self.df.to_csv(path, index=False)

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
