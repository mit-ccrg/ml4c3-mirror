# Imports: standard library
import os
from typing import Dict

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from ml4c3.ingest.icu.data_objects import BedmasterSignal


class BedmasterStats:
    """
    Class that gets together the summary data from all the writers.

    It is used to organize this data and create a csv with all the
    information.
    """

    def __init__(self):
        self.signal_stats: Dict[str, Dict[str, int]] = {}
        self.file_stats: Dict[str, int] = self.init_files_dict()

    @staticmethod
    def init_signal_dict():
        return {
            "channel": [],
            "files": 0,
            "source": "",
            "points": 0,
            "min": None,
            "mean": None,
            "max": None,
            "dataevents": 0,
            "sample_freq": {},
            "multiple_freq": 0,
            "units": [],
            "scale_factor": [],
            "nan_on_time": 0,
            "nan_on_values": 0,
            "overlapped_points": 0,
            "total_overlap_bundles": 0,
            "string_value_bundles": 0,
            "defective_signal": 0,
        }

    @staticmethod
    def init_files_dict():
        return {
            "total_files": 0,
            "missing_vs": 0,
            "missing_wv": 0,
            "no_label_signal": 0,
            "multiple_label_signal": 0,
        }

    @staticmethod
    def add_percentages(dataframe, column, denominator):
        col_idx = dataframe.columns.to_list().index(column)

        if dataframe[column].dtype.name == "object":
            dataframe[column] = pd.to_numeric(dataframe[column])
        if not isinstance(denominator, int) and denominator.dtype.name == "object":
            denominator = pd.to_numeric(denominator)

        new_column = (dataframe[column] / denominator * 100).fillna(0)
        dataframe.insert(col_idx + 1, f"{column}_%", new_column)

    def add_signal_stats(self, signal, key, value=1, overwrite=False, source=None):
        if source:
            signal = f"{signal}_vs" if source == "vitals" else f"{signal}_wv"

        if signal not in self.signal_stats:
            self.signal_stats[signal] = self.init_signal_dict()

        if key not in self.signal_stats[signal]:
            raise ValueError(f"Wrong key for summary stats: {key}")

        if isinstance(self.signal_stats[signal][key], dict):
            self._increment_dict(signal, key, value)
        elif isinstance(self.signal_stats[signal][key], list):
            self._increment_list(signal, key, value)
        else:
            if overwrite:
                self.signal_stats[signal][key] = value
            else:
                self.signal_stats[signal][key] += value

    def _increment_dict(self, signal, key, value):
        if value not in self.signal_stats[signal][key]:
            self.signal_stats[signal][key][value] = 1
        else:
            self.signal_stats[signal][key][value] += 1

    def _increment_list(self, signal, key, value):
        current_values = self.signal_stats[signal][key]
        if value not in current_values:
            current_values.append(value)

    def add_file_stats(self, key):
        if key not in self.file_stats:
            raise ValueError(f"Wrong key for summary stats: {key}")

        self.file_stats[key] += 1

    def _update_mean(self, signal_index: str, add_mean: float, add_points: int):
        old_mean = self.signal_stats[signal_index]["mean"]
        if old_mean:
            old_points = self.signal_stats[signal_index]["points"]
            all_points = old_points + add_points
            new_mean = (
                old_mean * old_points / all_points + add_mean * add_points / all_points
            )
        else:
            new_mean = add_mean

        return new_mean

    def add_from_signal(self, signal: BedmasterSignal):
        signal_name = (
            f"{signal.name}_vs" if signal.source == "vitals" else f"{signal.name}_wv"
        )

        if signal_name not in self.signal_stats:
            self.signal_stats[signal_name] = self.init_signal_dict()

        self.add_signal_stats(signal_name, "files")

        for field in ["channel", "units", "scale_factor"]:
            self.add_signal_stats(signal_name, field, getattr(signal, field))
        self.add_signal_stats(signal_name, "source", signal.source, overwrite=True)

        for sample_freq, _ in signal.sample_freq:
            self.add_signal_stats(signal_name, "sample_freq", sample_freq)
        if len(signal.sample_freq) > 1:
            self.add_signal_stats(signal_name, "multiple_freq")

        old_min = self.signal_stats[signal_name]["min"]
        add_min = signal.value.min()

        new_min = min(old_min, add_min) if old_min else add_min
        self.add_signal_stats(signal_name, "min", new_min, overwrite=True)

        new_mean = self._update_mean(
            signal_name,
            signal.value.mean(),
            signal.value.size,
        )
        self.add_signal_stats(signal_name, "mean", new_mean, overwrite=True)

        old_max = self.signal_stats[signal_name]["max"]
        add_max = signal.value.max()
        new_max = max(old_min, add_min) if old_max else add_max
        self.add_signal_stats(signal_name, "max", new_max, overwrite=True)

        self.add_signal_stats(signal_name, "points", signal.value.size)

        de_num = np.where(np.unpackbits(signal.time_corr_arr))[0].size
        self.add_signal_stats(signal_name, "dataevents", de_num)

        time_nans = np.where(np.isnan(signal.time))[0].size
        self.add_signal_stats(signal_name, "nan_on_time", time_nans)
        value_nans = np.where(np.isnan(signal.value))[0].size
        self.add_signal_stats(signal_name, "nan_on_values", value_nans)

    def to_csv(self, output_dir, files_base_name):
        # Create signals dataframe
        signal_stats_df = pd.DataFrame(self.signal_stats).T
        for column in ["nan_on_time", "nan_on_values", "overlapped_points"]:
            self.add_percentages(signal_stats_df, column, signal_stats_df["points"])
        self.add_percentages(signal_stats_df, "files", self.file_stats["total_files"])
        self.add_percentages(
            signal_stats_df,
            "total_overlap_bundles",
            signal_stats_df["files"],
        )
        signal_stats_df = signal_stats_df.round(2)
        signal_stats_df = signal_stats_df.rename_axis("signal").reset_index()
        signal_stats_df = signal_stats_df.sort_values(
            by=["source", "files"],
            ascending=[False, False],
        )
        signal_stats_df["signal"] = signal_stats_df["signal"].apply(lambda x: x[:-3])

        # Save DF to csv
        signal_stats_df.to_csv(
            os.path.join(output_dir, f"{files_base_name}_bedmaster_signal_stats.csv"),
            index=False,
        )

        # Create files dataframe
        file_stats_df = pd.DataFrame(
            self.file_stats.items(),
            columns=["issue", "count"],
        )
        self.add_percentages(file_stats_df, "count", self.file_stats["total_files"])
        file_stats_df = file_stats_df.round(2)

        # Save df to csv
        file_stats_df.to_csv(
            os.path.join(output_dir, f"{files_base_name}_bedmaster_files_stats.csv"),
            index=False,
        )
