# Imports: standard library
import os
import logging
from typing import Dict, List

# Imports: third party
import h5py
import numpy as np
import pandas as pd

# Imports: first party
from ml4c3.tensormap.TensorMap import TensorMap, update_tmaps
from ml4c3.tensormap.tensor_maps_icu_static import get_tmap as GET_STATIC_TMAP

# pylint: disable=too-many-branches


class ExploreWrapper:
    """
    Class that creates summary data for a set of hd5 files.
    """

    def __init__(
        self,
        input_tensors: List[str],
        output_dir: str,
        output_files_prefix: str,
        no_csns: bool,
        n_workers: int,
    ):
        """
        :param input_tensors: <List[str]> List of tensor maps names to calculate
                              their summary_statistics.
        :param output_dir: <str> Directory where the set of hd5 files is stored.
        :param output_files_prefix: <str> Base name of the summary stats .csv files.
        :param no_csns: <bool> Bool indicating if it loops through csns or not.
        :param n_workers: <int> Number of workers used in parallel processing.
        """
        self.output_dir = output_dir
        self.output_files_prefix = output_files_prefix
        self.no_csns = no_csns
        self.n_workers = n_workers

        self.df_continuous = pd.DataFrame()
        self.df_categorical = pd.DataFrame()
        self.df_language = pd.DataFrame()
        self.df_event = pd.DataFrame()
        self.df_timeseries = pd.DataFrame()
        self.df_union = pd.DataFrame()
        self.df_intersect = pd.DataFrame()

        self.tmaps: Dict[str, TensorMap] = {}
        for tensor in input_tensors:
            try:
                self.tmaps = update_tmaps(tensor, self.tmaps)
            except KeyError:
                logging.info(
                    f"Tensor map {tensor} doesn't exist. "
                    "No statistics will be performed for this tensor.",
                )

    @staticmethod
    def _remove_nans(array):
        """
        Remove nan values from an array.
        """
        try:
            idx = ~np.isnan(array)
        except TypeError:
            idx = np.where(array.astype(str) != str(np.nan))[0]
        return array[idx]

    def explore(self, tmaps: Dict[str, TensorMap]):
        """
        Given a list of tensor maps, obtains the data associated to each tensor
        and computes basic statistics for each of the hd5 files in
        self.output_dir.
        """
        continuous = [tmaps[tm] for tm in tmaps if tmaps[tm].is_continuous]
        categorical = [tmaps[tm] for tm in tmaps if tmaps[tm].is_categorical]
        language = [tmaps[tm] for tm in tmaps if tmaps[tm].is_language]
        event = [tmaps[tm] for tm in tmaps if tmaps[tm].is_event]
        timeseries = [tmaps[tm] for tm in tmaps if tmaps[tm].is_timeseries]

        hd5_files_dir = os.path.join(self.output_dir)
        hd5_files = [
            hd5_file
            for hd5_file in os.listdir(hd5_files_dir)
            if hd5_file.endswith(".hd5")
        ]
        if len(hd5_files) == 0:
            raise FileNotFoundError(
                f"No hd5 files found in the given directory: {self.output_dir}",
            )

        k = 0
        demo_stats = self._create_demo_stats()
        for hd5_file in hd5_files:
            hd5_file_path = os.path.join(self.output_dir, hd5_file)
            hd5 = h5py.File(hd5_file_path, "r")
            mrn = hd5_file[:-4]
            visits = GET_STATIC_TMAP("visits").tensor_from_file(
                GET_STATIC_TMAP("visits"),
                hd5,
            )
            demo_stats["unique_mrns"] += 1
            if self.no_csns:
                stats = {"mrn": mrn}
                for tm in continuous:
                    self._continuous_statistics(tm, hd5, stats)
                for tm in categorical:
                    self._categorical_language_statistics(tm, hd5, stats)
                for tm in language:
                    self._categorical_language_statistics(tm, hd5, stats)
                for tm in event:
                    self._event_statistics(tm, hd5, stats)
                for tm in timeseries:
                    self._timeseries_statistics(tm, hd5, stats)
                self._update_demo_stats(demo_stats, hd5, visits[0])
            else:
                for csn in visits:
                    stats = {"mrn": mrn, "csn": csn}
                    for tm in continuous:
                        self._continuous_statistics(tm, hd5, stats, csn)
                    for tm in categorical:
                        self._categorical_language_statistics(tm, hd5, stats, csn)
                    for tm in language:
                        self._categorical_language_statistics(tm, hd5, stats, csn)
                    for tm in event:
                        self._event_statistics(tm, hd5, stats, csn)
                    for tm in timeseries:
                        self._timeseries_statistics(tm, hd5, stats, csn)
                    self._update_demo_stats(demo_stats, hd5, csn)
            k += 1
            logging.info(f"Obtained statistics from file number {k}/{len(hd5_files)}")
        self._save_df(demo_stats, "hd5_demographics")

    def _continuous_statistics(self, tm, hd5, stats, csn=None):
        key = tm.name
        try:
            if csn:
                values = tm.tensor_from_file(tm, hd5, visits=csn)[0]
            else:
                values = tm.tensor_from_file(tm, hd5)
        except KeyError:
            return
        stats["total"] = values.shape[0]
        values = self._remove_nans(values)
        if values.shape[0] > 0:
            stats["min"] = np.min(values)
            stats["max"] = np.max(values)
            stats["mean"] = np.mean(values)
            stats["std"] = np.std(values)
            stats["count"] = values.shape[0]
        self.df_continuous = pd.concat(
            [self.df_continuous, pd.DataFrame([stats], index=[key])],
        )

    def _categorical_language_statistics(self, tm, hd5, stats, csn=None):
        key = tm.name
        try:
            if csn:
                values = tm.tensor_from_file(tm, hd5, visits=csn)[0]
            else:
                values = tm.tensor_from_file(tm, hd5)
        except KeyError:
            return
        stats["total"] = values.shape[0]
        values = self._remove_nans(values)
        stats["count"] = values.shape[0]
        stats["list_unique"] = set(tm.channel_map.keys())
        if tm.is_categorical:
            for channel_map in tm.channel_map:
                values_cm = values[values == tm.channel_map[channel_map]]
                stats[f"count_{channel_map}"] = values_cm.shape[0]
            self.df_categorical = pd.concat(
                [self.df_categorical, pd.DataFrame([stats], index=[key])],
            )
        elif tm.is_language:
            self.df_language = pd.concat(
                [self.df_language, pd.DataFrame([stats], index=[key])],
            )

    def _event_statistics(self, tm, hd5, stats, csn=None):
        key = tm.name
        try:
            if csn:
                times = tm.tensor_from_file(tm, hd5, visits=csn)[0]
            else:
                times = tm.tensor_from_file(tm, hd5)
        except KeyError:
            return
        stats["total"] = times.shape[0]
        times = self._remove_nans(times)
        stats["first"] = np.min(times)
        stats["last"] = np.max(times)
        stats["count"] = times.shape[0]
        self.df_event = pd.concat([self.df_event, pd.DataFrame([stats], index=[key])])

    def _timeseries_statistics(self, tm, hd5, stats, csn=None):
        key = tm.name
        try:
            if csn:
                values, times = tm.tensor_from_file(tm, hd5, visits=csn)[0]
            else:
                values, times = tm.tensor_from_file(tm, hd5)
        except KeyError:
            return
        stats["total"] = values.shape[0]
        values = self._remove_nans(values)
        stats["min"] = np.min(values)
        stats["max"] = np.max(values)
        stats["mean"] = np.mean(values)
        stats["std"] = np.std(values)
        stats["count"] = values.shape[0]
        stats["first"] = np.nanmin(times)
        stats["last"] = np.nanmax(times)
        self.df_timeseries = pd.concat(
            [self.df_timeseries, pd.DataFrame([stats], index=[key])],
        )

    def list_mrns(self):
        """
        Creates two .csv files with a list of mrns with any of the tensors and
        a list of mrns with all the tensor specified when running `explore`
        function.

        This function should be runned after runing `explore` function.
        """
        data_frames = [
            self.df_continuous,
            self.df_categorical,
            self.df_language,
            self.df_event,
            self.df_timeseries,
        ]

        for data_frame in data_frames:
            for signal, row in data_frame.iterrows():
                if row.mrn in self.df_union.index and signal in self.df_union.columns:
                    self.df_union.at[row.mrn, signal] = 1
                elif row.mrn in self.df_union.index:
                    self.df_union = pd.concat(
                        [self.df_union, pd.DataFrame([{signal: 1}], index=[row.mrn])],
                        axis=1,
                    )
                else:
                    self.df_union = pd.concat(
                        [self.df_union, pd.DataFrame([{signal: 1}], index=[row.mrn])],
                        axis=0,
                    )
        self.df_intersect = self.df_union.dropna()

        for data_frame, name in [
            (self.df_union, "union"),
            (self.df_intersect, "intersection"),
        ]:
            demo_stats = self._create_demo_stats()
            for index, _row in data_frame.iterrows():
                hd5_file = f"{index}.hd5"
                hd5_file_path = os.path.join(self.output_dir, hd5_file)
                hd5 = h5py.File(hd5_file_path, "r")
                visits = GET_STATIC_TMAP("visits").tensor_from_file(
                    GET_STATIC_TMAP("visits"),
                    hd5,
                )
                demo_stats["unique_mrns"] += 1
                for csn in visits:
                    self._update_demo_stats(demo_stats, hd5, csn)
            if not data_frame.empty:
                self._save_df(data_frame, f"mrns_list_{name}")
                self._save_df(demo_stats, f"hd5_{name}_demographics")

    def compute_statistics(self, mrns_df: pd.DataFrame, name: str):
        """
        Computes statistics for each tensor using the results obtained in
        `explore` function. This function should be runned after runing
        `explore` function.

        :param mrns_df: <pd.DataFrame> DataFrame whose indixes contains the mrns to
                        calculate the summary stats.
        :param name: <str> Name of the output file.
        """
        data_frames = {
            "continuous": self.df_continuous,
            "categorical": self.df_categorical,
            "language": self.df_language,
            "event": self.df_event,
            "timeseries": self.df_timeseries,
        }

        for key, data_frame in data_frames.items():
            stats_df = pd.DataFrame()
            for signal in data_frame.index.unique():
                signal_df = data_frame.loc[signal]
                if isinstance(signal_df, pd.Series):
                    signal_df = signal_df.to_frame().transpose()
                signal_df = signal_df[signal_df["mrn"].isin(mrns_df.index.unique())]
                if signal_df.empty:
                    continue
                stats = self._compute_basic_statistics(signal_df)
                if key in ["continuous", "timeseries"]:
                    stats.update(self._compute_tendency_statistics(signal_df))
                if key in ["categorical", "language"]:
                    stats.update(self._compute_unique_statistics(signal_df))
                if key in ["categorical"]:
                    for channel_map in self.tmaps[signal].channel_map:  # type: ignore
                        stats[f"count_{channel_map}"] = signal_df[
                            f"count_{channel_map}"
                        ].sum()
                if key in ["event", "timeseries"]:
                    stats.update(self._compute_temporal_statistics(signal_df))
                stats_df = pd.concat([stats_df, pd.DataFrame([stats], index=[signal])])
            self._save_df(stats_df, f"{key}_signals_{name}")

    @staticmethod
    def _compute_basic_statistics(signal_df):
        stats = {}
        stats["total"] = signal_df["total"].sum()
        stats["count"] = signal_df["count"].sum()
        stats["missing"] = stats["total"] - stats["count"]
        stats["missing_%"] = stats["missing"] / stats["total"] * 100
        return stats

    @staticmethod
    def _compute_temporal_statistics(signal_df):
        stats = {}
        stats["first"] = signal_df["first"].min()
        stats["last"] = signal_df["last"].max()
        return stats

    @staticmethod
    def _compute_tendency_statistics(signal_df):
        stats = {}
        stats["mean"] = (signal_df["mean"] * signal_df["count"]).sum() / signal_df[
            "count"
        ].sum()
        stats["min"] = signal_df["min"].min()
        stats["max"] = signal_df["max"].max()
        count = 0
        mean = 0
        variance = 0
        for _, row in signal_df.iterrows():
            if np.isnan(row["std"]) or np.isnan(row["mean"]) or np.isnan(row["count"]):
                continue
            if count == 0:
                variance = row["std"] ** 2
                mean = row["mean"]
                count = row["count"]
                continue
            variance = (
                (count - 1) * variance + (row["count"] - 1) * row["std"] ** 2
            ) / (count + row["count"] - 1) + (
                count * row["count"] * (mean - row["mean"]) ** 2
            ) / (
                (count + row["count"]) * (count + row["count"] - 1)
            )
            mean = (mean * count + row["mean"] * row["count"]) / (count + row["count"])
            count += row["count"]
        stats["std"] = variance ** (1 / 2)
        return stats

    @staticmethod
    def _compute_unique_statistics(signal_df):
        stats = {}
        list_unique = set()
        for _index, row in signal_df.iterrows():
            list_unique = list_unique.union(row["list_unique"])
        stats["list_unique"] = list_unique
        stats["count_unique"] = len(stats["list_unique"])
        return stats

    def _save_df(self, data_frame, name):
        if isinstance(data_frame, dict):
            for key in data_frame:
                data_frame[key] = [data_frame[key]]
            data_frame = pd.DataFrame.from_dict(data_frame)
        if not data_frame.empty:
            fpath = os.path.join(
                self.output_dir,
                f"{self.output_files_prefix}_{name}.csv",
            )
            data_frame = data_frame.round(2)
            data_frame.to_csv(fpath, index=True)

    def write_summary_stats(self):
        """
        Pipeline that creates the following .csv files:

        - Three demographics tables. One for the whole data set in self.output_dir,
        another for the mrns having at least one tensor specified in self.tmaps
        and the last one for the mrns having all tensors.
        - Two statistics tables for each type of tensor (continuous, timeseries,
        event, language or categorical).
        One corresponding to the list of mrns having at least one tensor and the other
        one corresponding to the list of mrns having all the tensor.
        """
        self.explore(self.tmaps)
        self.list_mrns()
        self.compute_statistics(self.df_union, "union")
        self.compute_statistics(self.df_intersect, "intersection")

    @staticmethod
    def _create_demo_stats():
        """
        Creates a default demo summary dict.

        :return demo_stats: <Dict[str, Any]> Dictionary with the keys to compute
                            demographics statistics with _update_demo_stats.
        """
        demo_stats = {
            "unique_mrns": 0,
            "unique_csns": 0,
            "Male": 0,
            "Female": 0,
            "Deceased": 0,
            "Alive": 0,
            "min_age": np.inf,
            "max_age": 0,
            "mean_age": 0,
            "min_weight": np.inf,
            "max_weight": 0,
            "mean_weight": 0,
            "min_height": np.inf,
            "max_height": 0,
            "mean_height": 0,
            "min_length_stay": np.inf,
            "max_length_stay": 0,
            "mean_length_stay": 0,
            "earliest_transfer_in": pd.to_datetime("2200"),
            "latest_transfer_in": pd.to_datetime("1900"),
        }
        return demo_stats

    @staticmethod
    def _update_demo_stats(demo_stats: Dict[str, int], hd5_file: str, csn: int):
        """
        Update the demo summary data of a set of hd5 files with the data stored in
        a hd5 file.

        :param demo_stats: <Dict[str, int]> Demo summary data.
        :param hd5_file: <str> a hd5 file in reader mode.
        :param csn: <int> Csn to extract the data from the hd5_file.
        """
        age = pd.to_datetime(
            GET_STATIC_TMAP("birth_date").tensor_from_file(
                GET_STATIC_TMAP("birth_date"),
                hd5_file,
                visits=csn,
            )[0],
        )
        transfer_in = pd.to_datetime(
            GET_STATIC_TMAP("admin_date").tensor_from_file(
                GET_STATIC_TMAP("admin_date"),
                hd5_file,
                visits=csn,
            )[0],
        )
        transfer_out = pd.to_datetime(
            GET_STATIC_TMAP("end_date").tensor_from_file(
                GET_STATIC_TMAP("end_date"),
                hd5_file,
                visits=csn,
            )[0],
        )
        sex_map = GET_STATIC_TMAP("sex").tensor_from_file(
            GET_STATIC_TMAP("sex"),
            hd5_file,
            visits=csn,
        )[0]
        sex = "uknown"
        for channel_map, index in GET_STATIC_TMAP("sex").channel_map.items():
            if sex_map[index]:
                sex = channel_map.capitalize()
                break
        weight = GET_STATIC_TMAP("weight").tensor_from_file(
            GET_STATIC_TMAP("weight"),
            hd5_file,
            visits=csn,
        )[0]
        height = GET_STATIC_TMAP("height").tensor_from_file(
            GET_STATIC_TMAP("height"),
            hd5_file,
            visits=csn,
        )[0]
        end_stay_type = (
            GET_STATIC_TMAP("end_stay_type")
            .tensor_from_file(GET_STATIC_TMAP("end_stay_type"), hd5_file, visits=csn)[0]
            .capitalize()
        )

        age = (
            transfer_in.year
            - age.year
            - ((transfer_in.month, transfer_in.day) < (age.month, age.day))
        )
        if str(transfer_out) != "NaT":
            length_stay = (
                float(
                    transfer_out.to_datetime64() - transfer_in.to_datetime64(),
                )
                / (10 ** 9 * 60 * 60)
            )
        else:
            length_stay = np.nan

        demo_stats["unique_csns"] += 1
        demo_stats[sex] += 1
        demo_stats[end_stay_type] += 1
        for key, value in [
            ("age", age),
            ("weight", weight),
            ("height", height),
            ("length_stay", length_stay),
        ]:
            demo_stats[f"min_{key}"] = min([demo_stats[f"min_{key}"], value])
            demo_stats[f"max_{key}"] = max([demo_stats[f"max_{key}"], value])
            if not np.isnan(value):
                demo_stats[f"mean_{key}"] = (
                    demo_stats[f"mean_{key}"] * (demo_stats["unique_mrns"] - 1) + value
                ) / demo_stats["unique_mrns"]
        demo_stats["earliest_transfer_in"] = min(
            demo_stats["earliest_transfer_in"],
            transfer_in,
        )
        demo_stats["latest_transfer_in"] = max(
            demo_stats["latest_transfer_in"],
            transfer_in,
        )


def explore_icu(args):
    explorer = ExploreWrapper(
        input_tensors=args.input_tensors,
        output_dir=args.tensors,
        output_files_prefix=args.output_files_prefix,
        no_csns=args.no_csns,
        n_workers=args.num_workers,
    )
    explorer.write_summary_stats()
