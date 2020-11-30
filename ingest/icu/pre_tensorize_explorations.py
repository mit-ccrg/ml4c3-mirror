# Imports: standard library
import os
import logging
from typing import Any, Set, Dict, List, Optional

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from definitions.icu import EDW_FILES, BEDMASTER_EXT
from ingest.icu.utils import get_files_in_directory
from ingest.icu.readers import EDWReader, BedmasterReader
from ingest.icu.bedmaster_stats import BedmasterStats

EXPECTED_FILES = []
for FILE in EDW_FILES:
    EXPECTED_FILES.append(EDW_FILES[FILE]["name"])
EXPECTED_FILES.remove(EDW_FILES["adt_file"]["name"])


class PreTensorizeExplorer:
    """
    Class that creates summary data for a set of Bedmaster and edw data.

    It is used to organize this data and create a csv with all the
    information.
    """

    def __init__(
        self,
        path_bedmaster: str,
        path_edw: str,
        path_xref: str,
        patient_csv: str,
    ):
        self.path_bedmaster = path_bedmaster
        self.path_edw = path_edw
        self.path_xref = path_xref
        self.reset()
        self.xref_fields = [
            "edw_mrns",
            "bedmaster_mrns",
            "common_mrns",
            "edw_csns",
            "bedmaster_csns",
            "common_csns",
            "bedmaster_files",
            "cross_referenced_bedmaster_files",
        ]
        self.edw_fields = [
            "Male",
            "Female",
            "Deceased",
            "Alive",
            "age",
            "weight[pounds]",
            "height[m]",
            "length_stay[h]",
            "transfer_in",
        ]
        self.signal_fields = ["signal", "count", "total", "%", "source"]
        self.edw_mrns, self.edw_csns, self.edw_mrns_csns = self.get_mrns_and_csns(
            patient_csv,
        )

    def reset(self, signals: Any = None):
        """
        Function to reset some parameters.

        :param signals: <List[str]> list of signals to calculate their
                        summary statistics. If no signal is specified,
                        statistics for all signals are calculated.
        """
        self.signals_summary: Dict[str, Dict[str, Set[int]]] = {}
        self.signals = signals
        self.summary = {
            "Male": 0,
            "Female": 0,
            "Deceased": 0,
            "Alive": 0,
            "max_age": 0,
            "min_age": np.inf,
            "mean_age": 0,
            "max_weight": 0,
            "min_weight": np.inf,
            "mean_weight": 0,
            "max_height": 0,
            "min_height": np.inf,
            "mean_height": 0,
            "max_length_stay": 0,
            "min_length_stay": np.inf,
            "mean_length_stay": 0,
            "earliest_transfer_in": pd.to_datetime("2200"),
            "latest_transfer_in": pd.to_datetime("1900"),
        }

    def get_mrns_and_csns(self, patient_csv: str):
        edw_mrns = set()
        edw_csns = set()
        edw_mrns_csns = set()
        df = pd.read_csv(patient_csv)
        if "PatientEncounterID" in df.columns:
            for _, row in df.iterrows():
                mrn = str(int(row["MRN"]))
                csn = str(int(row["PatientEncounterID"]))
                fpath = os.path.join(self.path_edw, mrn, csn)
                if not os.path.isdir(fpath):
                    continue
                csv_files = [
                    csv_file
                    for csv_file in os.listdir(fpath)
                    if csv_file.endswith(".csv")
                ]
                for expected_file in EXPECTED_FILES:
                    if expected_file not in csv_files:
                        break
                else:
                    edw_mrns.add(int(mrn))
                    edw_csns.add(int(csn))
                    edw_mrns_csns.add((mrn, csn))
        else:
            mrns = list(df["MRN"].unique())
            for mrn in mrns:
                mrn = str(int(mrn))
                if not os.path.isdir(os.path.join(self.path_edw, mrn)):
                    continue
                csns = os.listdir(os.path.join(self.path_edw, mrn))
                for csn in csns:
                    fpath = os.path.join(self.path_edw, mrn, csn)
                    if not os.path.isdir(fpath):
                        continue
                    csv_files = [
                        csv_file
                        for csv_file in os.listdir(
                            os.path.join(fpath),
                        )
                        if csv_file.endswith(".csv")
                    ]
                    for expected_file in EXPECTED_FILES:
                        if expected_file not in csv_files:
                            break
                    else:
                        edw_mrns.add(int(mrn))
                        edw_csns.add(int(csn))
                        edw_mrns_csns.add((mrn, csn))
        return edw_mrns, edw_csns, edw_mrns_csns

    def _cross_reference_stats(self):
        """
        Assess coverage between EDW and Bedmaster datasets.
        """
        xref_file = pd.read_csv(self.path_xref)
        xref_file = xref_file.dropna(subset=["MRN"])
        bedmaster_mrns = set(xref_file["MRN"].unique())
        bedmaster_csns = set(xref_file["PatientEncounterID"].unique())
        cross_ref_bedmaster_files = set(xref_file["fileID"].unique())

        bedmaster_files, _ = get_files_in_directory(
            directory=self.path_bedmaster,
            extension=BEDMASTER_EXT,
        )

        self.summary["edw_mrns"] = len(self.edw_mrns)
        self.summary["edw_csns"] = len(self.edw_csns)
        self.summary["bedmaster_mrns"] = len(bedmaster_mrns)
        self.summary["bedmaster_csns"] = len(bedmaster_csns)
        self.summary["bedmaster_files"] = len(bedmaster_files)
        self.summary["common_mrns"] = len(self.edw_mrns.intersection(bedmaster_mrns))
        self.summary["common_csns"] = len(self.edw_csns.intersection(bedmaster_csns))
        self.summary["cross_referenced_mrns"] = len(self.edw_mrns.union(bedmaster_mrns))
        self.summary["cross_referenced_csns"] = len(self.edw_csns.union(bedmaster_csns))
        self.summary["cross_referenced_bedmaster_files"] = len(
            cross_ref_bedmaster_files,
        )

    def _update_list_signals(self, signals, csn):
        """
        Increase signal counter.
        """
        for source in signals:
            for signal in signals[source]:
                csn = csn.copy()
                if source not in self.signals_summary:
                    self.signals_summary[source] = {signal: csn}
                elif signal not in self.signals_summary[source]:
                    self.signals_summary[source][signal] = csn
                else:
                    self.signals_summary[source][signal].update(csn)

    def _get_signals_stats(self):
        """
        Obtain list of signals for every file in path_edw and path_bedmaster and update
        the corresponding counter using _update_list_signals.
        """
        if self.signals:
            xref_file = pd.read_csv(self.path_xref)
            xref_file = xref_file.dropna(subset=["MRN"])
            k = 0
            bedmaster_files_paths, _ = get_files_in_directory(
                directory=self.path_bedmaster,
                extension=BEDMASTER_EXT,
            )
            for bedmaster_file_path in bedmaster_files_paths:
                k += 1
                try:
                    reader = BedmasterReader(bedmaster_file_path)
                except OSError:
                    continue
                bedmaster_signals = {
                    "vitals": reader.list_vs(),
                    "waveform": list(reader.list_wv()),
                }
                file = os.path.split(bedmaster_file_path)[-1]
                # pylint: disable=cell-var-from-loop
                csns = set(
                    xref_file[list(map(lambda x: x in file, xref_file["fileID"]))][
                        "PatientEncounterID"
                    ].unique(),
                )
                self._update_list_signals(bedmaster_signals, csns)
                logging.info(
                    "Obtained statistics from Bedmaster file number "
                    f"{k}/{len(bedmaster_files_paths)}.",
                )

        k = 0
        for mrn, csn in self.edw_mrns_csns:
            k += 1
            reader = EDWReader(self.path_edw, mrn, csn)
            edw_signals = {
                "med": reader.list_medications(),
                "flowsheet": reader.list_vitals(),
                "labs": reader.list_labs(),
                "surgery": reader.list_surgery(),
                "procedures": reader.list_other_procedures(),
                "transfusions": reader.list_transfusions(),
            }
            self._update_list_signals(edw_signals, {int(csn)})
            logging.info(
                "Obtained statistics from csns folder number "
                f"{k}/{len(self.edw_mrns_csns)}",
            )

    def _get_demo_stats(self):
        """
        Obtain demographic statistics from EDW dataset.
        """
        count = 0
        for mrn, csn in self.edw_mrns_csns:
            reader = EDWReader(self.path_edw, mrn, csn)
            static_data = reader.get_static_data()
            sex = static_data.sex
            end_stay_type = static_data.end_stay_type
            weight = static_data.weight
            height = static_data.height
            age = pd.to_datetime(static_data.birth_date)
            transfer_in = pd.to_datetime(static_data.admin_date)
            age = (
                transfer_in.year
                - age.year
                - ((transfer_in.month, transfer_in.day) < (age.month, age.day))
            )
            if str(static_data.end_date) != "nan":
                length_stay = (
                    float(
                        np.datetime64(static_data.end_date)
                        - transfer_in.to_datetime64(),
                    )
                    / (10 ** 9 * 60 * 60)
                )
            else:
                length_stay = np.nan

            count += 1
            self.summary[sex] += 1
            self.summary[end_stay_type] += 1
            for key, value in [
                ("age", age),
                ("weight", weight),
                ("height", height),
                ("length_stay", length_stay),
            ]:
                if not np.isnan(value):
                    self.summary[f"min_{key}"] = np.nanmin(
                        [self.summary[f"min_{key}"], value],
                    )
                    self.summary[f"max_{key}"] = np.nanmax(
                        [self.summary[f"max_{key}"], value],
                    )
                    self.summary[f"mean_{key}"] = (
                        self.summary[f"mean_{key}"] * (count - 1) + value
                    ) / count
            self.summary["earliest_transfer_in"] = min(
                self.summary["earliest_transfer_in"],
                transfer_in,
            )
            self.summary["latest_transfer_in"] = max(
                self.summary["latest_transfer_in"],
                transfer_in,
            )

    def _get_xref_df(self):
        logging.info("Obtaining cross reference stats...")
        self._cross_reference_stats()
        logging.info("Cross reference stats obtained.")

        xref_rows = []
        for field in self.xref_fields:
            xref_rows.append([field, self.summary[field]])

        columns = ["field", "count"]
        cross_ref_summary_df = pd.DataFrame(xref_rows, columns=columns).round(3)
        return cross_ref_summary_df

    def _get_edw_df(self):
        def _add_edw_row(
            field,
            count=None,
            total=None,
            percent=None,
            min_value=None,
            max_value=None,
            mean_value=None,
        ):
            return [
                field,
                count,
                total,
                percent,
                min_value,
                max_value,
                mean_value,
            ]

        logging.info("Obtaining demographics stats...")
        self._get_demo_stats()
        logging.info("Demographics stats obtained.")
        edw_rows = []

        if "edw_csns" not in self.summary:
            self.summary["edw_csns"] = len(self.edw_csns)

        total = self.summary["edw_csns"]

        for field in self.edw_fields[:4]:
            field_count = self.summary[field]
            percent = (field_count / total) * 100 if total else 0
            edw_rows.append([field, field_count, total, percent])
        for field in self.edw_fields[4:-1]:
            field_name = field.split("[")[0]
            edw_rows.append(
                _add_edw_row(
                    field,
                    total=total,
                    min_value=self.summary[f"min_{field_name}"],
                    max_value=self.summary[f"max_{field_name}"],
                    mean_value=self.summary[f"mean_{field_name}"],
                ),
            )
        edw_rows.append(
            _add_edw_row(
                self.edw_fields[-1],
                total=self.summary["edw_csns"],
                min_value=self.summary["earliest_transfer_in"],
                max_value=self.summary["latest_transfer_in"],
            ),
        )
        columns = ["field", "count", "total", "%", "min", "max", "mean"]
        edw_df = pd.DataFrame(edw_rows, columns=columns).round(3)
        return edw_df

    def _get_signals_df(self):
        logging.info("Obtaining signals stats...")
        self._get_signals_stats()
        logging.info("Signals stats obtained.")
        signals_summary: List[Dict[str, Any]] = []
        for source in self.signals_summary:
            if source in [
                "med",
                "flowsheet",
                "labs",
                "surgery",
                "procedures",
                "transfusions",
            ]:
                total = self.summary["edw_csns"]
            elif source in ["vitals", "waveform"]:
                total = self.summary["bedmaster_csns"]
            else:
                raise ValueError(f"{type(signal)} is not EDWType or BedmasterType")
            for signal in self.signals_summary[source]:
                count = len(self.signals_summary[source][signal])
                percent = count / total * 100 if total else 0
                signals_summary.append(
                    {
                        "signal": signal,
                        "count": len(self.signals_summary[source][signal]),
                        "total": total,
                        "%": percent,
                        "source": source,
                    },
                )
        columns = ["signal", "count", "total", "%", "source"]
        signals_summary_df = pd.DataFrame(signals_summary, columns=columns).round(3)
        signals_summary_df = signals_summary_df.sort_values(by=["signal"])
        return signals_summary_df

    def _get_bedmaster_files(self):
        bedmaster_files_dict = {}
        bedmaster_files, _ = get_files_in_directory(
            directory=self.path_bedmaster,
            extension=BEDMASTER_EXT,
        )
        for bedmaster_file in bedmaster_files:
            file_id = "_".join(os.path.split(bedmaster_file)[-1].split("_")[:2])
            if file_id in bedmaster_files_dict:
                bedmaster_files_dict[file_id].append(bedmaster_file)
            else:
                bedmaster_files_dict[file_id] = [bedmaster_file]
        return bedmaster_files_dict

    def get_detailed_bedmaster_stats(self, detailed_bedmaster_writer):
        """
        Creates an additional stats file with detailed Bedmaster information.
        """
        bedmaster_files = self._get_bedmaster_files()
        total_bedmaster_files_ids = len(bedmaster_files)

        for idx, file_id in enumerate(bedmaster_files):
            logging.info(f"Analyzing fileID {idx} of {total_bedmaster_files_ids}...")
            total_files = len(bedmaster_files[file_id])

            previous_max = None
            for file_idx, bedmaster_file_path in enumerate(bedmaster_files[file_id]):
                logging.info(
                    f"... file {bedmaster_file_path}: {file_idx+1} of {total_files}",
                )

                with BedmasterReader(
                    bedmaster_file_path,
                    summary_stats=detailed_bedmaster_writer,
                ) as reader:
                    if previous_max:
                        reader.get_interbundle_correction(previous_max)

                    vitalsigns = reader.list_vs()
                    for vital in vitalsigns:
                        reader.get_vs(vital)

                    waveforms = reader.list_wv()
                    for waveform, channel in waveforms.items():
                        reader.get_wv(channel, waveform)

                    previous_max = reader.max_segment

    def write_pre_tensorize_summary(
        self,
        output_folder: str = "./results",
        summary_stats_base_name: str = "pre_tensorize",
        signals: Optional[List[str]] = None,
        detailed_bedmaster: bool = False,
        no_xref: bool = False,
    ):
        """
        Creates csv files with summary statistics of the date before tensorization.

        :param output_folder: <str> directory where output files will be saved.
        :param summary_stats_base_name: <str> base name of the output files.
                                        By default: pre_tensorize_summary
        :param signals: <List> list of signals to calculate Bedmaster statistics.
        :param detailed_bedmaster: <Bool> Generate detailed Bedmaster statistics.
        :param no_xref: <Bool> Don't use crossreference data. Enable along with
                               detailed_bedmaster to create  bedmaster_statistics
                               without using path_edw or  xref_file.
        """
        self.reset(signals)

        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        if no_xref and not detailed_bedmaster:
            logging.warning(
                "Option --no_xref requires --detailed_bedmaster which "
                "is not set. Ignoring --no_xref setting.",
            )
            no_xref = False

        if not no_xref:
            edw_df = self._get_edw_df()
            file_name = f"{summary_stats_base_name}_edw_demographics.csv"
            edw_df.to_csv(os.path.join(output_folder, file_name), index=False)
            logging.info(f"Demographics stats saved as {file_name}.")

            xref_df = self._get_xref_df()
            file_name = f"{summary_stats_base_name}_coverage.csv"
            xref_df.to_csv(os.path.join(output_folder, file_name), index=False)
            logging.info(f"Cross reference stats saved as {file_name}.")

            signals_summary_df = self._get_signals_df()
            file_name = f"{summary_stats_base_name}_signals_summary.csv"
            signals_summary_df.to_csv(
                os.path.join(output_folder, file_name),
                index=False,
            )
            logging.info(f"Signals stats saved as {file_name}.")

        if detailed_bedmaster:
            logging.info("Starting detailed report. This might take a long time")
            detailed_bedmaster_writer = BedmasterStats()
            self.get_detailed_bedmaster_stats(detailed_bedmaster_writer)
            detailed_bedmaster_writer.to_csv(output_folder, summary_stats_base_name)


def pre_tensorize_explore(args):
    explorer = PreTensorizeExplorer(
        path_bedmaster=args.path_bedmaster,
        path_edw=args.path_edw,
        path_xref=args.path_xref,
        patient_csv=args.patient_csv,
    )
    explorer.write_pre_tensorize_summary(
        output_folder=args.output_folder,
        summary_stats_base_name=args.summary_stats_base_name,
        signals=args.signals,
        detailed_bedmaster=args.detailed_bedmaster,
        no_xref=args.no_xref,
    )
