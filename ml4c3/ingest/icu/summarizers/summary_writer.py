# Imports: standard library
import os
import logging
from typing import Any, Set, Dict, List, Optional

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from ml4c3.definitions.icu import EDW_FILES, BM_SOURCES
from ml4c3.ingest.icu.readers import BMReader, EDWReader
from ml4c3.ingest.icu.statistics import BMStats


class PreTensorizeSummaryWriter:
    """
    Class that creates summary data for a set of bm and edw data.

    It is used to organize this data and create a csv with all the
    information.
    """

    def __init__(self, bm_dir: str, edw_dir: str, xref_file: str):
        self.bm_dir = bm_dir
        self.edw_dir = edw_dir
        self.xref_file = xref_file
        self.reset()
        self.xref_fields = [
            "edw_mrns",
            "bm_mrns",
            "common_mrns",
            "edw_csns",
            "bm_csns",
            "common_csns",
            "bm_files",
            "cross_referenced_bm_files",
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

    def get_mrns_and_csns(self):
        edw_mrns = {
            int(mrn)
            for mrn in os.listdir(self.edw_dir)
            if os.path.isdir(os.path.join(self.edw_dir, mrn))
        }
        edw_csns = {
            int(csn)
            for mrn in edw_mrns
            for csn in os.listdir(os.path.join(self.edw_dir, str(mrn)))
            if os.path.isdir(os.path.join(self.edw_dir, str(mrn), csn))
        }
        return edw_mrns, edw_csns

    def _cross_reference_stats(self):
        """
        Assess coverage between EDW and Bedmaster datasets.
        """

        xref_file = pd.read_csv(self.xref_file)
        xref_file = xref_file.dropna(subset=["MRN"])
        bm_mrns = set(xref_file["MRN"].unique())
        bm_csns = set(xref_file["VisitIdentifier"].unique())
        bm_ids = set(xref_file["fileID"].unique())
        cross_ref_bm_files = set()
        for bm_id in bm_ids:
            cross_ref_bm_files = cross_ref_bm_files.union(
                {
                    bm_file
                    for bm_file in os.listdir(self.bm_dir)
                    if bm_file.startswith(bm_id)
                },
            )
        bm_files = [
            bm_file for bm_file in os.listdir(self.bm_dir) if bm_file.endswith(".mat")
        ]

        edw_mrns, edw_csns = self.get_mrns_and_csns()

        self.summary["edw_mrns"] = len(edw_mrns)
        self.summary["edw_csns"] = len(edw_csns)
        self.summary["bm_mrns"] = len(bm_mrns)
        self.summary["bm_csns"] = len(bm_csns)
        self.summary["bm_files"] = len(bm_files)
        self.summary["common_mrns"] = len(edw_mrns.intersection(bm_mrns))
        self.summary["common_csns"] = len(edw_csns.intersection(bm_csns))
        self.summary["cross_referenced_mrns"] = len(edw_mrns.union(bm_mrns))
        self.summary["cross_referenced_csns"] = len(edw_csns.union(bm_csns))
        self.summary["cross_referenced_bm_files"] = len(cross_ref_bm_files)

    def _update_list_signals(self, signals, csn):
        """
        Increase signal counter.
        """
        for source in signals:
            for signal in signals[source]:
                if "EDW" in source or signal in self.signals or self.signals == ["all"]:
                    csn = csn.copy()
                    if source not in self.signals_summary:
                        self.signals_summary[source] = {signal: csn}
                    elif signal not in self.signals_summary[source]:
                        self.signals_summary[source][signal] = csn
                    else:
                        self.signals_summary[source][signal].update(csn)

    def _get_signals_stats(self):
        """
        Obtain list of signals for every file in edw_dir and bm_dir and update
        the corresponding counter using _update_list_signals.
        """
        if self.signals:
            xref_file = pd.read_csv(self.xref_file)
            xref_file = xref_file.dropna(subset=["MRN"])
            bm_files = [
                bm_file
                for bm_file in os.listdir(self.bm_dir)
                if bm_file.endswith(".mat")
            ]
            k = 0
            for bm_file in bm_files:
                k += 1
                bm_file_path = os.path.join(self.bm_dir, bm_file)
                reader = BMReader(bm_file_path)
                bm_signals = {
                    BM_SOURCES["vitals"]: reader.list_vs(),
                    BM_SOURCES["waveform"]: list(reader.list_wv()),
                }
                # pylint: disable=cell-var-from-loop
                csns = set(
                    xref_file[list(map(lambda x: x in bm_file, xref_file["fileID"]))][
                        "VisitIdentifier"
                    ].unique(),
                )
                self._update_list_signals(bm_signals, csns)
                logging.info(
                    "Obtained statistics from bm file number " f"{k}/{len(bm_files)}",
                )

        mrns = [
            mrn
            for mrn in os.listdir(self.edw_dir)
            if os.path.isdir(os.path.join(self.edw_dir, mrn))
        ]
        mrns_csns = []
        for mrn in mrns:
            mrns_csns.extend(
                [
                    (mrn, csn)
                    for csn in os.listdir(os.path.join(self.edw_dir, mrn))
                    if os.path.isdir(os.path.join(self.edw_dir, mrn, csn))
                ],
            )
        k = 0
        for mrn, csn in mrns_csns:
            k += 1
            reader = EDWReader(self.edw_dir, mrn, csn)
            edw_signals = {
                EDW_FILES["med_file"]["source"]: reader.list_medications(),
                EDW_FILES["vitals_file"]["source"]: reader.list_vitals(),
                EDW_FILES["lab_file"]["source"]: reader.list_labs(),
                EDW_FILES["surgery_file"]["source"]: reader.list_surgery(),
                EDW_FILES["other_procedures_file"][
                    "source"
                ]: reader.list_other_procedures(),
                EDW_FILES["transfusions_file"]["source"]: reader.list_transfusions(),
            }
            self._update_list_signals(edw_signals, {int(csn)})
            logging.info(
                f"Obtained statistics from csns folder number {k}/{len(mrns_csns)}",
            )

    def _get_demo_stats(self):
        """
        Obtain demographic statistics from EDW dataset.
        """
        count = 0
        mrns = [
            mrn
            for mrn in os.listdir(self.edw_dir)
            if os.path.isdir(os.path.join(self.edw_dir, mrn))
        ]
        for mrn in mrns:
            csns = [
                csn
                for csn in os.listdir(os.path.join(self.edw_dir, mrn))
                if os.path.isdir(os.path.join(self.edw_dir, mrn, csn))
            ]
            for csn in csns:
                reader = EDWReader(self.edw_dir, mrn, csn)
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
            self.summary["edw_csns"] = len(self.get_mrns_and_csns()[1])

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
            if "EDW" in source:
                total = self.summary["edw_csns"]
            else:
                total = self.summary["bm_csns"]
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

    def _get_bm_files(self):
        file_ids = {
            "_".join(bm_file.split("_")[:2])
            for bm_file in os.listdir(self.bm_dir)
            if bm_file.endswith(".mat")
        }
        bm_files = {}
        for file_id in file_ids:
            files = [
                file for file in os.listdir(self.bm_dir) if file.startswith(file_id)
            ]
            files.sort(key=lambda x: int(x.split("_")[-2]))
            bm_files[file_id] = files

        return bm_files

    def get_detailed_bm_stats(self, detailed_bm_writer):
        """
        Creates an additional stats file with detailed Bedmaster information.
        """
        bm_files = self._get_bm_files()
        total_bm_files_ids = len(bm_files)

        for idx, file_id in enumerate(bm_files):
            logging.info(f"Analyzing fileID {idx} of {total_bm_files_ids}...")
            total_files = len(bm_files[file_id])

            previous_max = None
            for file_idx, file in enumerate(bm_files[file_id]):
                logging.info(f"... file {file} : {file_idx+1} of {total_files}")
                bm_file_path = os.path.join(self.bm_dir, file)

                with BMReader(bm_file_path, summary_stats=detailed_bm_writer) as reader:
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
        output_dir: str = "./results",
        output_files_base_name: str = "pre_tensorize",
        signals: Optional[List[str]] = None,
        detailed_bm: bool = False,
        ignore_xref: bool = False,
    ):
        """
        Creates csv files with summary statistics of the date before
        tensorization.

        :param output_dir: <str> directory where output files will be saved.
        :param output_files_base_name: <str> base name of the output files.
                                       By default: pre_tensorize_summary
        :param signals: <List> list of signals to calculate bm statistics.
        :param detailed_bm: <Bool> Generate detailed Bedmaster statistics.
        :param ignore_xref: <Bool> Don't use crossreference data. Enable along
                                   with detailed_bm to create  bm_statistics
                                   without using edw_dir or  xref_file.
        """
        self.reset(signals)

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        if ignore_xref and not detailed_bm:
            logging.warning(
                "Option --no_xref requires --detailed_bm which "
                "is not set. Ignoring --no_xref setting",
            )
            ignore_xref = False

        if not ignore_xref:
            edw_df = self._get_edw_df()
            file_name = f"{output_files_base_name}_edw_demographics.csv"
            edw_df.to_csv(os.path.join(output_dir, file_name), index=False)
            logging.info(f"Demographics stats saved as {file_name}.")

            xref_df = self._get_xref_df()
            file_name = f"{output_files_base_name}_mrn_csn_coverage_edw_bm.csv"
            xref_df.to_csv(os.path.join(output_dir, file_name), index=False)
            logging.info(f"Cross reference stats saved as {file_name}.")

            signals_summary_df = self._get_signals_df()
            file_name = f"{output_files_base_name}_signals_summary.csv"
            signals_summary_df.to_csv(os.path.join(output_dir, file_name), index=False)
            logging.info(f"Signals stats saved as {file_name}.")

        if detailed_bm:
            logging.info("Starting detailed report. This might take a long time")
            detailed_bm_writer = BMStats()
            self.get_detailed_bm_stats(detailed_bm_writer)
            detailed_bm_writer.to_csv(output_dir, output_files_base_name)
