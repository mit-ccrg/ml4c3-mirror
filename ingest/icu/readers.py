# Imports: standard library
import os
import re
import logging
from typing import Dict, List, Tuple
from collections import defaultdict

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from ml4c3.utils import get_unix_timestamps
from definitions.edw import EDW_FILES
from tensorize.bedmaster.match_patient_bedmaster import PatientBedmasterMatcher


def _dl():
    return defaultdict(list)


def _ddl():
    return defaultdict(_dl)


class CrossReferencer:
    """
    Class that cross-references Bedmaster and EDW data.

    Used to ensure correspondence between the data.
    """

    def __init__(
        self,
        bedmaster_dir: str,
        edw_dir: str,
        xref_file: str,
        adt: str,
        bedmaster_index: str = None,
    ):
        self.bedmaster_dir = bedmaster_dir
        self.edw_dir = edw_dir
        self.xref_file = xref_file
        self.adt = adt
        self.bedmaster_index = bedmaster_index
        self.crossref: Dict[
            str,
            Dict[str, Dict[Tuple[float, float], List[str]]],
        ] = defaultdict(_ddl)

    def get_xref_files(
        self,
        mrns: List[str] = None,
        starting_time: int = None,
        ending_time: int = None,
        overwrite_hd5: bool = True,
        n_patients: int = None,
        tensors: str = None,
        allow_one_source: bool = False,
    ) -> Dict[str, Dict[str, Dict[Tuple[float, float], List[str]]]]:
        """
        Get the cross-referenced Bedmaster files and EDW files.

        The output dictionary will have the format:

        {
            "MRN1": {
                "visitID": {
                    (xfer_in, xfer_out): [bedmaster_files],
                    (xfer_in2, xfer_out2): ...,
                },
                "visitID2": ...,
            },
            "MRN2: ...,
         }

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
        self.crossref = defaultdict(_ddl)
        if not os.path.exists(self.xref_file):
            if self.bedmaster_index is None or not os.path.exists(self.bedmaster_index):
                raise ValueError(
                    "No method to get xref table.  Specify a valid path to an existing "
                    "xref table or a bedmaster index table.",
                )
            bedmaster_matcher = PatientBedmasterMatcher(
                bedmaster=self.bedmaster_dir,
                adt=self.adt,
            )
            bedmaster_matcher.match_files(
                bedmaster_index=self.bedmaster_index,
                xref=self.xref_file,
            )

        adt_df = pd.read_csv(self.adt)
        adt_columns = EDW_FILES["adt_file"]["columns"]
        adt_df = adt_df[adt_columns].drop_duplicates()

        xref = pd.read_csv(self.xref_file)
        xref = xref.drop_duplicates(subset=["MRN", "PatientEncounterID", "Path"])
        xref["MRN"] = xref["MRN"].astype(str)

        edw_mrns = [
            folder
            for folder in os.listdir(self.edw_dir)
            if os.path.isdir(os.path.join(self.edw_dir, folder))
        ]

        if mrns:
            xref = xref[xref["MRN"].isin(mrns)]
            edw_mrns = [ele for ele in edw_mrns if ele in mrns]
        if starting_time:
            xref = xref[xref["TransferInDTS"] > starting_time]
            adt_df = adt_df[adt_df[adt_columns[0]].isin(edw_mrns)]
            adt_df[adt_columns[4]] = get_unix_timestamps(adt_df[adt_columns[4]].values)
            adt_df = adt_df[adt_df[adt_columns[4]] > starting_time]
            edw_mrns = list(adt_df[adt_columns[0]].drop_duplicates().astype(str))
        if ending_time:
            xref = xref[xref["TransferOutDTS"] < ending_time]
            adt_df = adt_df[adt_df[adt_columns[0]].isin(edw_mrns)]
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
        if allow_one_source:
            self.add_edw_elements(edw_mrns)
        self.assess_coverage()
        self.stats()

        # Get only the first n patients
        if (n_patients or 0) > len(self.crossref):
            logging.warning(
                f"Number of patients set to tensorize "
                f"exceeds the amount of patients stored. "
                f"Number of patients to tensorize will be changed to "
                f"{len(self.crossref)}.",
            )
            return self.crossref
        return dict(list(self.crossref.items())[:n_patients])

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
            bedmaster_path = os.path.join(self.bedmaster_dir, row["Path"])
            xfer_in, xfer_out = row["TransferInDTS"], row["TransferOutDTS"]
            self.crossref[mrn][csn][(xfer_in, xfer_out)].append(bedmaster_path)

        for _mrn, visits in self.crossref.items():
            for _csn, beds in visits.items():
                for _, bedmaster_files in beds.items():
                    bedmaster_files.sort(
                        key=lambda x: (
                            int(re.split("[_-]", x)[-3]),
                            int(x.split("_")[-2]),
                        ),
                    )

    def add_edw_elements(self, edw_mrns):
        # Add elements from EDW folders
        for mrn in edw_mrns:
            csns = [
                csn
                for csn in os.listdir(os.path.join(self.edw_dir, mrn))
                if os.path.isdir(os.path.join(self.edw_dir, mrn, csn))
            ]
            for csn in csns:
                _ = self.crossref[mrn][csn]  # this populates the defaultdict

    def assess_coverage(self):
        for mrn in self.crossref:
            for csn in self.crossref[mrn]:
                # Check if there exist Bedmaster data for this mrn-csn pair
                if not self.crossref[mrn][csn]:
                    logging.warning(f"No Bedmaster data for MRN: {mrn}, CSN: {csn}.")
                # Check if there exist EDW data for this mrn-csn pair
                edw_file_path = os.path.join(self.edw_dir, mrn, csn)
                if (
                    not os.path.isdir(edw_file_path)
                    or len(os.listdir(edw_file_path)) == 0
                ):
                    logging.warning(f"No EDW data for MRN: {mrn}, CSN: {csn}.")

    def stats(self):
        edw_mrns_set = {
            mrn
            for mrn in os.listdir(self.edw_dir)
            if os.path.isdir(os.path.join(self.edw_dir, mrn))
        }
        edw_csns_set = {
            csn
            for mrn in edw_mrns_set
            for csn in os.listdir(os.path.join(self.edw_dir, mrn))
            if os.path.isdir(os.path.join(self.edw_dir, mrn, csn))
        }

        xref = pd.read_csv(self.xref_file, dtype=str)
        xref_mrns_set = set(xref["MRN"].unique())
        try:
            _xref_csns_set = np.array(
                list(xref["PatientEncounterID"].unique()),
                dtype=float,
            )
            xref_csns_set = set(_xref_csns_set.astype(int).astype(str))
        except ValueError:
            xref_csns_set = set(xref["PatientEncounterID"].unique())
        xref_bedmaster_files_set = set(xref["Path"].unique())

        crossref_mrns_set = set(self.crossref.keys())
        crossref_csns_set = set()
        crossref_bedmaster_files_set = set()
        for _, visits in self.crossref.items():
            for visit_id, xfers in visits.items():
                crossref_csns_set.add(visit_id)
                for _, bedmaster_files in xfers.items():
                    for bedmaster_file in bedmaster_files:
                        crossref_bedmaster_files_set.add(bedmaster_file)
        logging.info(
            f"MRNs in {self.edw_dir}: {len(edw_mrns_set)}\n"
            f"MRNs in {self.xref_file}: {len(xref_mrns_set)}\n"
            f"Union MRNs: {len(edw_mrns_set.intersection(xref_mrns_set))}\n"
            f"Intersect MRNs: {len(crossref_mrns_set)}\n"
            f"CSNs in {self.edw_dir}: {len(edw_csns_set)}\n"
            f"CSNs in {self.xref_file}: {len(xref_csns_set)}\n"
            f"Union CSNs: {len(edw_csns_set.intersection(xref_csns_set))}\n"
            f"Intersect CSNs: {len(crossref_csns_set)}\n"
            f"Bedmaster files IDs in {self.xref_file}: "
            f"{len(xref_bedmaster_files_set)}\n"
            f"Intersect Bedmaster files: {len(crossref_bedmaster_files_set)}\n",
        )
