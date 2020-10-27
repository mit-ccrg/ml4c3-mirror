# Imports: standard library
import os
import logging
from typing import Dict, List

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from ml4c3.utils import get_unix_timestamps
from ml4c3.definitions.icu import EDW_FILES
from ml4c3.ingest.icu.matchers import PatientBMMatcher


class CrossReferencer:
    """
    Class that cross-references Bedmaster and EDW data.

    Used to ensure correspondence between the data.
    """

    def __init__(
        self,
        bm_dir: str,
        edw_dir: str,
        xref_file: str,
        adt_file: str = EDW_FILES["adt_file"]["name"],
    ):
        self.bm_dir = bm_dir
        self.edw_dir = edw_dir
        self.xref_file = xref_file
        if not adt_file.endswith(".csv"):
            adt_file += ".csv"
        self.adt_file = os.path.join(self.edw_dir, adt_file)
        self.crossref: Dict[str, Dict[str, List[str]]] = {}

    def get_xref_files(
        self,
        mrns: List[str] = None,
        starting_time: int = None,
        ending_time: int = None,
        overwrite_hd5: bool = True,
        n_patients: int = None,
        tensors: str = None,
        flag_one_source: bool = False,
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Get the cross-referenced Bedmaster files and EDW files.

        The output dictionary will have the format:

        {"MRN1": {"visitID":[bm_files],
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
        :param flag_one_source: <bool> bool indicating whether a patient with
                                just one type of data will be tensorized or not.
        :return: <dict> a dictionary with the MRNs, visit ID and BM files.
        """
        self.crossref = {}
        if not os.path.exists(self.xref_file):
            bm_matcher = PatientBMMatcher(False, self.bm_dir, self.edw_dir)
            bm_matcher.match_files(self.xref_file)

        adt = pd.read_csv(self.adt_file)
        adt_columns = EDW_FILES["adt_file"]["columns"]
        adt = adt[adt_columns].drop_duplicates()

        xref = pd.read_csv(self.xref_file)
        xref = xref.drop_duplicates(subset=["MRN", "PatientEncounterID", "fileID"])
        edw_mrns = [
            folder
            for folder in os.listdir(self.edw_dir)
            if os.path.isdir(os.path.join(self.edw_dir, folder))
        ]

        if mrns:
            xref = xref[xref["MRN"].isin(mrns)]
            edw_mrns = [ele for ele in edw_mrns if ele in mrns]
        if starting_time:
            xref = xref[xref["unixFileEndTime"] > starting_time]
            adt = adt[adt[adt_columns[0]].isin(edw_mrns)]
            adt[adt_columns[4]] = get_unix_timestamps(adt[adt_columns[4]].values)
            adt = adt[adt[adt_columns[4]] > starting_time]
            edw_mrns = list(adt[adt_columns[0]].drop_duplicates().astype(str))
        if ending_time:
            xref = xref[xref["unixFileStartTime"] < ending_time]
            adt = adt[adt[adt_columns[0]].isin(edw_mrns)]
            adt[adt_columns[3]] = get_unix_timestamps(adt[adt_columns[3]].values)
            adt = adt[adt[adt_columns[3]] < ending_time]
            edw_mrns = list(adt[adt_columns[0]].drop_duplicates().astype(str))
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

        self.add_bm_elements(xref, edw_mrns, flag_one_source)
        if flag_one_source:
            self.add_edw_elements(edw_mrns)
        self.assess_coverage()
        self.stats()

        # Get only the first n patients
        if (n_patients or 0) > len(self.crossref):
            logging.warning(
                f"Number of patients set in --num_patients_to_tensorize "
                f"exceeds the amount of patients stored. "
                f"Number of patients to tensorize will be changed to "
                f"{len(self.crossref)}.",
            )
            return self.crossref

        return dict(list(self.crossref.items())[:n_patients])

    def add_bm_elements(self, xref, edw_mrns, flag_one_source):
        # Add elements from cross referencer .csv
        for _, row in xref.iterrows():
            mrn = str(row.MRN)
            if flag_one_source and mrn not in edw_mrns:
                continue
            try:
                csn = str(int(row.PatientEncounterID))
            except ValueError:
                csn = str(row.PatientEncounterID)
            bm_file_id = str(row.fileID)
            bm_path = [
                os.path.join(self.bm_dir, file)
                for file in os.listdir(self.bm_dir)
                if file.startswith(bm_file_id)
            ]
            if mrn not in self.crossref:
                self.crossref[mrn] = {csn: bm_path}
            elif csn not in self.crossref[mrn]:
                self.crossref[mrn][csn] = bm_path
            else:
                self.crossref[mrn][csn].extend(bm_path)
        for _mrn, visits in self.crossref.items():
            for _csn, bm_files in visits.items():
                bm_files.sort(key=lambda x: int(x.split("_")[-2]))

    def add_edw_elements(self, edw_mrns):
        # Add elements from EDW folders
        for mrn in edw_mrns:
            csns = [
                csn
                for csn in os.listdir(os.path.join(self.edw_dir, mrn))
                if os.path.isdir(os.path.join(self.edw_dir, mrn, csn))
            ]
            if mrn not in self.crossref:
                self.crossref[mrn] = {}
                for csn in csns:
                    self.crossref[mrn][csn] = []
            else:
                for csn in csns:
                    if csn not in self.crossref[mrn]:
                        self.crossref[mrn][csn] = []

    def assess_coverage(self):
        for mrn in self.crossref:
            for csn in self.crossref[mrn]:
                # Check if there exist BM data for this mrn-csn pair
                if not self.crossref[mrn][csn]:
                    logging.warning(f"No BM data for MRN: {mrn}, CSN: {csn}.")
                # Check if there exist EDW data for this mrn-csn pair
                edw_file_path = os.path.join(self.edw_dir, mrn, csn)
                if (
                    not os.path.isdir(edw_file_path)
                    or len(os.listdir(edw_file_path)) == 0
                ):
                    logging.warning(f"No EDW data for MRN: {mrn}, CSN: {csn}.")

    def stats(self):
        """
        :param crossref: <dict> a dictionary with the MRNs, visit ID and BM files.
        """
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
        xref_bmf_set = set(xref["fileID"].unique())

        crossref_mrns_set = set(self.crossref.keys())
        crossref_csns_set = set()
        crossref_bmf_set = set()
        for _, visits in self.crossref.items():
            for visit_id, bmfiles in visits.items():
                crossref_csns_set.add(visit_id)
                for bmfile in bmfiles:
                    crossref_bmf_set.add(bmfile)
        logging.info(
            f"MRNs in {self.edw_dir}: {len(edw_mrns_set)}\n"
            f"MRNs in {self.xref_file}: {len(xref_mrns_set)}\n"
            f"Union MRNs: {len(edw_mrns_set.intersection(xref_mrns_set))}\n"
            f"Intersect MRNs: {len(crossref_mrns_set)}\n"
            f"CSNs in {self.edw_dir}: {len(edw_csns_set)}\n"
            f"CSNs in {self.xref_file}: {len(xref_csns_set)}\n"
            f"Union CSNs: {len(edw_csns_set.intersection(xref_csns_set))}\n"
            f"Intersect CSNs: {len(crossref_csns_set)}\n"
            f"Bm files IDs in {self.xref_file}: {len(xref_bmf_set)}\n"
            f"Intersect bm files: {len(crossref_bmf_set)}\n",
        )
