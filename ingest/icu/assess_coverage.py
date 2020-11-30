# Imports: standard library
import os
import re
import logging
from typing import Any, Set, Dict, List

# Imports: third party
import h5py
import numpy as np
import pandas as pd

# Imports: first party
from ml4c3.utils import get_unix_timestamps
from definitions.icu import EDW_FILES, MAPPING_DEPARTMENTS
from ml4c3.tensormap.icu_signals import get_tmap as GET_SIGNAL_TMAP
from ingest.icu.match_patient_bedmaster import PatientBedmasterMatcher

EXPECTED_FILES = []
for file_type in EDW_FILES:
    if file_type != "adt_file":
        EXPECTED_FILES.append(EDW_FILES[file_type]["name"])


class ICUCoverageAssesser:
    """
    Assesses the coverage of Bedmaster files for a given cohort of patients.
    """

    def __init__(
        self,
        output_dir: str,
        cohort_query: str = None,
        cohort_csv: str = None,
        adt_csv: str = None,
    ):
        """
        Init Assesser class.

        :param cohort_query: <str> Name of the query to obtain list of patients
                             and ADT table.
        :param cohort_csv: <str> Full path of the .csv file containing a list
                           of patients. If --cohort_query is set, this parameter
                           will be ignored.
        :param adt_csv: <str> Full path of the ADT table of the list of patients
                              in --cohort_csv. If --cohort_query is set, this
                              parameter will be ignored.
        :param output_dir: <str> Directory where the analysis results are stored.
        """
        self.output_dir = output_dir
        self.adt_path = os.path.join(output_dir, "adt.csv")

        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        if cohort_query:
            # Obtain list of patients and the corresponding ADT table
            os.system(
                f"python3 /home/$USER/repos/edw/icu/pipeline.py \
                    --cohort_query {cohort_query} \
                    --destination {self.output_dir} \
                    --compute_adt",
            )
            self.cohort_path = os.path.join(self.output_dir, f"{cohort_query}.csv")
        elif cohort_csv and adt_csv:
            self.cohort_path = cohort_csv
            adt = pd.read_csv(adt_csv)
            adt.to_csv(os.path.join(self.output_dir, "adt.csv"))
        elif cohort_csv or adt_csv:
            raise Exception("If --cohort_csv is set, --adt_csv has to be set too.")
        else:
            raise Exception(
                "Or --cohort_query or --cohort_csv and --adt_csv has to be set to "
                "perform the assessment.",
            )

    @staticmethod
    def _list_existing_mrns_csns(folder: str = "./", remove: bool = False):
        """
        List all the MRNs and CSNs in folder with the expected files.

        :param folder: <str> Directory where the analysis is performed.
        :param remove: <bool> Bool indicating if the unfinished folders are
                       deleted or not.
        """
        mrns = os.listdir(folder)
        existing: Dict[str, Set] = {"MRN": set(), "CSN": set()}
        for mrn in mrns:
            if not os.path.isdir(os.path.join(folder, mrn)):
                continue
            csns = os.listdir(os.path.join(folder, mrn))
            for csn in csns:
                if not os.path.isdir(os.path.join(folder, mrn, csn)):
                    continue
                files = os.listdir(os.path.join(folder, mrn, csn))
                if sorted(files) != sorted(EXPECTED_FILES):
                    if remove:
                        for file_name in files:
                            os.remove(os.path.join(folder, mrn, csn, file_name))
                        os.rmdir(os.path.join(folder, mrn, csn))
                        print(f"Removed unfinished CSN: {mrn}/{csn}")
                        if len(os.listdir(os.path.join(folder, mrn))) == 0:
                            os.rmdir(mrn)
                            print(f"Removed unfinished MRN: {mrn}")
                else:
                    existing["MRN"].add(int(mrn))
                    existing["CSN"].add(int(csn))
        return existing

    def _compare(
        self,
        file_path: str = "./adt.csv",
        folder: str = "./",
        remove: bool = False,
    ):
        """
        Creates a .csv file with the subset of MRNs and CSNs on  file_path that
        doesn't exist in folder.

        :param file_path: <str> File with a list of MRNs and CSNs to check their
                          existance.
        :param folder: <str> Directory where the analysis is performed.
        :param remove: <bool> Bool indicating if the unfinished MRNs and CSNs in
                       folder are deleted or not.
        """
        existing = self._list_existing_mrns_csns(folder, remove)
        patients_list = pd.read_csv(file_path)[
            ["MRN", "PatientEncounterID"]
        ].drop_duplicates()
        remaining_patients = patients_list[
            ~patients_list["PatientEncounterID"].isin(existing["CSN"])
        ]
        return remaining_patients

    @staticmethod
    def _filter_xref(cohort: pd.DataFrame, xref: pd.DataFrame, time_column: str):
        """
        Filter xref table by time_column and PatientEncoutnerID from cohort.

        :param cohort: <pd.DataFrame> Pandas data frame with a list of patients
                       with the associated event and the time and where it happened.
        :param xref: <pd.DataFrame> Pandas data frame with the cross-reference table.
                     Pairs the CSNs with the Bedmaster files.
        :param time_column: <str> Column in cohort df that defines the time of
                            the event.
        """
        count = 0
        new_xref = pd.DataFrame()
        for _, row in cohort.iterrows():
            unix_time = get_unix_timestamps(np.array(row[time_column]))
            xref_f = xref[xref["unixFileStartTime"] <= unix_time]
            xref_f = xref_f[xref_f["unixFileEndTime"] >= unix_time]
            xref_f = xref_f[xref_f["PatientEncounterID"] == row["PatientEncounterID"]]
            xref_f = xref_f[xref_f["Department"] == row["DepartmentDSC"]]
            new_xref = pd.concat([new_xref, xref_f], ignore_index=True)
            if len(xref_f) >= 1:
                count += 1
        return (
            xref[
                xref["PatientEncounterID"].isin(new_xref["PatientEncounterID"].unique())
            ],
            count,
        )

    @staticmethod
    def _coverage(csv: pd.DataFrame, data: Dict[str, Any], count: int = None):
        """
        Update the dictionary with the coverage assessment results with the
        values given in csv and count.

        :param csv: <pd.DataFrame> Pandas data frame to analyze the coverage.
        :param data: <Dict[str, Any]> Dictionary with the coverage assessment.
        :param count: <int> Event counts (if any) to be added in data.
        """
        if "fileID" in csv.columns:
            data["Bedmaster files [u]"].append(len(csv["fileID"].unique()))
            csv = csv.drop_duplicates(
                subset=["PatientEncounterID", "unixFileStartTime", "unixFileEndTime"],
            )
        else:
            data["Bedmaster files [u]"].append(np.nan)
        if count:
            data["Count [u]"].append(count)
            data["Count [%]"].append(data["Count [u]"][-1] / data["Count [u]"][0] * 100)
        data["Unique MRNs [u]"].append(len(csv["MRN"].unique()))
        data["Unique MRNs [%]"].append(
            data["Unique MRNs [u]"][-1] / data["Unique MRNs [u]"][0] * 100,
        )
        data["Unique CSNs [u]"].append(len(csv["PatientEncounterID"].unique()))
        data["Unique CSNs [%]"].append(
            data["Unique CSNs [u]"][-1] / data["Unique CSNs [u]"][0] * 100,
        )
        return data

    @staticmethod
    def _hd5_coverage(
        csv: pd.DataFrame,
        path_hd5: str,
        data: Dict[str, Any],
        count: bool = False,
    ):
        """
        Assesses the coverage of hd5 files and update accordingly the
        dictionary with the coverage assessment results.

        :param csv: <pd.DataFrame> Pandas data frame to analyze the coverage.
        :param path_hd5: <str> Directory with .hd5 files.
        :param data: <Dict[str, Any]> Dictionary with the coverage assessment.
        :param count: <bool> Bool indicating whether or not the number of events
                      is calculated.
        """
        hd5_files = [
            hd5_file for hd5_file in os.listdir(path_hd5) if hd5_file.endswith(".hd5")
        ]
        new_csv = pd.DataFrame()
        for _, row in csv.iterrows():
            if f"{int(row['MRN'])}.hd5" in hd5_files:
                hd5_file = h5py.File(os.path.join(path_hd5, f"{row['MRN']}.hd5"), "r")
                csns = GET_SIGNAL_TMAP("visits").tensor_from_file(
                    GET_SIGNAL_TMAP("visits"),
                    hd5_file,
                )
                if str(int(row["PatientEncounterID"])) in csns:
                    new_csv = pd.concat(
                        [new_csv, row.to_frame().transpose()],
                        ignore_index=True,
                    )

        if len(new_csv) == 0:
            data["Bedmaster files [u]"].append(np.nan)
            if count:
                data["Count [u]"].append(0)
                data["Count [%]"].append(np.nan)
            data["Unique MRNs [u]"].append(0)
            data["Unique MRNs [%]"].append(np.nan)
            data["Unique CSNs [u]"].append(0)
            data["Unique CSNs [%]"].append(np.nan)
            return data

        data["Bedmaster files [u]"].append(np.nan)
        if count:
            data["Count [u]"].append(len(new_csv))
            data["Count [%]"].append(data["Count [u]"][-1] / data["Count [u]"][0] * 100)
        data["Unique MRNs [u]"].append(len(new_csv["MRN"].unique()))
        data["Unique MRNs [%]"].append(
            data["Unique MRNs [u]"][-1] / data["Unique MRNs [u]"][0] * 100,
        )
        data["Unique CSNs [u]"].append(len(new_csv["PatientEncounterID"].unique()))
        data["Unique CSNs [%]"].append(
            data["Unique CSNs [u]"][-1] / data["Unique CSNs [u]"][0] * 100,
        )
        return data

    def assess_coverage(
        self,
        path_bedmaster: str,
        path_edw: str,
        path_hd5: str,
        desired_departments: List[str] = None,
        event_column: str = None,
        time_column: str = None,
        count: bool = False,
    ):
        """
        Main function of the class. Creates one .csv file comparing the MRNs
        and CSNs available in EDW, Bedmaster and HD5 and a second .csv file
        listing the MRNs and CSNs from EDW that haven't been downloaded yet.

        :param path_bedmaster: <str> Directory with Bedmaster .mat files.
        :param path_edw: <str> Directory with EDW .csv files.
        :param path_hd5: <str> Directory with .hd5 files.
        :param desired_departments: <List[str]> List of department names.
        :param event_column: <str> Name of the event column (if exists) in
                             --cohort_query/--cohort_csv.
        :param time_column: <str> Name of the event time column (if exists) in
                             --cohort_query/--cohort_csv.
        :param count: <bool> Count the number of unique rows (events) in
                             --cohort_query/--cohort_csv.
        """

        departments: Dict[Any, str] = {
            None: "all_departments",
        }

        if desired_departments:
            for department in desired_departments:
                dept_name = department.upper()
                pattern = re.compile("([a-zA-Z]+)([0-9]+)")
                dept_name = " ".join(pattern.findall(dept_name)[0]).upper()
                for key in MAPPING_DEPARTMENTS:
                    if dept_name in key or dept_name.replace(" ", "") in key:
                        departments[key] = MAPPING_DEPARTMENTS[key][0].lower()
                        break
                else:
                    logging.warning(
                        "Unable to find {department} in MAPPING_DEPARTMENTS.",
                    )

        columns = []
        for key in departments:
            columns.append(departments[key])
            columns.append(f"{departments[key]} with Bedmaster data")
            columns.append(f"{departments[key]} with HD5 file")

        # Execute matching algorithm
        matching_depts = []
        for key in departments:
            if key:
                matching_depts.append(key)
        bedmaster_matcher = PatientBedmasterMatcher(
            path_bedmaster=path_bedmaster,
            path_adt=os.path.join(self.output_dir, "adt.csv"),
            desired_departments=matching_depts,
        )
        bedmaster_matcher.match_files(os.path.join(self.output_dir, "xref.csv"))

        # Initialize empty table
        data: Dict[str, Any] = {}
        if count:
            data.update({"Count [u]": [], "Count [%]": []})
        data.update(
            {
                "Unique MRNs [u]": [],
                "Unique MRNs [%]": [],
                "Unique CSNs [u]": [],
                "Unique CSNs [%]": [],
                "Bedmaster files [u]": [],
            },
        )

        # Iterate through departments
        for department in departments:
            cohort = pd.read_csv(self.cohort_path)
            xref = pd.read_csv(os.path.join(self.output_dir, "xref.csv"))
            xref = xref.dropna(subset=["MRN", "PatientEncounterID"])

            if time_column and event_column:
                cohort = cohort.drop_duplicates(
                    subset=["MRN", "PatientEncounterID", event_column, time_column],
                )
            else:
                cohort = cohort.drop_duplicates(subset=["MRN", "PatientEncounterID"])

            if department:
                cohort = cohort[cohort["DepartmentDSC"] == department]
                xref = xref[xref["Department"] == department]
                xref = xref[
                    xref["PatientEncounterID"].isin(
                        cohort["PatientEncounterID"].unique(),
                    )
                ]

            if time_column:
                xref, counter = self._filter_xref(cohort, xref, time_column)
            else:
                counter = len(xref)

            if count:
                data = self._coverage(cohort, data, len(cohort))
                data = self._coverage(xref, data, counter)
            else:
                data = self._coverage(cohort, data)
                data = self._coverage(xref, data)
            data = self._hd5_coverage(cohort, path_hd5, data, count)
        data_frame = pd.DataFrame.from_dict(
            data,
            orient="index",
            columns=columns,
        ).round(2)
        csv_path = os.path.join(self.output_dir, "coverage.csv")
        data_frame.to_csv(csv_path)

        remaining_patients = self._compare(self.adt_path, path_edw, False)
        remaining_patients.to_csv(
            os.path.join(self.output_dir, "remaining_patients.csv"),
            index=False,
        )


def assess_coverage(args):
    assesser = ICUCoverageAssesser(
        output_dir=args.output_folder,
        cohort_query=args.cohort_query,
        cohort_csv=args.cohort_csv,
        adt_csv=args.path_adt,
    )
    assesser.assess_coverage(
        path_bedmaster=args.path_bedmaster,
        path_edw=args.path_edw,
        path_hd5=args.tensors,
        desired_departments=args.departments,
        event_column=args.event_column,
        time_column=args.time_column,
        count=args.count,
    )
