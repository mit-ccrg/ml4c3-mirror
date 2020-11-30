# Imports: standard library
import os
import re
import logging
from typing import Any, Dict, List, Tuple

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from ml4c3.utils import get_unix_timestamps
from definitions.icu import EDW_FILES, BEDMASTER_EXT, MAPPING_DEPARTMENTS
from ingest.icu.utils import get_files_in_directory

# pylint: disable=too-many-branches, line-too-long


class PatientBedmasterMatcher:
    """
    Implementation of Patient Bedmaster matching algorithm.
    """

    def __init__(
        self,
        path_bedmaster: str,
        path_adt: str,
        desired_departments: List[str] = None,
    ):
        """
        Init Patient Bedmaster matcher.

        :param path_bedmaster: <str> Directory containing all the Bedmaster .mat data.
        :param path_adt: <str> Path to ADT table.
        :param desired_departments: <List[str]> List of all desired departments.
        """
        self.path_bedmaster = path_bedmaster
        self.path_adt = path_adt
        self.desired_departments = desired_departments

        self.dept_to_folder = MAPPING_DEPARTMENTS
        self.folder_to_dept = {}
        for key, values in MAPPING_DEPARTMENTS.items():
            for value in values:
                self.folder_to_dept.update({value: key})

        self.table_dic: Dict[str, List[Any]] = self._new_table_dic()

    @staticmethod
    def _new_table_dic():
        """
        Clear "self.table_dic" when initializing class and running match_files
        method.
        """
        return {
            "MRN": [],
            "PatientEncounterID": [],
            "transferIn": [],
            "transferOut": [],
            "fileID": [],
            "unixFileStartTime": [],
            "unixFileEndTime": [],
            "Department": [],
        }

    @staticmethod
    def _take_bedmaster_file_info(bedmaster_file: str) -> Tuple[str, ...]:
        """
        Take information from Bedmaster files name.

        :param bedmaster_file: <str> bedmaster file name.
        :return: <Tuple[str]> array with all the info from the name. Element 0
                              is department, element 1 is room, element 2 is
                              starting time.
        """
        # Split Bedmaster file name into fields:
        # <department>_<roomBed>-<start_t>-#_v4.mat --> department, roomBed, start_time.
        split_info = re.split("[^a-zA-Z0-9]", bedmaster_file)
        # If department name contains a hyphen, rejoin the name again:
        # ELL-7 --> ELL, 7 --> ELL-7
        if split_info[0].isalpha() and len(split_info[1]) == 1:
            split_info[0] = split_info[0] + "-" + split_info[1]
            del split_info[1]
        k = 2
        # If Bedmaster file name contains a "+" or "~" sign, take next field as it
        # will contain an empty one.
        if split_info[k] == "":
            k = 3
        return split_info[0], split_info[1], split_info[k]

    def match_files(self, path_xref: str = None, store_unmatched: bool = False):
        """
        Match Bedmaster files with patient's MRN and EncounterID.

        :param path_xref: <str> directory of the output xref table.
        :param store_unmatched: <bool> Bool indicating if the unmatched Bedmaster files
                                are saved in the xref table.
        """
        logging.info("Cross referencing ADT table with Bedmaster files.")
        # Clear dictionary if it has any information
        if any(self.table_dic[key] for key in self.table_dic.keys()):
            self.table_dic = self._new_table_dic()

        # Read ADT table and take desired columns, remove duplicates and sort
        adt_df = pd.read_csv(self.path_adt)
        adt_df = (
            adt_df[EDW_FILES["adt_file"]["columns"]]
            .sort_values("TransferInDTS")
            .drop_duplicates()
        )
        # Convert dates to Unix Time Stamps
        adt_df["TransferInDTS"] = get_unix_timestamps(adt_df["TransferInDTS"].values)
        adt_df["TransferOutDTS"] = get_unix_timestamps(adt_df["TransferOutDTS"].values)
        # Get departments from ADT:
        adt_departments = np.array(adt_df["DepartmentDSC"].drop_duplicates(), dtype=str)
        # Get names of all the files
        bedmaster_files = []
        undesired_files = []

        # Filter departments to only those that appear
        # in ADT table and those given by user
        departments = set(adt_departments)
        if self.desired_departments is not None:
            departments &= set(self.desired_departments)
        logging.info(f"Checking Bedmaster files from {len(departments)} departments.")

        for dept in sorted(departments):
            if dept not in self.dept_to_folder:
                logging.warning(
                    f"Department {dept} is not found in MAPPING_DEPARTMENTS "
                    "(definitions/icu.py). No matching will be performed with "
                    "patients from this department. Please, add this information "
                    "in MAPPING_DEPARTMENTS (definitions/icu.py).",
                )
                continue
            for subfolder in self.dept_to_folder[dept]:
                bedmaster_files_set, undesired_files_set = get_files_in_directory(
                    directory=os.path.join(self.path_bedmaster, subfolder),
                    extension=BEDMASTER_EXT,
                )
                bedmaster_files.extend(bedmaster_files_set)
                undesired_files.extend(undesired_files_set)
        if len(bedmaster_files) == 0:
            bedmaster_files, undesired_files = get_files_in_directory(
                directory=self.path_bedmaster,
                extension=BEDMASTER_EXT,
            )
        if len(bedmaster_files) == 0:
            raise FileNotFoundError(
                f"No Bedmaster files found in {self.path_bedmaster}.",
            )

        bedmaster_files = sorted(
            [os.path.split(bedmaster_file)[-1] for bedmaster_file in bedmaster_files],
        )
        if undesired_files:
            logging.warning(
                "Not all listed files end with .mat extension. They are "
                "going to be removed from the list.",
            )

        # Iterate over bedmaster_files and match them to the corresponding
        # patient in the ADT table
        unmatched_files = []
        for _, bedmaster_file in enumerate(bedmaster_files):
            dept, room_bed, start_time = self._take_bedmaster_file_info(bedmaster_file)
            if dept not in self.folder_to_dept:
                self.folder_to_dept[dept] = dept
                logging.warning(
                    f"Bedmaster department {dept} doesn't have it's corresponding "
                    "mapping.",
                )

            if (
                self.desired_departments
                and self.folder_to_dept[dept] not in self.desired_departments
            ):
                continue
            # Set room and bed string properly, if it has no room already, add
            # letter A (default in ADT)
            if any(s.isalpha() for s in room_bed):
                room_bed = room_bed[:-1] + " " + room_bed[-1]
            else:
                room_bed = room_bed + " A"

            while len(room_bed) < 6:
                room_bed = "0" + room_bed
            room_bed_nm = dept[0] + room_bed

            # Filter by department and room
            filt_adt_df = adt_df[adt_df["BedLabelNM"] == room_bed_nm]
            # Filter by transfer in and out times (note that one hour
            # margin is considered as Bedmaster files sometimes start a little
            # before than the transfer of the patient)
            filt_adt_df = filt_adt_df[
                (filt_adt_df["TransferInDTS"] - 3600) <= float(start_time)
            ]
            filt_adt_df = filt_adt_df[
                float(start_time) <= filt_adt_df["TransferOutDTS"]
            ]

            if len(filt_adt_df.index) > 0:
                mrn = int(filt_adt_df["MRN"].values[0])
                csn = int(filt_adt_df["PatientEncounterID"].values[0])
                t_start = filt_adt_df["TransferInDTS"].values[0]
                t_end = filt_adt_df["TransferOutDTS"].values[0]

                self.table_dic["MRN"].append(mrn)
                self.table_dic["PatientEncounterID"].append(csn)
                self.table_dic["transferIn"].append(t_start)
                self.table_dic["transferOut"].append(t_end)
                self.table_dic["fileID"].append(bedmaster_file[:-4])
                self.table_dic["unixFileStartTime"].append(start_time)
                self.table_dic["unixFileEndTime"].append(t_end)
                self.table_dic["Department"].append(self.folder_to_dept[dept])
            elif store_unmatched:
                self.table_dic["MRN"].append(None)
                self.table_dic["PatientEncounterID"].append(None)
                self.table_dic["transferIn"].append(None)
                self.table_dic["transferOut"].append(None)
                self.table_dic["fileID"].append(bedmaster_file[:-4])
                self.table_dic["unixFileStartTime"].append(start_time)
                self.table_dic["unixFileEndTime"].append(t_end)
                self.table_dic["Department"].append(self.folder_to_dept[dept])
            else:
                unmatched_files.append(bedmaster_file)

        if len(unmatched_files) > 0:
            logging.warning(
                f"Could not match {len(unmatched_files)} files with patients",
            )
            logging.debug(f"Unmatched files: {unmatched_files}")

        if path_xref:
            xref_table = pd.DataFrame(self.table_dic)
            xref_table.to_csv(path_xref, index=False)


def match_data(args):
    bedmaster_matcher = PatientBedmasterMatcher(
        path_bedmaster=args.path_bedmaster,
        path_adt=args.path_adt,
        desired_departments=args.departments,
    )
    bedmaster_matcher.match_files(path_xref=args.path_xref)
