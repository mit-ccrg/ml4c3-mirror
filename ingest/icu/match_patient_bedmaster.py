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
    def __init__(
        self,
        path_bedmaster: str,
        path_adt: str,
        desired_departments: List[str] = None,
    ):
        """
        Init Patient Bedmaster matcher.

        :param path_bedmaster: <str> Directory containing all the Bedmaster files.
        :param path_adt: <str> Path to ADT table.
        :param desired_departments: <List[str]> List of all desired departments.
        """
        self.path_bedmaster = path_bedmaster
        self.path_adt = path_adt
        self.desired_departments = desired_departments
        self.folder_to_dept: Dict[str, List[str]] = {}
        for key, values in MAPPING_DEPARTMENTS.items():
            for value in values:
                if value not in self.folder_to_dept:
                    self.folder_to_dept[value] = [key]
                else:
                    self.folder_to_dept[value].append(key)

    @staticmethod
    def get_department_short_names(departments):
        departments_new = []
        for d in departments:
            if d in MAPPING_DEPARTMENTS:
                keys = MAPPING_DEPARTMENTS[d]
                for k in keys:
                    if k is not None:
                        departments_new.append(k)
        return set(departments_new)

    @staticmethod
    def parse_bedmaster_file_paths(
        path_bedmaster: str,
        departments_short_names: set = None,
    ):
        """
        Given a path to a directory, recursively iterate over all Bedmaster files
        and retrieve metadata from the paths, inlcuding department, room,
        and start time. Optionally, focus the search on subset of department folders.

        :param path_bedmaster: <str> path to directory with subdirectories organized
               by department; subdirectories contain Bedmaster .mat files.
        :param departments_short_names: <set> Names of departments in short format;
               these are hard to remember and are thus saved in a dictionary keyed
               by human-readable department names; this dict is in definitions/icu.
        :return: <pd.DataFrame> Table of full paths to Bedmaster files, departments,
                 room_beds, and start time of each file.
        """
        logging.info(f"Getting paths to Bedmaster files at {path_bedmaster}")
        if departments_short_names is not None:
            logging.info(
                f"Restricting search to the following departments: {departments_short_names}",
            )
        bedmaster_files, _ = get_files_in_directory(
            directory=path_bedmaster,
            file_extension=BEDMASTER_EXT,
            departments_short_names=departments_short_names,
        )

        departments = []
        room_beds = []
        start_times = []
        num_files = len(bedmaster_files)
        fraction = round(num_files / 20) - 1

        for i, bedmaster_file in enumerate(bedmaster_files):
            if i % fraction == 0:
                percent = i / num_files * 100
                logging.info(
                    f"Parsed metadata from {i} / {num_files} ({percent:.0f}%) "
                    "Bedmaster file paths",
                )

            # Split Bedmaster file name into fields:
            # <department>_<room_bed>-<start_t>-#_v4.mat is mapped to
            # department, room_bed, start_t
            file_name = os.path.split(bedmaster_file)[1]
            split_info = re.split("[^a-zA-Z0-9]", file_name)

            # If department name contains a hyphen, rejoin the name again:
            # ELL-7 --> ELL, 7 --> ELL-7
            if split_info[0].isalpha() and len(split_info[1]) == 1:
                department = split_info[0] + "-" + split_info[1]
                # split_info[1] contains the second part of the department name,
                # e.g. 7 in ELL-7. So we need to delete it so the subsequent indices
                # are consistent with non-hyphenated department names.
                del split_info[1]
            else:
                department = split_info[0]
            departments.append(department)
            k = 2

            # If Bedmaster file name contains a "+" or "~" sign, take next field as it
            # will contain an empty one.
            if split_info[k] == "":
                k = 3

            room_bed = split_info[1]

            if any(s.isalpha() for s in room_bed):
                room_bed = room_bed[:-1] + " " + room_bed[-1]
            else:
                room_bed = room_bed + " A"
            while len(room_bed) < 6:
                room_bed = "0" + room_bed

            # Prepend room bed identifier with first character of department
            room_bed_nm = department[0] + room_bed
            room_beds.append(room_bed_nm)

            start_times.append(split_info[k])

        metadata = pd.DataFrame(
            {
                "path": bedmaster_files,
                "department": departments,
                "room_bed": room_beds,
                "start_time": start_times,
            },
        )
        return metadata

    def match_files(
        self,
        path_xref: str = None,
    ):
        """
        Match Bedmaster files with patient's MRN and EncounterID.

        :param path_xref: <str> Path to the CSV file containing the xref table.
        """
        logging.info("Matching Bedmaster files with MRNs and CSNs.")

        # Read ADT table and take desired columns, remove duplicates, and sort
        adt = pd.read_csv(self.path_adt)
        adt = (
            adt[EDW_FILES["adt_file"]["columns"]]
            .drop_duplicates()
            .sort_values("TransferInDTS")
        )

        # Convert dates to Unix Time Stamps
        adt["TransferInDTS"] = get_unix_timestamps(adt["TransferInDTS"].values)
        adt["TransferOutDTS"] = get_unix_timestamps(adt["TransferOutDTS"].values)

        # Get departments from ADT:
        departments = set(adt["DepartmentDSC"])
        departments_short_names = self.get_department_short_names(
            departments=departments,
        )
        if self.desired_departments is not None:
            departments_short_names = self.get_department_short_names(
                departments=self.desired_departments,
            )

        # Parse Bedmaster file paths into metadata
        metadata = self.parse_bedmaster_file_paths(
            path_bedmaster=self.path_bedmaster,
            departments_short_names=departments_short_names,
        )

        # Now, we have:
        # adt: patient MRNs, CSNs, and stay information
        # metadata: Bedmaster file info
        logging.info(f"Cross referencing metadata and ADT DataFrames")

        # Merge adt and metadata on the room_bed
        xref = adt.merge(metadata, left_on="BedLabelNM", right_on="room_bed")

        # Keep rows with ADT start time within Bedmaster file transfer times
        xref = xref[
            (xref["TransferInDTS"] - 3600 <= xref["start_time"].astype(float))
            & (xref["start_time"].astype(float) < xref["TransferOutDTS"])
        ]

        # Save xref table to CSV
        xref.to_csv(path_xref, index=False)
        logging.info(f"Saved {path_xref}")


def match_data(args):
    bedmaster_matcher = PatientBedmasterMatcher(
        path_bedmaster=args.path_bedmaster,
        path_adt=args.path_adt,
        desired_departments=args.departments,
    )
    bedmaster_matcher.match_files(path_xref=args.path_xref)
