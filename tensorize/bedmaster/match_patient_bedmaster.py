# Imports: standard library
import logging
from typing import Dict, List

# Imports: third party
import pandas as pd

# Imports: first party
from ml4c3.utils import get_unix_timestamps
from definitions.edw import EDW_FILES
from definitions.icu import MAPPING_DEPARTMENTS

# pylint: disable=too-many-branches, line-too-long


class PatientBedmasterMatcher:
    """
    Match Bedmaster files with patient's MRN and EncounterID using ADT table.
    """

    def __init__(
        self,
        bedmaster: str,
        adt: str,
        desired_departments: List[str] = None,
    ):
        """
        Init Patient Bedmaster Matcher.

        :param bedmaster: <str> Directory containing all the Bedmaster files.
        :param adt: <str> Path to ADT table.
        :param desired_departments: <List[str]> List of all desired departments.
        """
        self.bedmaster = bedmaster
        self.adt = adt
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

    def match_files(
        self,
        bedmaster_index: str,
        xref: str = None,
    ):
        """
        Match Bedmaster files with patient's MRN and EncounterID.

        :param bedmaster_index: <str> Path to the CSV file containing all the bedmaster
               files information.
        :param xref: <str> Path to the CSV file containing the xref table.
        """
        logging.info("Matching Bedmaster files with MRNs and CSNs.")

        # Read ADT table and take desired columns, remove duplicates, and sort
        adt_df = pd.read_csv(self.adt)
        adt_df = (
            adt_df[EDW_FILES["adt_file"]["columns"]]
            .drop_duplicates()
            .sort_values("TransferInDTS")
        )

        # Convert dates to Unix Time Stamps
        adt_df["TransferInDTS"] = get_unix_timestamps(adt_df["TransferInDTS"].values)
        adt_df["TransferOutDTS"] = get_unix_timestamps(adt_df["TransferOutDTS"].values)

        # Get departments from ADT:
        departments = set(adt_df["DepartmentDSC"])
        departments_short_names = self.get_department_short_names(
            departments=departments,
        )
        if self.desired_departments is not None:
            departments_short_names = self.get_department_short_names(
                departments=self.desired_departments,
            )

        # Parse Bedmaster file paths into metadata
        metadata = pd.read_csv(bedmaster_index)
        metadata = metadata[metadata["DepartmentDSC"].isin(departments_short_names)]
        metadata = metadata.drop(["DepartmentDSC"], axis=1)

        # Now, we have:
        # adt: patient MRNs, CSNs, and stay information
        # metadata: Bedmaster file info
        logging.info("Cross referencing metadata and ADT DataFrames")

        # Merge adt and metadata on the room_bed
        xref_df = adt_df.merge(metadata, how="left", on="BedLabelNM")

        # Keep rows with overlap between ADT and Bedmaster file transfer times
        xref_df = xref_df[
            (xref_df["StartTime"].astype(float) <= xref_df["TransferOutDTS"])
            & (xref_df["EndTime"].astype(float) >= xref_df["TransferInDTS"])
        ]
        # Define type of overlap
        xref_df["OverlapID"] = (xref_df["StartTime"] < xref_df["TransferInDTS"]) * 1
        xref_df["OverlapID"] = (
            xref_df["OverlapID"] + (xref_df["EndTime"] > xref_df["TransferOutDTS"]) * 2
        )
        xref_df.loc[
            xref_df["OverlapID"] == 0,
            "OverlapDSC",
        ] = ".mat file is completely within patient's stay"
        xref_df.loc[
            xref_df["OverlapID"] == 1,
            "OverlapDSC",
        ] = ".mat file overhangs patient's stay on the left"
        xref_df.loc[
            xref_df["OverlapID"] == 2,
            "OverlapDSC",
        ] = ".mat file overhangs patient's stay on the right"
        xref_df.loc[
            xref_df["OverlapID"] == 3,
            "OverlapDSC",
        ] = "patient's stay is completely within the .mat file"

        # Keep just useful columns
        xref_df = xref_df[
            [
                "MRN",
                "PatientEncounterID",
                "Path",
                "TransferInDTS",
                "TransferOutDTS",
                "DepartmentID",
                "DepartmentDSC",
                "StartTime",
                "EndTime",
                "OverlapID",
                "OverlapDSC",
            ]
        ]

        # Save xref table to CSV
        xref_df.to_csv(path_or_buf=xref, index=False)
        logging.info(f"Saved {xref}")


def match_data(args):
    bedmaster_matcher = PatientBedmasterMatcher(
        bedmaster=args.bedmaster,
        adt=args.adt,
        desired_departments=args.departments,
    )
    bedmaster_matcher.match_files(bedmaster_index=args.bedmaster_index, xref=args.xref)
