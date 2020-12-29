# Imports: standard library
import os
import logging
from typing import Set

# Imports: third party
import h5py
import pandas as pd

# Imports: first party
from ml4c3.datasets import patient_csv_to_set
from definitions.icu import (
    EDW_FILES,
    ALARMS_FILES,
    BEDMASTER_EXT,
    MATFILE_EXPECTED_GROUPS,
)
from ingest.icu.utils import get_files_in_directory


class EDWChecker:
    """
    Implementation of Checker for EDW.
    """

    def __init__(self, edw_dir: str):
        """
        Init EDW Checker.

        :param edw_dir: <str> directory containing all the EDW data.
        """
        self.edw_dir = edw_dir

    @staticmethod
    def _check_file_columns(full_file_path, file_expected_columns):
        data_frame = pd.read_csv(full_file_path)
        columns = set(data_frame.columns)
        missing_columns = file_expected_columns.difference(columns)
        if len(missing_columns) > 0:
            logging.error(
                f"Wrong file format: the columns {sorted(missing_columns)} "
                f"were not found in the input file {full_file_path}.",
            )

    def _check_adt(self):
        adt_file = EDW_FILES["adt_file"]["name"]
        adt_file_path = os.path.join(self.edw_dir, adt_file)
        adt_columns = set(EDW_FILES["adt_file"]["columns"])
        other_files_path = [
            os.path.join(self.edw_dir, other_file)
            for other_file in os.listdir(self.edw_dir)
            if not os.path.isdir(os.path.join(self.edw_dir, other_file))
        ]
        # Check if adt table exists.
        if adt_file_path not in other_files_path:
            logging.error(
                f"Wrong folder format: {adt_file} was not found in "
                f"the input directory {self.edw_dir}.",
            )
        # Check adt table content.
        else:
            other_files_path.remove(adt_file_path)
            self._check_file_columns(adt_file_path, adt_columns)
        # Check if there are any unexpected file in edw_dir.
        if len(other_files_path) > 0:
            logging.warning(
                f"Unexpected files: {sorted(other_files_path)}. Just an adt "
                f"table and mrns folders should be stored in {self.edw_dir}.",
            )

    def check_structure(self, patient_csv: str = None):
        """
        Checks if edw_dir is structured properly.

        :param patient_csv: <str> Path to CSV with MRNs to parse; no other MRNs will be parsed.
        """
        self._check_adt()

        expected_columns = {}
        for element in EDW_FILES:
            columns: Set[str] = set()
            for col in EDW_FILES[element]["columns"]:
                columns &= set(col if isinstance(col, list) else [col])
            expected_columns[EDW_FILES[element]["name"]] = columns
        expected_files = set(expected_columns.keys())
        expected_files.remove(EDW_FILES["adt_file"]["name"])

        mrns_folders = [
            os.path.join(self.edw_dir, folder)
            for folder in os.listdir(self.edw_dir)
            if os.path.isdir(os.path.join(self.edw_dir, folder))
        ]
        if patient_csv:
            mrns = patient_csv_to_set(patient_csv)
        for mrn_folder in mrns_folders:
            if patient_csv and mrn_folder not in mrns:
                continue
            csns_folders = [
                os.path.join(mrn_folder, folder)
                for folder in os.listdir(mrn_folder)
                if os.path.isdir(os.path.join(mrn_folder, folder))
            ]
            unexpected_files = [
                os.path.join(mrn_folder, file_name)
                for file_name in os.listdir(mrn_folder)
                if not os.path.isdir(os.path.join(mrn_folder, file_name))
            ]
            # Check that there is at least one folder inside each mrn folder.
            if len(csns_folders) < 1:
                logging.error(
                    f"Wrong folder format: {mrn_folder} doesn't contain any folder.",
                )
            # Check if there are any unexpected files in mrns folders.
            if len(unexpected_files) > 0:
                logging.warning(
                    f"Unexpected files: {sorted(unexpected_files)}. Just "
                    "folders should be stored inside mrns folders.",
                )
            for csn_folder in csns_folders:
                files = set(os.listdir(csn_folder))
                missing_files = expected_files.difference(files)
                unexpected = files.difference(expected_files)
                # Check that inside each csn folder are found all the
                # expected .csv.
                if len(missing_files) > 0:
                    logging.error(
                        "Wrong folder format: the files "
                        f"{sorted(missing_files)} were not found in the "
                        f"input folder {csn_folder}.",
                    )
                # Check that all the expected_files have the expected format.
                for file_name in expected_files.intersection(files):
                    full_file_path = os.path.join(csn_folder, file_name)
                    file_expected_columns = expected_columns[file_name]
                    self._check_file_columns(full_file_path, file_expected_columns)
                # Check if if there are any unexpected file in csns folders.
                if len(unexpected) > 0:
                    unexpected_list = [
                        os.path.join(csn_folder, unexpected_file)
                        for unexpected_file in unexpected
                    ]
                    logging.warning(
                        f"Unexpected files: {sorted(unexpected_list)}. Just "
                        "the specific .csv files should be saved in csns folders.",
                    )


class BedmasterChecker:
    """
    Implementation of Checker for Bedmaster.
    """

    def __init__(self, bedmaster_dir: str, alarms_dir: str):
        """
        Init Bedmaster Checker.

        :param bedmaster_dir: <str> directory containing all the Bedmaster data.
        """
        self.bedmaster_dir = bedmaster_dir
        self.alarms_dir = alarms_dir

    def check_mat_files_structure(self, patient_csv: str = None, xref: str = None):
        """
        Checks if bedmaster_dir is structured properly.

        Checks that the .mat files in bedmaster_dir are in the right format and
        it doesn't have any unexpected file.
        :param patient_csv: <str> Path to CSV with Sample IDs to restrict MRNs.
        :param xref: <str> Path to CSV with Sample IDs to restrict Bedmaster files.
        """
        bedmaster_files_paths, unexpected_files = get_files_in_directory(
            directory=self.bedmaster_dir,
            file_extension=BEDMASTER_EXT,
        )

        if patient_csv and xref:
            mrns = list(pd.read_csv(patient_csv)["MRN"].unique())
            xref_df = pd.read_csv(xref)
            bedmaster_files_names = list(
                xref_df[xref_df["MRN"].isin(mrns)]["fileID"].unique(),
            )

        # Check if there are any unexpected file in bedmaster_dir
        if len(unexpected_files) > 0:
            logging.warning(
                f"Unexpected files: {sorted(unexpected_files)}. "
                f"Just .mat files should be stored in {self.bedmaster_dir}.",
            )

        for bedmaster_file_path in bedmaster_files_paths:
            bedmaster_file_name = os.path.split(bedmaster_file_path)[-1].split(".")[0]
            if (
                patient_csv
                and xref
                and bedmaster_file_name not in bedmaster_files_names
            ):
                continue
            bedmaster_file = h5py.File(bedmaster_file_path, "r")
            groups = set(MATFILE_EXPECTED_GROUPS)
            missing_groups = groups.difference(set(bedmaster_file.keys()))
            # Check missing groups in each Bedmaster file
            if len(missing_groups) > 0:
                logging.error(
                    f"Wrong file format: the groups {sorted(missing_groups)} "
                    f"were not found in the input file {bedmaster_file_path}.",
                )
            # For each Bedmaster file, check each group content
            for group in groups.intersection(set(bedmaster_file.keys())):
                if not isinstance(bedmaster_file[group], h5py.Group):
                    logging.error(
                        f"{group} from input file {bedmaster_file_path} seems to be "
                        "empty or in a wrong format.",
                    )
                elif not bedmaster_file[group].keys():
                    logging.error(
                        f"{group} from input file {bedmaster_file_path} is empty.",
                    )

    def check_alarms_files_structure(self):
        expected_columns = set(ALARMS_FILES["columns"][1:])
        expected_files: Set[str] = set()
        for key_list in ALARMS_FILES["names"]:
            for file_key in ALARMS_FILES["names"][key_list]:
                file_name = f"bedmaster_alarms_{file_key}.csv"
                expected_files.add(os.path.join(self.alarms_dir, file_name))
        alarms_files_path = [
            os.path.join(self.alarms_dir, alarms_file)
            for alarms_file in os.listdir(self.alarms_dir)
            if alarms_file.endswith(".csv")
        ]
        unexpected_files = [
            os.path.join(self.alarms_dir, unexpected_file_name)
            for unexpected_file_name in os.listdir(self.alarms_dir)
            if not unexpected_file_name.endswith(".csv")
        ]

        # Check if there are any unexpected file in alarms_dir
        if len(unexpected_files) > 0:
            logging.warning(
                f"Unexpected files: {sorted(unexpected_files)}. "
                f"Just .csv files should be stored in {self.alarms_dir}.",
            )

        # Check files structure
        for alarms_file_path in alarms_files_path:
            alarms_file = pd.read_csv(alarms_file_path)
            columns = set(alarms_file.columns)
            missing_columns = expected_columns.difference(columns)
            if len(missing_columns) > 0:
                logging.error(
                    f"Wrong file format: the columns {sorted(missing_columns)} "
                    f"were not found in the input file {alarms_file_path}.",
                )

        # Check that exist a mapping name for every file
        unknown_files = set(alarms_files_path).difference(expected_files)
        for unknown_file in unknown_files:
            logging.warning(
                f"File name {unknown_file} is not mapped in ALARMS_FILES['names'].",
            )

        # Check if all the mapping names have their own file
        missing_files = expected_files.difference(set(alarms_files_path))
        for missing_file in missing_files:
            missing_map = missing_file.split("_")[-1][:-4]
            logging.warning(
                f"Missing file: the mapping name {missing_map} in ALARMS_FILES['names']"
                f" doesn't have its corresponding file {missing_file}.",
            )


def check_icu_structure(args):
    if args.check_edw:
        edw_checker = EDWChecker(args.edw)
        edw_checker.check_structure(args.patient_csv)
    if args.check_bedmaster:
        bedmaster_checker = BedmasterChecker(args.bedmaster, args.alarms)
        bedmaster_checker.check_mat_files_structure(args.patient_csv, args.xref)
        bedmaster_checker.check_alarms_files_structure()
