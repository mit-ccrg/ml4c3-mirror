# Imports: standard library
import os
import logging
from typing import Set

# Imports: third party
import h5py
import pandas as pd

# Imports: first party
from ml4c3.definitions.icu import ALARMS_FILES, MATFILE_EXPECTED_GROUPS


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

    def check_mat_files_structure(self):
        """
        Checks if bedmaster_dir is structured properly.

        Checks that the .mat files in bedmaster_dir are in the right format and
        it doesn't have any unexpected file.
        """
        bedmaster_files_paths = [
            os.path.join(self.bedmaster_dir, bedmaster_file_name)
            for bedmaster_file_name in os.listdir(self.bedmaster_dir)
            if bedmaster_file_name.endswith(".mat")
        ]
        unexpected_files = [
            os.path.join(self.bedmaster_dir, unexpected_file_name)
            for unexpected_file_name in os.listdir(self.bedmaster_dir)
            if not unexpected_file_name.endswith(".mat")
        ]
        # Check if there are any unexpected file in bedmaster_dir
        if len(unexpected_files) > 0:
            logging.warning(
                f"Unexpected files: {sorted(unexpected_files)}. "
                f"Just .mat files should be stored in {self.bedmaster_dir}.",
            )

        for bedmaster_file_path in bedmaster_files_paths:
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
                f"File name {unknown_file} is not mapped in "
                "ALARMS_FILES['names'] (ml4icu/globals.py).",
            )

        # Check if all the mapping names have their own file
        missing_files = expected_files.difference(set(alarms_files_path))
        for missing_file in missing_files:
            missing_map = missing_file.split("_")[-1][:-4]
            logging.warning(
                f"Missing file: the mapping name {missing_map} in "
                "ALARMS_FILES['names'] (ml4icu/globals.py) doesn't have its "
                f"corresponding file {missing_file}.",
            )
