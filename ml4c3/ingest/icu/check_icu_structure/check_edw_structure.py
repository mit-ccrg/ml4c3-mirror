# Imports: standard library
import os
import logging

# Imports: third party
import pandas as pd

# Imports: first party
from ml4c3.definitions.icu import EDW_FILES


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

    def check_structure(self):
        """
        Checks if edw_dir is structured properly.
        """
        self._check_adt()

        expected_columns = {}
        for element in EDW_FILES:
            expected_columns[EDW_FILES[element]["name"]] = set(
                EDW_FILES[element]["columns"],
            )
        expected_files = set(expected_columns.keys())
        expected_files.remove(EDW_FILES["adt_file"]["name"])

        mrns_folders = [
            os.path.join(self.edw_dir, folder)
            for folder in os.listdir(self.edw_dir)
            if os.path.isdir(os.path.join(self.edw_dir, folder))
        ]

        for mrn_folder in mrns_folders:
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
