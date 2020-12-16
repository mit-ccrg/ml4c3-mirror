# Imports: standard library
import os
import shutil
import logging
import warnings
import multiprocessing
from typing import Dict, List

# Imports: third party
import pandas as pd

# Imports: first party
from definitions.icu import MAPPING_DEPARTMENTS


def save_mrns_and_csns_csv(
    path_staging_dir: str,
    hd5_dir: str,
    path_adt: str,
    first_mrn_index: int,
    last_mrn_index: int,
    overwrite_hd5: bool,
):
    """
    Get unique MRNs and CSNs from ADT and save to patients.csv.

    :param path_staging_dir: <str> Path to temporary staging directory.
    :param hd5_dir: <str> Path to directory where hd5 files are stored.
    :param path_adt: <str> Path to CSV containing ADT table.
    :param first_mrn_index: <int> First index of desired MRNs.
    :param last_mrn_index: <int> Last index of desired MRNs.
    :param overwrite_hd5: <bool> Overwrite existing hd5 files.
    """
    adt = pd.read_csv(path_adt).sort_values(by=["MRN"], ascending=True)
    patients = adt[["MRN", "PatientEncounterID"]].drop_duplicates().dropna()
    mrns = patients["MRN"].drop_duplicates()[first_mrn_index:last_mrn_index]
    mrns_and_csns = patients[patients["MRN"].isin(mrns)]
    if not overwrite_hd5 and os.path.isdir(hd5_dir):
        hd5_mrns = [
            int(hd5_mrn.split(".")[0])
            for hd5_mrn in os.listdir(hd5_dir)
            if hd5_mrn.endswith(".hd5")
        ]
        mrns_and_csns = mrns_and_csns[~mrns_and_csns["MRN"].isin(hd5_mrns)]

    mrns_and_csns_path = os.path.join(path_staging_dir, "patients.csv")
    mrns_and_csns.to_csv(mrns_and_csns_path, index=False)
    logging.info(f"Saved {mrns_and_csns_path}")


def stage_bedmaster_alarms(
    path_staging_dir: str,
    path_adt: str,
    path_alarms: str,
):
    """
    Find Bedmaster alarms and copy them to staging directory.

    :param path_staging_dir: <str> Path to temporary staging directory.
    :param path_adt: <str> Path to CSV containing ADT table.
    :param path_alarms: <str> Path to directory with alarm data.
    """
    path_patients = os.path.join(path_staging_dir, "patients.csv")
    mrns_and_csns = pd.read_csv(path_patients)
    mrns = mrns_and_csns["MRN"].drop_duplicates()

    adt = pd.read_csv(path_adt).sort_values(by=["MRN"], ascending=True)
    adt_filt = adt[adt["MRN"].isin(mrns)]

    departments = adt_filt["DepartmentDSC"].drop_duplicates()
    dept_names = []
    for department in departments:
        try:
            short_names = MAPPING_DEPARTMENTS[department]
        except KeyError:
            continue
        # Skip department short names that are None
        for short_name in [sn for sn in short_names if sn is not None]:
            source_path = os.path.join(
                path_alarms,
                f"bedmaster_alarms_{short_name}.csv",
            )
            destination_path = os.path.join(
                path_staging_dir,
                "bedmaster_alarms_temp",
            )
            try:
                shutil.copy(source_path, destination_path)
            except FileNotFoundError as e:
                logging.warning(f"{source_path} not found. Error given: {e}")


def stage_edw_files(
    path_staging_dir: str,
    path_edw: str,
    path_adt: str,
    path_xref: str,
):
    """
    Find EDW files and copy them to local folder.

    :param path_staging_dir: <str> Path to temporary staging directory.
    :param path_edw: <str> Path to directory with EDW data.
    :param path_xref: <str> Path to xref.csv with Bedmaster metadata.
    """
    path_patients = os.path.join(path_staging_dir, "patients.csv")
    mrns_and_csns = pd.read_csv(path_patients)
    mrns = mrns_and_csns["MRN"].drop_duplicates()

    list_mrns = []
    flag_found = []

    for mrn in mrns:
        source_path = os.path.join(path_edw, str(mrn))
        destination_path = os.path.join(path_staging_dir, "edw_temp", str(mrn))
        try:
            shutil.copytree(source_path, destination_path)
        except FileNotFoundError as e:
            logging.warning(f"{source_path} not found. Error given: {e}")

    # Copy ADT table
    path_adt_new = os.path.join(path_staging_dir, "edw_temp", "adt.csv")
    shutil.copy(path_adt, path_adt_new)

    # Copy xref table
    path_xref_new = os.path.join(path_staging_dir, "edw_temp", "xref.csv")
    shutil.copy(path_xref, path_xref_new)


def stage_bedmaster_files(
    path_staging_dir: str,
    path_xref: str,
    path_bedmaster: str,
):
    """
    Find Bedmaster files and copy them to local folder.

    :param path_staging_dir: <str> Path to temporary staging directory.
    :param path_xref: <str> Path to xref.csv with Bedmaster metadata.
    :param path_bedmaster: <str> Path to directory with department subdirectories
           that contain Bedmaster .mat files.
    """
    path_patients = os.path.join(path_staging_dir, "patients.csv")
    mrns_and_csns = pd.read_csv(path_patients)
    mrns = mrns_and_csns["MRN"].drop_duplicates()

    xref = pd.read_csv(path_xref).sort_values(by=["MRN"], ascending=True)
    xref_subset = xref[xref["MRN"].isin(mrns)]

    list_bedmaster_files = []
    folder = []
    flag_found = []

    # Iterate over all Bedmaster file paths to copy to staging directory
    path_destination_dir = os.path.join(path_staging_dir, "bedmaster_temp")
    for path_source_file in xref_subset["path"]:
        if os.path.exists(path_source_file):
            try:
                shutil.copy(path_source_file, path_destination_dir)
            except FileNotFoundError as e:
                logging.warning(f"{path} not found. Error given: {e}")
        else:
            logging.warning(f"{path} not found.")


def get_files_in_directory(
    directory: str,
    file_extension: str,
    departments_short_names: set = None,
) -> tuple:
    """
    Given a path to a directory and a file extension, returns a list of full paths
    to all files ending in the file extension, and a list of full paths to all files
    that do not end in the file extension.

    Optionally, limit search to a subset of departments.
    """
    fpaths = []
    not_fpaths = []
    for root, dirs, files in os.walk(directory, topdown=True):
        if departments_short_names is not None:
            dirs[:] = [d for d in dirs if d in departments_short_names]
        for file in files:
            fpath = os.path.join(root, file)
            if file.endswith(file_extension):
                fpaths.append(fpath)
            else:
                not_fpaths.append(fpath)
    return fpaths, not_fpaths
