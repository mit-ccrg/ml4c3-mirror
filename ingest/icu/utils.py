# Imports: standard library
import os
import shutil
import logging

# Imports: third party
import pandas as pd

# Imports: first party
from definitions.icu import MAPPING_DEPARTMENTS


def save_mrns_and_csns_csv(
    staging_dir: str,
    hd5_dir: str,
    adt: str,
    first_mrn_index: int,
    last_mrn_index: int,
    overwrite_hd5: bool,
):
    """
    Get unique MRNs and CSNs from ADT and save to patients.csv.

    :param staging_dir: <str> Path to temporary staging directory.
    :param hd5_dir: <str> Path to directory where hd5 files are stored.
    :param adt: <str> Path to CSV containing ADT table.
    :param first_mrn_index: <int> First index of desired MRNs.
    :param last_mrn_index: <int> Last index of desired MRNs.
    :param overwrite_hd5: <bool> Overwrite existing hd5 files.
    """
    adt_df = pd.read_csv(adt).sort_values(by=["MRN"], ascending=True)
    patients = adt_df[["MRN", "PatientEncounterID"]].drop_duplicates().dropna()
    mrns = patients["MRN"].drop_duplicates()[first_mrn_index:last_mrn_index]
    mrns_and_csns = patients[patients["MRN"].isin(mrns)]
    if not overwrite_hd5 and os.path.isdir(hd5_dir):
        hd5_mrns = [
            int(hd5_mrn.split(".")[0])
            for hd5_mrn in os.listdir(hd5_dir)
            if hd5_mrn.endswith(".hd5")
        ]
        mrns_and_csns = mrns_and_csns[~mrns_and_csns["MRN"].isin(hd5_mrns)]

    mrns_and_csns_path = os.path.join(staging_dir, "patients.csv")
    mrns_and_csns.to_csv(mrns_and_csns_path, index=False)
    logging.info(f"Saved {mrns_and_csns_path}")


def stage_bedmaster_alarms(
    staging_dir: str,
    adt: str,
    alarms: str,
):
    """
    Find Bedmaster alarms and copy them to staging directory.

    :param staging_dir: <str> Path to temporary staging directory.
    :param adt: <str> Path to CSV containing ADT table.
    :param alarms: <str> Path to directory with alarm data.
    """
    path_patients = os.path.join(staging_dir, "patients.csv")
    mrns_and_csns = pd.read_csv(path_patients)
    mrns = mrns_and_csns["MRN"].drop_duplicates()

    adt_df = pd.read_csv(adt).sort_values(by=["MRN"], ascending=True)
    adt_filt = adt_df[adt_df["MRN"].isin(mrns)]

    departments = adt_filt["DepartmentDSC"].drop_duplicates()
    for department in departments:
        try:
            short_names = MAPPING_DEPARTMENTS[department]
        except KeyError:
            continue
        # Skip department short names that are None
        for short_name in [sn for sn in short_names if sn is not None]:
            source_path = os.path.join(
                alarms,
                f"bedmaster_alarms_{short_name}.csv",
            )
            destination_path = os.path.join(
                staging_dir,
                "bedmaster_alarms_temp",
            )
            try:
                shutil.copy(source_path, destination_path)
            except FileNotFoundError as e:
                logging.warning(f"{source_path} not found. Error given: {e}")


def stage_edw_files(
    staging_dir: str,
    edw: str,
    adt: str,
    xref: str,
):
    """
    Find EDW files and copy them to local folder.

    :param staging_dir: <str> Path to temporary staging directory.
    :param edw: <str> Path to directory with EDW data.
    :param xref: <str> Path to xref.csv with Bedmaster metadata.
    """
    path_patients = os.path.join(staging_dir, "patients.csv")
    mrns_and_csns = pd.read_csv(path_patients)
    mrns = mrns_and_csns["MRN"].drop_duplicates()

    for mrn in mrns:
        source_path = os.path.join(edw, str(mrn))
        destination_path = os.path.join(staging_dir, "edw_temp", str(mrn))
        try:
            shutil.copytree(source_path, destination_path)
        except FileNotFoundError as e:
            logging.warning(f"{source_path} not found. Error given: {e}")

    # Copy ADT table
    adt_new = os.path.join(staging_dir, "edw_temp", "adt.csv")
    shutil.copy(adt, adt_new)

    # Create filtered xref table
    df_xref_filt = pd.read_csv(xref)
    df_xref_filt = df_xref_filt[df_xref_filt["MRN"].isin(mrns)]
    xref_new = os.path.join(staging_dir, "edw_temp", "xref.csv")
    df_xref_filt.to_csv(xref_new, index=False)


def stage_bedmaster_files(
    staging_dir: str,
    xref: str,
):
    """
    Find Bedmaster files and copy them to local folder.

    :param staging_dir: <str> Path to temporary staging directory.
    :param xref: <str> Path to xref.csv with Bedmaster metadata.
    """
    path_patients = os.path.join(staging_dir, "patients.csv")
    mrns_and_csns = pd.read_csv(path_patients)
    mrns = mrns_and_csns["MRN"].drop_duplicates()

    xref_df = pd.read_csv(xref).sort_values(by=["MRN"], ascending=True)
    xref_subset = xref_df[xref_df["MRN"].isin(mrns)]

    # Iterate over all Bedmaster file paths to copy to staging directory
    path_destination_dir = os.path.join(staging_dir, "bedmaster_temp")
    for path_source_file in xref_subset["Path"]:
        fpath_source_file = os.path.join(path_destination_dir, path_source_file)
        fpath_destination_file = os.path.join(path_destination_dir, path_source_file)
        if os.path.exists(fpath_source_file):
            try:
                shutil.copy(fpath_source_file, fpath_destination_file)
            except FileNotFoundError as e:
                logging.warning(f"{fpath_source_file} not found. Error given: {e}")
        else:
            logging.warning(f"{fpath_source_file} not found.")


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
