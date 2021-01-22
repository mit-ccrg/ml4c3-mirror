# Imports: standard library
import os
import logging

# Imports: third party
import pandas as pd


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
