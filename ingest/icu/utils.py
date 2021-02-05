# Imports: standard library
import os
import shutil
import logging

# Imports: third party
import pandas as pd


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
    bedmaster: str,
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
        fpath_source_file = os.path.join(bedmaster, path_source_file)
        fpath_destination_file = os.path.join(path_destination_dir, path_source_file)
        fpath_destination_dir = os.path.split(fpath_destination_file)[0]
        if os.path.exists(fpath_source_file):
            try:
                os.makedirs(fpath_destination_dir, exist_ok=True)
                shutil.copy(fpath_source_file, fpath_destination_file)
            except FileNotFoundError as e:
                logging.warning(f"{fpath_source_file} not found. Error given: {e}")
        else:
            logging.warning(f"{fpath_source_file} not found.")
