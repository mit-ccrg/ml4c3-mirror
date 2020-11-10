# Imports: standard library
import os
import shutil
import argparse
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count

# Imports: third party
import h5py
import numpy as np
import pandas as pd

# Imports: first party
from ml4c3.datasets import _sample_csv_to_set
from ml4c3.definitions.ecg import ECG_PREFIX
from ml4c3.definitions.globals import CSV_EXT, TENSOR_EXT, MRN_COLUMNS

"""
To add a new data source to deidentify:
1. add the function to get MRNs from the data to _get_mrns()
2. add the function to deidentify the data to run()
"""

phi_keys = {
    "attendingmdfirstname",
    "attendingmdhisid",
    "attendingmdlastname",
    "consultingmdfirstname",
    "consultingmdhisid",
    "consultingmdlastname",
    "hisorderingmdfirstname",
    "hisorderingmdlastname",
    "orderingmdfirstname",
    "orderingmdhisid",
    "orderingmdid",
    "orderingmdlastname",
    "placersfirstname",
    "placershisid",
    "placerslastname",
    "patientfirstname",
    "patientlastname",
    "patientid",
    "patientid_clean",
    "acquisitiontechfirstname",
    "acquisitiontechid",
    "acquisitiontechlastname",
    "admittingmdfirstname",
    "admittingmdhisid",
    "admittingmdlastname",
    "fellowfirstname",
    "fellowlastname",
    "fellowid",
    "hisaccountnumber",
    "referringmdid",
    "editorfirstname",
    "editorlastname",
    "editorid",
    "overreaderfirstname",
    "overreaderlastname",
    "overreaderid",
}


def _get_hd5_mrns(args):
    mrns = set()
    if args.path_to_hd5_deidentified is not None:
        for root, dirs, files in os.walk(args.path_to_hd5):
            for file in files:
                split = os.path.splitext(file)
                if split[-1] != TENSOR_EXT:
                    continue
                try:
                    mrn = int(split[0])
                    mrns.add(mrn)
                except ValueError:
                    print(f"Could not get MRN from ECG HD5: {os.path.join(root, file)}")
                    continue
    return mrns


path_of_csv_to_skip = set()


def _get_csv_mrns(args):
    mrns = set()
    if args.path_to_csv_deidentified is not None:

        # Get list of full paths to CSV files
        fpaths = []
        if os.path.isdir(args.path_to_csv):
            for root, dirs, fnames in os.walk(args.path_to_csv):
                for fname in fnames:
                    split = os.path.splitext(fname)
                    if split[-1] != CSV_EXT:
                        continue
                    fpath = os.path.join(root, fname)
                    fpaths.append(fpath)
        # If user gave path to single CSV, instead of a directory, use that path
        else:
            fpaths.append(args.path_to_csv)

        # Iterate over paths to CSV files
        for fpath in fpaths:
            try:
                _mrns = _sample_csv_to_set(sample_csv=fpath)
            except ValueError:
                print(f"Could not get MRNs from {fpath}, skipping de-identification")
                global path_of_csv_to_skip
                path_of_csv_to_skip.add(fpath)
                continue
            _mrns = {int(mrn) for mrn in _mrns}
            mrns |= _mrns

    return mrns


def _get_mrns(args, skip_mrns=set()):
    """
    Get a list of unique MRNs from the data sources that are being remapped.
    """
    mrns = set()
    mrns |= _get_hd5_mrns(args)
    mrns |= _get_csv_mrns(args)
    mrns -= skip_mrns
    return mrns


def _remap_mrns(args):
    """
    Remap and save the MRNs from the data sources that are being remapped to new random IDs.
    """
    mrn_map = dict()
    starting_id = args.starting_id
    if os.path.isfile(args.mrn_map):
        # call to _get_csv_mrns to determine which files to skip for sts deidentification
        _get_csv_mrns(args)

        mrn_map = pd.read_csv(args.mrn_map, low_memory=False, usecols=["mrn", "new_id"])
        mrn_map = mrn_map.set_index("mrn")
        mrn_map = mrn_map["new_id"].to_dict()
        if starting_id is None:
            starting_id = max(mrn_map.values()) + 1
        print(f"Existing MRN map loaded from {args.mrn_map}")

    new_mrns = _get_mrns(args, skip_mrns=set(mrn_map.keys()))
    new_ids = list(range(starting_id, len(new_mrns) + starting_id))
    np.random.shuffle(new_ids)

    mrn_map.update(dict(zip(new_mrns, new_ids)))
    print(f"New MRNs remapped starting at ID {starting_id}")

    df = pd.DataFrame.from_dict(mrn_map, orient="index", columns=["new_id"])
    df.index.name = "mrn"
    os.makedirs(os.path.dirname(args.mrn_map), exist_ok=True)
    df.sort_values("new_id").to_csv(args.mrn_map)
    print(f"MRN map saved to {args.mrn_map}")
    print(f"Last ID used in remapping MRNs was {df['new_id'].max()}")

    return mrn_map


def _swap_path_prefix(path, prefix, new_prefix):
    """
    Given:
        path = /foo/bar
        prefix = /foo
        new_prefix = /baz
    Returns:
        /baz/bar
    """
    path_relative_root = path.replace(prefix, "").lstrip("/")
    new_path = os.path.join(new_prefix, path_relative_root)
    if not os.path.isdir(new_path):
        os.makedirs(new_path, exist_ok=True)
    return new_path


def _deidentify_hd5(old_new_path):
    """
    Given a path to an existing HD5, copy it to a new path and delete all identifiable
    information. Currently only set up for ECG data.
    """
    old_path, new_path = old_new_path
    if os.path.exists(new_path):
        os.remove(new_path)
    shutil.copyfile(old_path, new_path)

    with h5py.File(new_path, "r+") as hd5:
        # Only delete PHI keys from HD5s that lack 'deidentified' flag
        if ECG_PREFIX in hd5 and "deidentified" not in hd5:
            for ecg_date in hd5[ECG_PREFIX]:
                for key in hd5[ECG_PREFIX][ecg_date]:
                    if key in phi_keys:
                        del hd5[ECG_PREFIX][ecg_date][key]

            # Add bool to hd5 indicating this file is de-identified
            hd5.create_dataset("deidentified", data=True, dtype=bool)


def _deidentify_hd5s(args, mrn_map):
    """
    Create de-identified HD5 files in parallel.
    """
    if args.path_to_hd5_deidentified is None:
        return

    old_new_paths = []
    for root, dirs, files in os.walk(args.path_to_hd5):
        new_root = _swap_path_prefix(
            root,
            args.path_to_hd5,
            args.path_to_hd5_deidentified,
        )
        for file in files:
            split = os.path.splitext(file)
            if split[-1] != TENSOR_EXT:
                continue
            try:
                mrn = int(split[0])
                new_id = mrn_map[mrn]
            except (ValueError, KeyError):
                print(f"Bad MRN mapping for ECG HD5: {os.path.join(root, file)}")
                continue
            old_path = os.path.join(root, file)
            new_path = os.path.join(new_root, f"{new_id}{TENSOR_EXT}")
            old_new_paths.append((old_path, new_path))

    with Pool(processes=args.num_workers) as pool:
        pool.map(_deidentify_hd5, old_new_paths)

    print(f"De-identified {len(old_new_paths)} ECGs at {args.path_to_hd5_deidentified}")


def _deidentify_csv(path, mrn_map):
    """
    Given a path to a CSV, delete all identifiable information.
    """
    df = pd.read_csv(path, header=None, low_memory=False)

    # Infer csv header
    try:
        # If first cell is an int, it's likely a sample ID and there is no header
        int(df.iloc[0].values[0])
    except ValueError:
        df.columns = df.iloc[0]
        df = df[1:]

    # Cast each column name in df to string (in case column name is an int)
    # and check if defined MRN column names match the column names in the df
    matches = {
        col for col in df.columns for mrn_col in MRN_COLUMNS if mrn_col in str(col)
    }
    if len(matches) == 0:
        # If none of the known MRN columns are in the csv, assume it's the first column
        mrn_cols = [df.columns[0]]
    else:
        mrn_cols = list(matches)

    # Remap MRNs and drop PHI columns
    df[mrn_cols] = df[mrn_cols].applymap(lambda mrn: mrn_map[int(float(mrn))])
    phi_cols = set(df.columns) & phi_keys
    df = df.drop(phi_cols, axis=1)

    df.to_csv(path, index=False)


def _deidentify_csvs(args, mrn_map):
    """
    Create de-identified STS data.
    """
    if args.path_to_csv_deidentified is None:
        return

    count = 0
    for root, dirs, files in os.walk(args.path_to_csv):
        new_root = _swap_path_prefix(
            root,
            args.path_to_csv,
            args.path_to_csv_deidentified,
        )
        for file in files:
            split = os.path.splitext(file)
            if split[-1] != CSV_EXT:
                continue
            old_path = os.path.join(root, file)
            new_path = os.path.join(new_root, file)
            if os.path.exists(new_path):
                os.remove(new_path)
            shutil.copyfile(old_path, new_path)
            # if there was no PHI in the original file, copy it without trying to deidentify
            global path_of_csv_to_skip
            if old_path in path_of_csv_to_skip:
                continue
            _deidentify_csv(new_path, mrn_map)
            count += 1

    print(f"De-identified {count} CSV files at {args.path_to_csv_deidentified}")


def run(args):
    start_time = timer()
    mrn_map = _remap_mrns(args)
    _deidentify_hd5s(args, mrn_map)
    _deidentify_csvs(args, mrn_map)
    end_time = timer()
    elapsed_time = end_time - start_time
    print(f"De-identification took {elapsed_time:.2f} seconds.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mrn_map",
        default=os.path.expanduser("~/dropbox/mrn-deid-maps/mgh.csv"),
        help="Path to CSV in which to store map from MRN -> deidentified ID.",
    )
    parser.add_argument(
        "--starting_id",
        type=int,
        default=1,
        help="Starting value for new IDs.",
    )
    parser.add_argument(
        "--path_to_hd5",
        default="/storage/shared/ecg/mgh",
        help="Path to directory containing hd5 files.",
    )
    parser.add_argument(
        "--path_to_hd5_deidentified",
        help="Path to save de-identified ECG HD5s to. "
        "Skip this argument to skip de-identification of ECG data.",
    )
    parser.add_argument(
        "--path_to_csv",
        help="Path to directory of CSV files, or CSV file.",
    )
    parser.add_argument(
        "--path_to_csv_deidentified",
        help="Path to save de-identified CSVs to. "
        "Skip this argument to skip de-identification of STS data.",
    )
    parser.add_argument(
        "--num_workers",
        default=cpu_count() - 1,
        type=int,
        help="Number of worker processes to use if processing in parallel.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
