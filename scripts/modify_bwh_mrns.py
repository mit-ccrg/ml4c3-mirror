# Imports: standard library
import os
import shutil
import argparse
from timeit import default_timer as timer

# Imports: third party
import pandas as pd
from tqdm import tqdm


def _modify_csv(
    path: str,
    bwh_mrn_columns: list,
    mrn_delta: int,
    mrn_columns_to_drop: list,
    csv_new_file_suffix: str,
):
    df = pd.read_csv(path)
    df.columns = map(str.lower, df.columns)
    bwh_mrn_columns_found = list(set(df.columns) & set(bwh_mrn_columns))
    df_new = pd.DataFrame()

    # If data has BWH MRN column, create copy of df,
    # increment MRNs, rename column, and append to new df
    if len(bwh_mrn_columns_found) == 1:
        bwh_mrn_column_found = bwh_mrn_columns_found[0]
        _df = df.copy(deep=True)
        _df[bwh_mrn_column_found] += mrn_delta
        _df.rename(columns={bwh_mrn_column_found: "mrn"}, inplace=True)
        df_new = pd.concat([df_new, _df])
    elif len(bwh_mrn_columns_found) > 1:
        raise KeyError(
            f">1 possible BWH MRN columns. Format the data in {path}",
        )

    # If data has MGH MRN column, create copy of df, rename column,
    # and append to new df
    if "mgh_mrn" in df.columns:
        _df = df.copy(deep=True)
        _df.rename(columns={"mgh_mrn": "mrn"}, inplace=True)
        df_new = pd.concat([df_new, _df])

    # Remove legacy MRN columns
    df_new.drop(columns=mrn_columns_to_drop, inplace=True, errors="ignore")

    # If id in df, sort by id
    if "id" in df_new:
        df_new.sort_values(by=["id"], inplace=True)

    # Drop all NaN MRNs
    df_new.dropna(subset=["mrn"], inplace=True)

    # Reset index
    df_new.reset_index(drop=True, inplace=True)

    # Cast MRN column to ints
    df_new["mrn"] = df_new["mrn"].astype(int)

    # Get new file name, removing "-raw"
    path_prefix, filename = os.path.split(path)
    filename = filename.replace(".csv", f"{csv_new_file_suffix}.csv")
    path_new = os.path.join(path_prefix, filename)

    # Save new CSV
    df_new.to_csv(path_or_buf=path_new, index=False)
    print(f"Saved {path_new}")


def _get_new_mrn(path: str, mrn_delta: int) -> tuple:
    path_prefix, filename = os.path.split(path)
    mrn = int(filename.replace(".hd5", ""))
    mrn_new = mrn + mrn_delta
    return path_prefix, mrn_new


def _get_paths(root_path: str, ext: str) -> list:
    if os.path.isfile(root_path):
        return [root_path]
    paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            split = os.path.splitext(file)
            if split[-1] != ext:
                continue
            path = os.path.join(root, file)
            paths.append(path)
    return paths


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bwh_mrn_columns",
        default=["bwh_mrn"],
        help="Names of BWH MRN colums in CSV files",
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        default=[],
        help="Path to directories of CSV files, or individual CSV files.",
    )
    parser.add_argument(
        "--csv_new_file_suffix",
        default="formatted-mrn",
        help="String to append to file name of modified CSV",
    )
    parser.add_argument(
        "--hd5",
        nargs="+",
        default=[],
        help="Path to directories with HD5 files of BWH patients",
    )
    parser.add_argument(
        "--hd5_new_dir",
        help="Path to directory to save new HD5 files of BWH patients",
    )
    parser.add_argument(
        "--rename_hd5",
        action="store_true",
        help="If true, rename HD5 files. If false, create new files in hd5-new-dir",
    )
    parser.add_argument(
        "--mrn_columns_to_drop",
        default=["mgh_mrn", "bwh_mrn"],
        help="Names of MRN columns that should be completely removed",
    )
    parser.add_argument(
        "--mrn_delta",
        type=int,
        default=100000000,
        help="Delta to add to BWH MRNs. Must be 100M or more since BWH MRNs have 8 digits",
    )
    args = parser.parse_args()
    if not isinstance(args.bwh_mrn_columns, list):
        args.bwh_mrn_columns = [args.bwh_mrn_columns]
    if len(args.hd5) > 0 and not args.rename_hd5 and args.hd5_new_dir is None:
        raise ValueError(f"Must specify --hd5_new_dir if --rename is False.")
    if (
        len(args.hd5) > 0
        and args.hd5_new_dir is not None
        and not os.path.isdir(args.hd5_new_dir)
    ):
        os.mkdir(args.hd5_new_dir)
        print(f"Created directory: {args.hd5_new_dir}")
    return args


if __name__ == "__main__":
    """
    Finds CSV files
    - checks if any columns have columns such as bwh_mrn
    - adds delta to each MRN
    - saves data in new row with a new column name "mrn"
    - if "mgh_mrn" column exists, rename to "mrn"
    - saves this dataframe to a new CSV

    Finds HD5 files
    - Renames or moves them to new directory
    """
    start_time = timer()
    args = _parse_args()

    # Process CSV files
    for root_path in args.csv:
        paths = _get_paths(root_path=root_path, ext=".csv")
        print(f"Processing {len(paths)} CSV files")
        for path in tqdm(paths):
            try:
                _modify_csv(
                    path=path,
                    bwh_mrn_columns=args.bwh_mrn_columns,
                    mrn_delta=args.mrn_delta,
                    mrn_columns_to_drop=args.mrn_columns_to_drop,
                    csv_new_file_suffix=args.csv_new_file_suffix,
                )
            except Exception as e:
                print(f"Failed parsing {path} with error: {e}")

    # Process HD5 files
    for root_path in args.hd5:
        paths = _get_paths(root_path=root_path, ext=".hd5")
        print(f"Processing {len(paths)} HD5 files")
        for path in tqdm(paths):
            try:
                path_prefix, mrn_new = _get_new_mrn(path=path, mrn_delta=args.mrn_delta)
                if args.rename_hd5:
                    path_new = os.path.join(path_prefix, str(mrn_new))
                    os.rename(path, path_new)
                else:
                    path_new = os.path.join(args.hd5_new_dir, str(mrn_new) + ".hd5")
                    shutil.copy(path, path_new)
            except Exception as e:
                print(f"Failed parsing {path} with error: {e}")
    elapsed_time = timer() - start_time
    print(f"Total run time: {elapsed_time:.2f}s")
