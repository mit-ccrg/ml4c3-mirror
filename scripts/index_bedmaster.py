# Imports: standard library
import os
import re
import socket
import argparse
import subprocess
import multiprocessing as mp
from typing import Set, Dict, List, Tuple, Optional
from datetime import datetime

# Imports: third party
import h5py
import pyodbc

cur = None


def connect_to_index(username: str, password: str) -> pyodbc.Cursor:
    conn = pyodbc.connect(
        f"DRIVER={{PostgreSQL Unicode}};"
        f"SERVER=mithril.partners.org;"
        f"PORT=5432;"
        f"DATABASE=ml4c3;"
        f"UID={username};"
        f"PASSWORD={password};",
    )
    conn.autocommit = True
    cur = conn.cursor()
    return cur


def list_files(bedmaster: str, departments: Optional[Set[str]]) -> Dict[str, str]:
    file_timestamps = {}
    # Regex pattern to match output of ls --full-time.
    # - The first fragment matches the file edit time with YYYY-MM-DD HH:MM:SS
    # - The [^\n]*[\s] is key to matching the filename on the same line as the edit time
    # - The end fragment matches the name of the file [^\s]*.mat
    pattern = r"([\d]{4}-[\d]{2}-[\d]{2} [\d]{2}:[\d]{2}:[\d]{2})[^\n]*[\s]([^\s]*.mat)"
    for root, dirs, files in os.walk(bedmaster):
        root = root.replace(bedmaster, "").lstrip("/")
        for _dir in dirs:
            if departments is not None and _dir not in departments:
                continue
            _dir = os.path.join(root, _dir)
            # raw output from ls as bytes
            _ls_dir = subprocess.check_output(
                f"ls --full-time {os.path.join(bedmaster, _dir)}".split(),
            )
            # bytes -> str
            _ls_dir = _ls_dir.decode()
            # str -> list of tuples, edit datetime and file
            _ls_dir = re.findall(pattern, _ls_dir)
            # list -> dict mapping full path to edit datetime
            _ls_dir = {os.path.join(_dir, file): edit_t for edit_t, file in _ls_dir}
            file_timestamps.update(_ls_dir)
    return file_timestamps


def get_unindexed_files(
    file_timestamps: Dict[str, str],
    reindex_prior_errors: bool,
) -> List[Tuple[str, datetime]]:
    global cur
    # Create a temporary table and fill it with all the files we found
    temp_table = f"{socket.gethostname()}_temp"
    cur.execute(f"DROP TABLE IF EXISTS {temp_table}")
    cur.execute(
        f'CREATE TABLE {temp_table} ("Path" varchar(100), "EditTime" timestamp)',
    )
    cur.execute(
        f"INSERT INTO {temp_table} VALUES "
        + ", ".join(
            [f"('{fpath}', '{t_edit}')" for fpath, t_edit in file_timestamps.items()],
        ),
    )
    # Cross reference all the files with the ones we've previously indexed.
    # The ones that are yet to be indexed:
    # 1. do not appear in the bedmaster table,
    # 2. have an EditTime that differs between the temporary table and the bedmaster table,
    # 3. or, if reindexing files that previously errored, have a non-empty error column.
    unindexed_query = (
        f"SELECT {temp_table}.* FROM {temp_table} "
        f'LEFT JOIN bedmaster ON {temp_table}."Path" = bedmaster."Path" '
        f'WHERE bedmaster."Path" IS NULL '
        f'OR {temp_table}."EditTime" != bedmaster."EditTime"'
    )
    if reindex_prior_errors:
        unindexed_query += ' OR bedmaster."Error" IS NOT NULL'
    to_process = cur.execute(unindexed_query).fetchall()
    cur.execute(f"DROP TABLE IF EXISTS {temp_table}")
    return to_process


def process_batch(to_process_batch: List[Tuple[str, datetime]], bedmaster: str):
    global cur
    fpaths = []
    edits = []
    departments = []
    room_beds = []
    starts = []
    ends = []
    for fpath, t_edit in to_process_batch:
        try:
            # Split Bedmaster file name into fields:
            # <department>_<room_bed>-<start_t>-#_v4.mat is mapped to
            # department, room_bed, start_t
            file_name = os.path.split(fpath)[1]
            split_info = re.split("[^a-zA-Z0-9]", file_name)

            # If department name contains a hyphen, rejoin the name again:
            # ELL-7 --> ELL, 7 --> ELL-7
            if split_info[0].isalpha() and len(split_info[1]) == 1:
                department = split_info[0] + "-" + split_info[1]
                # split_info[1] contains the second part of the department name,
                # e.g. 7 in ELL-7. So we need to delete it so the subsequent indices
                # are consistent with non-hyphenated department names.
                del split_info[1]
            else:
                department = split_info[0]
            k = 2

            # If Bedmaster file name contains a "+" or "~" sign, take next field as it
            # will contain an empty one.
            if split_info[k] == "":
                k = 3

            room_bed = split_info[1]

            if any(s.isalpha() for s in room_bed):
                room_bed = room_bed[:-1] + " " + room_bed[-1]
            else:
                room_bed = room_bed + " A"
            while len(room_bed) < 6:
                room_bed = "0" + room_bed

            # Prepend room bed identifier with first character of department
            room_bed_nm = department[0] + room_bed
            start = 4102444800
            end = 0
            with h5py.File(os.path.join(bedmaster, fpath), "r") as hd5:
                for wv_vs in ["wv", "vs"]:
                    key1 = f"{wv_vs}_time_corrected"
                    if key1 not in hd5 or not isinstance(hd5[key1], h5py.Group):
                        continue
                    for signal in hd5[key1]:
                        key2 = f"{key1}/{signal}/res_{wv_vs}"
                        if key2 not in hd5 or not isinstance(hd5[key2], h5py.Dataset):
                            continue
                        t = hd5[key2]
                        if (_start := t[0][0]) < start:
                            start = int(_start)
                        if (_end := t[-1][0]) > end:
                            end = int(_end)

            if start == 4102444800 or end == 0:
                raise ValueError("could not get start/end time")

            fpaths.append(fpath)
            edits.append(t_edit)
            departments.append(department)
            room_beds.append(room_bed_nm)
            starts.append(start)
            ends.append(end)
        except Exception as e:
            e = str(e)[:100].replace("'", "")
            s = (
                f'INSERT INTO bedmaster ("Path", "EditTime", "Error") '
                f"VALUES ('{fpath}', '{t_edit}', '{e}') "
                f'ON CONFLICT ("Path") DO UPDATE SET '
                f'"Path"=EXCLUDED."Path", '
                f'"EditTime"=EXCLUDED."EditTime", '
                f'"DepartmentDSC"=NULL, '
                f'"BedLabelNM"=NULL, '
                f'"StartTime"=NULL, '
                f'"EndTime"=NULL, '
                f'"Error"=EXCLUDED."Error"'
            )
            try:
                cur.execute(s)
            except Exception as e:
                print(s)
                raise e

    values = ", ".join(
        [
            f"('{fpath}', '{t_edit}', '{dept}', '{bed}', '{t_start}', '{t_end}')"
            for fpath, t_edit, dept, bed, t_start, t_end in zip(
                fpaths,
                edits,
                departments,
                room_beds,
                starts,
                ends,
            )
        ],
    )
    cur.execute(
        f"INSERT INTO bedmaster "
        f'("Path", "EditTime", "DepartmentDSC", "BedLabelNM", "StartTime", "EndTime") '
        f"VALUES {values} "
        f'ON CONFLICT ("Path") DO UPDATE SET '
        f'"Path"=EXCLUDED."Path", '
        f'"EditTime"=EXCLUDED."EditTime", '
        f'"DepartmentDSC"=EXCLUDED."DepartmentDSC", '
        f'"BedLabelNM"=EXCLUDED."BedLabelNM", '
        f'"StartTime"=EXCLUDED."StartTime", '
        f'"EndTime"=EXCLUDED."EndTime", '
        f'"Error"=NULL',
    )


def run(args: argparse.Namespace):
    print("Indexer started")

    # List all files in the bedmaster directory, retrieving the file edit timestamp,
    # and optionally only including those in the requested departments
    file_timestamps = list_files(bedmaster=args.bedmaster, departments=args.departments)
    if args.departments:
        print(f"Got list of all files at {args.bedmaster} within specified departments")
    else:
        print(f"Got list of all files at {args.bedmaster}")
    global cur

    # Connect to the postgres database
    cur = connect_to_index(username=args.username, password=args.password)

    # Cross reference the list of all files with those already in the index, filtering
    # to a list of only the files that are yet to be indexed, optionally retrieving
    # the files which previously failed indexing
    to_process = get_unindexed_files(
        file_timestamps=file_timestamps,
        reindex_prior_errors=args.reindex_prior_errors,
    )
    if args.reindex_prior_errors:
        print(
            f"Found {len(to_process)} unindexed files, including previously failed files",
        )
    else:
        print(f"Found {len(to_process)} unindexed files")

    # Concurrently index batches of files and update the database
    batches = [
        to_process[i : i + args.batch_size]
        for i in range(0, len(to_process), args.batch_size)
    ]
    with mp.Pool(processes=args.num_workers) as pool:
        pool.starmap(process_batch, zip(batches, [args.bedmaster] * len(batches)))

    # Download the index to a CSV file
    cur.execute(f"COPY bedmaster to '{args.index_file}' DELIMITER ',' CSV HEADER")
    print(f"Downloaded index to {args.index_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help=(
            "The number of files to index before updating the database, setting a "
            "higher batch size reduces the number of database transactions but "
            "increases the amount of data that might be lost if the indexer is "
            "interrupted before updating the database."
        ),
    )
    parser.add_argument(
        "--bedmaster",
        help="The bedmaster directory to index.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=3,
        help="The number of concurrent workers that index batches of files.",
    )
    parser.add_argument(
        "--departments",
        nargs="*",
        help=(
            "An optional list of department names to restrict the index to, ideal for "
            "distributing indexing of different departments across index jobs."
        ),
    )
    parser.add_argument(
        "--reindex_prior_errors",
        action="store_true",
        help=(
            "If set, attempt to reindex files that previously errored, ideal for "
            "corrected files whose file creation time has not changed."
        ),
    )
    parser.add_argument(
        "--index_file",
        help="The path to save the index to after all files have been indexed.",
    )
    parser.add_argument(
        "--username",
        help="The username to connect to the index database with.",
    )
    parser.add_argument(
        "--password",
        help="The password of the database user.",
    )
    args = parser.parse_args()
    if args.departments is not None:
        args.departments = set(args.departments)
    return args


if __name__ == "__main__":
    args = parse_args()
    run(args)
