# pylint: disable=redefined-outer-name, broad-except
# Imports: standard library
import os
import glob
import shutil
import getpass
import pathlib
import argparse
from timeit import default_timer as timer

# Imports: third party
import numpy as np
import pandas as pd
import pyodbc

# DepartmentID lookup table
DEPT_ID = {}
DEPT_ID["blake7"] = "10020010622"
DEPT_ID["blake8"] = "10020010623"
DEPT_ID["blake10"] = "10020010616"
DEPT_ID["blake12"] = "10020010619"
DEPT_ID["lunder6"] = "10020010637"
DEPT_ID["ellison4"] = "10020010631"
DEPT_ID["ellison8"] = "10020010634"
DEPT_ID["ellison9"] = "10020010635"
DEPT_ID["ellison10"] = "10020010624"
DEPT_ID["ellison11"] = "10020010625"
DEPT_ID["ellison14"] = "10020011140"
DEPT_ID["procedurerm"] = "10020011402"
DEPT_ID["bigelow6"] = "10020010614"


def _format_sql_query(path: str, str_to_insert: str = None) -> str:
    """
    Formats a SQL query from a .sql file into a str, and inserts a str arg
    into the query to enable iteration over lists of parameters
    """
    with open(path, "r") as f:
        # Get SQL query from file as str
        sql_str = f.read()

    # Insert format string into {} in query string
    sql_str = sql_str.format(str_to_insert)

    # Remove leading and trailing whitespaces
    sql_str = sql_str.strip()

    # Remove carriage return characters
    sql_str = sql_str.replace("\r", "")
    return sql_str


def _run_sql_query(
    edw: pyodbc.Connection,
    query_fpath: str,
    str_to_insert: str = None,
    fpath: str = None,
    save_dir: str = "./",
):
    """
    Run SQL query via an EDW network connection, and save output in dataframe
    """
    # Format and run ADT query
    print(f"\tRunning query: {os.path.split(query_fpath)[1]}")
    sql_str = _format_sql_query(query_fpath, str_to_insert)

    # Read SQL query into DataFrame
    df = pd.read_sql_query(sql_str, edw)

    # Isolate name of the query
    if not fpath:
        query_name = os.path.split(query_fpath)[-1][:-4]
        fpath = os.path.join(save_dir, f"{query_name}.csv")

    # Save DataFrame to disk
    df.to_csv(fpath, index=False)
    print(f"\tSaved query results at: {fpath}")


def _run_queries(
    edw: pyodbc.Connection,
    query_fpaths: list,
    list_mrn: list,
    list_csn: list,
    batch_size: int,
    folder_idx: str,
    save_dir: str,
):
    save_dir = os.path.join(save_dir, folder_idx)

    # Check if encounter ID dir exists; if not, make it
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Save tables with lists of batched mrns and csns if there's batching
    if batch_size > 1:
        batch_list = pd.DataFrame(
            np.array((list_mrn, list_csn)).transpose(),
            columns=["MRN", "CSN"],
        )
        batch_list.to_csv(os.path.join(save_dir, "batch_list.csv"))

    # Iterate through SQL queries
    list_csn_str = str(list_csn)[1:-1]
    for query_fpath in query_fpaths:
        _run_sql_query(
            edw=edw,
            query_fpath=query_fpath,
            str_to_insert=list_csn_str,
            save_dir=save_dir,
        )


def _get_mrns_csns_at_destination(
    parent_dir: str,
    query_fpaths: list,
    patients: pd.DataFrame,
) -> tuple:
    """
    Given a directory, obtain a list of all MRN and CSN directories. Assumes a
    directory structure:
    /parent_dir
    ├── 12345 (mrn)
    │   ├── 78982336 (csn)
    │   └── 78937589 (csn)
    ├── 12346 (mrn)
    │   └── 12552657 (csn)
    etc.
    """
    # Determine the list of expected .csv files in each CSN folder
    expected_files = [
        f"{os.path.split(query_fpath)[-1][:-4]}.csv" for query_fpath in query_fpaths
    ]

    # Get MRNs of interest
    if args.just_sample_csv_csns:
        fpath = args.sample_csv
    else:
        fpath = args.path_adt
    mrns = pd.read_csv(fpath)["MRN"].unique()
    mrns = mrns[~np.isnan(mrns)]

    # Get full path to MRN-CSN directories
    patients["paths"] = patients.apply(
        lambda x: os.path.join(
            parent_dir,
            str(int(x["MRN"])),
            str(int(x["PatientEncounterID"])),
        ),
        axis=1,
    )
    csn_dirs = list(patients["paths"].unique())

    # Initialize empty master lists to store MRNs and CSNs
    mrns = []
    csns = []

    # Iterate through MRN-CSN dir and update master lists
    for csn_dir in csn_dirs:
        try:
            csv_files = [
                csv_file
                for csv_file in os.listdir(csn_dir)
                if csv_file.endswith(".csv")
            ]
            for expected_file in expected_files:
                if expected_file not in csv_files:
                    break
            else:
                # Remove the parent_dir prefix to just get '/mrn/csn'
                csn_dir = csn_dir.replace(parent_dir, "")
                _, mrn, csn = csn_dir.split("/")
                mrns.append(float(mrn))
                csns.append(float(csn))
        except FileNotFoundError:
            continue
    return mrns, csns


def _connect_edw(partners_username: str, pwd: str) -> pyodbc.Connection:
    """
    Connect to EDW databse using user: partners_username and password: pwd.
    """
    # EDW
    host = "phsedw.partners.org,1433"
    user = f"PARTNERS\\{partners_username}"

    # Connect to EDW
    try:
        edw = pyodbc.connect(
            driver="FreeTDS",
            TDS_version="8.0",
            host=host,
            user=user,
            pwd=pwd,
        )
        print(f"Connected to {host} using login {user}")
    except Exception as e:
        raise Exception(f"Failed to connect to {host} using login {user}") from e
    return edw


def _obtain_data(
    edw,
    path_to_queries: str,
    args,
):
    # Load EDW queries
    query_fpaths = glob.glob(f"{path_to_queries}/*.sql")
    query_fpaths = [
        query_fpath for query_fpath in query_fpaths if "adt" not in query_fpath
    ]
    print(f"\nFound {len(query_fpaths)} SQL queries in ./{path_to_queries}.")

    # Get path to ADT query
    if os.path.isfile(args.path_adt):
        print("ADT table found, no need to run adt query.")
    elif args.sample_csv:
        if not args.path_adt:
            args.path_adt = os.path.join(args.output_folder, "adt.csv")
        mrns = pd.read_csv(args.sample_csv)["MRN"].unique()
        mrns = mrns[~np.isnan(mrns)]
        list_mrns = str(list(mrns))[1:-1]
        str_to_insert = list_mrns
        adt_query_fpath = os.path.join(path_to_queries, "adt.sql")
        print(f"Path to ADT query: {adt_query_fpath}")
        _run_sql_query(
            edw=edw,
            query_fpath=adt_query_fpath,
            str_to_insert=str_to_insert,
            fpath=args.path_adt,
        )
    else:
        raise ValueError(
            "Unable to obtain adt table. Specify a sample_csv or "
            "a valid path_adt in args to proceed.",
        )

    # Isolate all unique PatientEncounterIDs and cast from np.array -> list
    adt = pd.read_csv(args.path_adt)
    if args.sample_csv:
        mrns = pd.read_csv(args.sample_csv)["MRN"].unique()
        mrns = mrns[~np.isnan(mrns)]
        adt = adt[adt["MRN"].isin(list(mrns))]
        if args.just_sample_csv_csns:
            csns = pd.read_csv(args.sample_csv)["PatientEncounterID"].unique()
            csns = csns[~np.isnan(csns)]
            adt = adt[adt["PatientEncounterID"].isin(list(csns))]
    patients = adt[["MRN", "PatientEncounterID"]].drop_duplicates()
    patients = patients.sort_values(by=["MRN", "PatientEncounterID"], ascending=True)
    num_rows = len(patients.index)
    print(f"Extracted {num_rows} unique MRN-PatientEncounterID values from adt table.")

    # Batch the patient encounter IDs
    patients = patients[args.first : args.last]
    patients = patients.reset_index()
    num_rows = len(patients.index)
    print(
        f"\nReduced the number of unique MRN-PatientEncounterID values to {num_rows}.",
    )

    # Get MRNs and CSNs at the destination directory
    if not args.overwrite:
        _, csns = _get_mrns_csns_at_destination(
            args.output_folder,
            query_fpaths,
            patients,
        )
    else:
        csns = []

    # Iterate through each MRN and CSN
    list_mrn = []
    list_csn = []
    for index, row in patients.iterrows():
        if row.PatientEncounterID in csns:
            print(
                f"\nMRN: {row.MRN:0.0f} ({index + 1}/{patients.shape[0]}), CSN: "
                f"{row.PatientEncounterID:0.0f} already exists at destination; "
                "skipping.",
            )
        else:
            print(
                f"\nMRN: {row.MRN:0.0f} ({index + 1}/{patients.shape[0]}), CSN: "
                f"{row.PatientEncounterID:0.0f} does not yet exist at destination; "
                "buffering for query...",
            )
            list_mrn.append(str(int(row.MRN)))
            list_csn.append(str(int(row.PatientEncounterID)))

        if (
            len(list_mrn) > 0
            and len(list_csn) > 0
            and (len(list_mrn) == args.num_batch or (index + 1) == len(patients.index))
        ):
            if args.num_batch > 1:
                folder_idx = args.batch_folder_name
            else:
                folder_idx = os.path.join(str(list_mrn[0]), str(list_csn[0]))
            try:
                _run_queries(
                    edw=edw,
                    query_fpaths=query_fpaths,
                    list_mrn=list_mrn,
                    list_csn=list_csn,
                    batch_size=args.num_batch,
                    folder_idx=folder_idx,
                    save_dir=args.output_folder,
                )
            except Exception as e:
                print(f"_run_queries failed due to error: {e}")
            list_mrn = []
            list_csn = []
            if args.num_batch > 1:
                _reorder_folders(args.output_folder, args.batch_folder_name)


def _reorder_folders(save_dir: str, directory: str):
    list_batch_df = pd.read_csv(os.path.join(save_dir, directory, "batch_list.csv"))
    mrn_list = list(list_batch_df["MRN"])
    csn_list = list(list_batch_df["CSN"])
    tables = next(os.walk(os.path.join(save_dir, directory)))[2]
    for mrn, csn in zip(mrn_list, csn_list):
        mrn = str(mrn)
        csn = str(csn)
        if not os.path.isdir(os.path.join(save_dir, mrn, csn)):
            os.makedirs(os.path.join(save_dir, mrn, csn))
        for table in tables:
            if table != "batch_list.csv":
                tab_df = pd.read_csv(os.path.join(save_dir, directory, table))
                try:
                    filt_tab_df = tab_df[tab_df["PatientEncounterID"] == int(csn)]
                except KeyError:
                    filt_tab_df = tab_df[tab_df["CSN"] == int(csn)]
                filt_tab_df.to_csv(os.path.join(save_dir, mrn, csn, table), index=False)
    shutil.rmtree(os.path.join(save_dir, directory))
    print("Batch of patients reorganized.")


def _obtain_remaining_patients():
    # Load EDW queries
    query_fpaths = glob.glob(f"{path_to_queries}/*.sql")
    query_fpaths = [
        query_fpath for query_fpath in query_fpaths if "adt" not in query_fpath
    ]

    # EDW CSNs
    if args.just_sample_csv_csns:
        fpath = args.sample_csv
    else:
        fpath = args.path_adt
    patients = pd.read_csv(fpath)[["MRN", "PatientEncounterID"]].drop_duplicates()

    # Existing CSNs
    _, csns = _get_mrns_csns_at_destination(
        args.output_folder,
        query_fpaths,
        patients,
    )

    # Get remaining patients
    remaining_patients = patients[~patients["PatientEncounterID"].isin(csns)]
    remaining_patients.to_csv(
        os.path.join(args.output_folder, "remaining_patients.csv"),
    )


def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(
        title="edw modes",
        description="Select one of the following modes: \n"
        "\t * obtain_cohort: obtain a list of MRNs and CSNs of interest. \n"
        "\t * pull_edw_data: obtain edw data of a list of MRNs and CSNs. \n",
        dest="mode",
    )
    io_parser = argparse.ArgumentParser(add_help=False)
    io_parser.add_argument(
        "--output_folder",
        type=str,
        default="/media/ml4c3/edw",
        help="Location to store output from EDW.",
    )
    pull_edw_data_parser = subparser.add_parser(
        name="pull_edw_data",
        parents=[io_parser],
    )
    pull_edw_data_parser.add_argument(
        "--first",
        type=int,
        default=0,
        help="First patient in ADT table to get data from.",
    )
    pull_edw_data_parser.add_argument(
        "--last",
        type=int,
        default=None,
        help="Last patient in ADT table to get data from.",
    )
    pull_edw_data_parser.add_argument(
        "--sample_csv",
        type=str,
        default=None,
        help="Path to .csv with list of MRNs and CSNs to get EDW data.",
    )
    pull_edw_data_parser.add_argument(
        "--just_sample_csv_csns",
        action="store_true",
        help="If this parameter is set together with --sample_csv, just CSNs from "
        "--sample_csv will be pulled.",
    )
    pull_edw_data_parser.add_argument(
        "--path_adt",
        type=str,
        default=None,
        help="Path to the ADT table. If it doesn't exist, it will be created using "
        "MRNs in sample_csv.",
    )
    pull_edw_data_parser.add_argument(
        "--num_batch",
        type=int,
        default=1,
        help="Number of patients to batch for each query.",
    )
    pull_edw_data_parser.add_argument(
        "--batch_folder_name",
        type=str,
        default="b",
        help="Name of the folder where the batched results will be saved temporaly.",
    )
    pull_edw_data_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If this parameter is set, EDW data in --output_folder will be "
        "overwriten.",
    )

    obtain_cohort_parser = subparser.add_parser(
        name="obtain_cohort",
        parents=[io_parser],
    )
    obtain_cohort_parser_exclusive = obtain_cohort_parser.add_mutually_exclusive_group(
        required=True,
    )
    obtain_cohort_parser_exclusive.add_argument(
        "--cohort_query",
        type=str,
        default=None,
        choices=["cabg-arrest-blake8", "cabg", "rr-and-codes"],
        help="Name of the SQL query to call in order to get a list of patients.",
    )
    obtain_cohort_parser_exclusive.add_argument(
        "--department",
        type=str,
        default=None,
        help="Department name to get a list of patients.",
    )
    obtain_cohort_parser_exclusive.add_argument(
        "--sample_csv",
        type=str,
        default=None,
        help="Path to list of MRNS in order to get a list of MRNs and CSNs.",
    )
    obtain_cohort_parser.add_argument(
        "--do_not_compute_adt",
        action="store_true",
        help="Do not compute ADT table for the query obtained via --cohort_query, "
        "--department or --query_mrns.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    start_time = timer()
    args = parse_args()

    # Connect to EDW
    partners_username = input("Enter EDW username: ")
    pwd = getpass.getpass("Enter EDW password: ")
    edw = _connect_edw(partners_username, pwd)
    global_path = pathlib.Path(__file__).parent.absolute()

    # File I/O
    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)

    if args.mode == "obtain_cohort":
        path_to_queries = os.path.join(global_path, "queries-cohorts")
        if args.cohort_query:
            query_fpath = os.path.join(path_to_queries, f"{args.cohort_query}.sql")
            query_name = os.path.split(query_fpath)[-1][:-4]
            str_to_insert = None
        elif args.department:
            query_fpath = os.path.join(path_to_queries, "department.sql")
            query_name = args.department
            str_to_insert = DEPT_ID[args.department]
        elif args.sample_csv:
            query_fpath = os.path.join(path_to_queries, "mrns.sql")
            query_name = os.path.split(args.sample_csv)[-1]
            query_name = ".".join(query_name.split(".")[:-1]) + "_all_csns"
            mrns = pd.read_csv(args.sample_csv)["MRN"].unique()
            mrns = mrns[~np.isnan(mrns)]
            str_to_insert = str(list(mrns))[1:-1]
        fpath = os.path.join(args.output_folder, f"{query_name}.csv")
        _run_sql_query(
            edw=edw,
            query_fpath=query_fpath,
            str_to_insert=str_to_insert,
            fpath=fpath,
        )

        if not args.do_not_compute_adt:
            # Isolate name of the query
            path_to_queries = os.path.join(global_path, "queries-pipeline")
            query_fpath = os.path.join(path_to_queries, "adt.sql")
            adt_fpath = os.path.join(args.output_folder, f"adt-{query_name}.csv")
            mrns = pd.read_csv(fpath)["MRN"].unique()
            mrns = mrns[~np.isnan(mrns)]
            list_mrns = str(list(mrns))[1:-1]
            _run_sql_query(
                edw=edw,
                query_fpath=query_fpath,
                str_to_insert=list_mrns,
                fpath=adt_fpath,
            )
    elif args.mode == "pull_edw_data":
        path_to_queries = os.path.join(global_path, "queries-pipeline")
        _obtain_data(edw, path_to_queries, args)
        _obtain_remaining_patients()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    end_time = timer()
    elapsed_time = end_time - start_time
    print(f"\npulled EDW data in {elapsed_time:.2f} sec.")
