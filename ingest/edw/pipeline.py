# pylint: disable=redefined-outer-name, broad-except
# Imports: standard library
import os
import shutil
import getpass
import logging
import argparse
from typing import Set, List, Tuple, Optional

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


def _format_sql_query(
    query_path: str,
    strings_to_insert: List[str] = [],
) -> str:
    """
    Loads a sql query from a file and inserts parameter strings
    """
    # Get SQL query from file as string
    with open(query_path, "r") as f:
        sql = f.read()

    # Insert format string into {} in query string
    sql = sql.format(*strings_to_insert)

    # Remove leading and trailing whitespaces
    sql = sql.strip()

    # Remove carriage return characters
    sql = sql.replace("\r", "")

    return sql


def query_to_dataframe(
    edw: pyodbc.Connection,
    query_path: str,
    strings_to_insert: List[str] = [],
) -> pd.DataFrame:

    logging.info(f"Running query: {os.path.split(query_path)[-1]}")
    sql = _format_sql_query(query_path, strings_to_insert)
    df = pd.read_sql_query(sql, edw)
    return df


def _run_queries(
    edw: pyodbc.Connection,
    query_paths: list,
    list_mrn: list,
    list_csn: list,
    batch_size: int,
    folder_idx: str,
    save_dir: str,
):
    save_dir = os.path.join(save_dir, folder_idx)
    os.makedirs(save_dir, exist_ok=True)

    # Save tables with lists of batched mrns and csns if there's batching
    if batch_size > 1:
        batch_list = pd.DataFrame(
            np.array((list_mrn, list_csn)).transpose(),
            columns=["MRN", "CSN"],
        )
        batch_list.to_csv(os.path.join(save_dir, "batch_list.csv"), index=False)

    # Iterate through SQL queries
    list_csn_str = str(list_csn)[1:-1]
    for query_path in query_paths:
        query_df = query_to_dataframe(
            edw=edw,
            query_path=query_path,
            strings_to_insert=[list_csn_str],
        )
        query_name = os.path.splitext(os.path.basename(query_path))[0]
        result_path = os.path.join(save_dir, f"{query_name}.csv")
        query_df.to_csv(result_path, index=False)
        logging.info(f"Saved results to: {result_path}")


def _get_mrns_csns_at_destination(
    edw_dir: str,
    query_paths: List[str],
    patients: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Given a directory, obtain a list of all MRN and CSN directories. Assumes a
    directory structure:
    /parent_dir
    ├── 12345 (mrn)
    │   ├── 78982336 (csn)
    │   └── 78937589 (csn)
    ├── 12346 (mrn)
    │   └── 12552657 (csn)
    etc.
    """
    # Determine the list of expected .csv files in each CSN folder
    expected_files = [
        f"{os.path.basename(query_path)[:-4]}.csv" for query_path in query_paths
    ]

    # Get full path to MRN-CSN directories
    make_path = lambda row: os.path.join(edw_dir, row["MRN"], row["PatientEncounterID"])
    csn_dirs = patients.apply(make_path, axis=1).unique()

    # Initialize empty master lists to store MRNs and CSNs
    mrns = []
    csns = []

    # Iterate through MRN-CSN dir and update master lists
    for csn_dir in csn_dirs:
        try:
            csv_files = {
                csv_file
                for csv_file in os.listdir(csn_dir)
                if csv_file.endswith(".csv")
            }
            for expected_file in expected_files:
                if expected_file not in csv_files:
                    break
            else:
                # Remove the parent_dir prefix to just get '/mrn/csn'
                csn_dir = csn_dir.replace(edw_dir, "")
                _, mrn, csn = csn_dir.split("/")
                mrns.append(mrn)
                csns.append(csn)
        except FileNotFoundError:
            continue
    return mrns, csns


def connect_edw() -> pyodbc.Connection:
    """
    Connect to EDW databse using user: partners_username and password: pwd.
    """
    partners_username = input(
        "=======================================================================\n"
        "Enter EDW username: ",
    )
    pwd = getpass.getpass("Enter EDW password: ")

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
        logging.info(f"Connected to {host} using login {user}")
    except Exception as e:
        raise Exception(f"Failed to connect to {host} using login {user}") from e
    return edw


def _reorder_folders(save_dir: str, directory: str):
    list_batch_df = pd.read_csv(
        os.path.join(save_dir, directory, "batch_list.csv"),
        low_memory=False,
    )
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
                tab_df = pd.read_csv(
                    os.path.join(save_dir, directory, table),
                    low_memory=False,
                )
                try:
                    filt_tab_df = tab_df[tab_df["PatientEncounterID"] == int(csn)]
                except KeyError:
                    filt_tab_df = tab_df[tab_df["CSN"] == int(csn)]
                filt_tab_df.to_csv(os.path.join(save_dir, mrn, csn, table), index=False)
    shutil.rmtree(os.path.join(save_dir, directory))
    logging.info("Batch of patients reorganized.")


def _obtain_data(
    args: argparse.Namespace,
    edw: pyodbc.Connection,
    query_directory: str,
    adt: pd.DataFrame,
):
    if args.staging_dir is None:
        raise ValueError("Must provide staging directory.")
    batch_size = args.staging_batch_size or 1

    query_paths = [
        os.path.join(query_directory, query_path)
        for query_path in os.listdir(query_directory)
        if query_path.endswith(".sql")
        and "adt" not in query_path
        and "mrn" not in query_path
        and "department" not in query_path
    ]

    # Get unique patients and encounters, sort, and select data
    patients = adt[["MRN", "PatientEncounterID"]].drop_duplicates()
    patients = patients.sort_values(by=["MRN", "PatientEncounterID"], ascending=True)
    num_patients = len(patients)
    logging.info(
        f"Extracted {num_patients} unique MRN-PatientEncounterID "
        f"values from adt table.",
    )

    patients = patients[args.adt_start_index : args.adt_end_index]
    patients = patients.reset_index()
    num_patients = len(patients.index)
    logging.info(
        f"Reduced the number of unique MRN-PatientEncounterID values to {num_patients}.",
    )

    # Get MRNs and CSNs at the destination directory
    csns = set()
    if not args.overwrite:
        _, csns = _get_mrns_csns_at_destination(
            edw_dir=args.path_edw,
            query_paths=query_paths,
            patients=patients,
        )
        csns = set(csns)

    # Iterate through each MRN and CSN
    list_mrn = []
    list_csn = []
    for index, row in patients.iterrows():
        if row.PatientEncounterID in csns:
            logging.info(
                f"MRN: {row.MRN} ({index + 1}/{patients.shape[0]}), CSN: "
                f"{row.PatientEncounterID} already exists at destination; "
                "skipping.",
            )
        else:
            logging.info(
                f"MRN: {row.MRN} ({index + 1}/{patients.shape[0]}), CSN: "
                f"{row.PatientEncounterID} does not yet exist at destination; "
                "buffering for query...",
            )
            list_mrn.append(str(int(row.MRN)))
            list_csn.append(str(int(row.PatientEncounterID)))

        if (
            len(list_mrn) > 0
            and len(list_csn) > 0
            and (len(list_mrn) == batch_size or (index + 1) == len(patients.index))
        ):
            if batch_size > 1:
                folder_idx = args.staging_dir
            else:
                folder_idx = os.path.join(str(list_mrn[0]), str(list_csn[0]))

            _run_queries(
                edw=edw,
                query_paths=query_paths,
                list_mrn=list_mrn,
                list_csn=list_csn,
                batch_size=batch_size,
                folder_idx=folder_idx,
                save_dir=args.path_edw,
            )

            list_mrn = []
            list_csn = []
            if args.staging_batch_size > 1:
                _reorder_folders(args.path_edw, args.staging_dir)


def _mrns_csns_from_df(df: pd.DataFrame) -> Tuple[Set[str], Optional[Set[str]]]:
    if "MRN" not in df:
        raise ValueError("Must include MRN column in patient_csv.")
    mrns = set(df["MRN"].dropna().astype(int).astype(str))
    csns = None
    if "PatientEncounterID" in df:
        csns = set(df["PatientEncounterID"].dropna().astype(int).astype(str))
    return mrns, csns


def _format_adt_table(adt: pd.DataFrame) -> pd.DataFrame:
    """
    Given ADT dataframe, returns the ADT table where the MRN and PatientEncounterID
    columns are strings and the HospitalAdmitDTS column is datetime.
    """
    if (
        "MRN" not in adt
        or "PatientEncounterID" not in adt
        or "HospitalAdmitDTS" not in adt
    ):
        raise ValueError(
            "ADT table must contain MRN, PatientEncounterID, and HospitalAdmitDTS columns.",
        )
    adt["MRN"] = adt["MRN"].astype(int).astype(str)
    adt["PatientEncounterID"] = adt["PatientEncounterID"].astype(int).astype(str)
    adt["HospitalAdmitDTS"] = pd.to_datetime(adt["HospitalAdmitDTS"])
    return adt


def get_adt_table(
    args: argparse.Namespace,
    edw: pyodbc.Connection,
    query_directory: str,
) -> pd.DataFrame:
    """
    Get ADT table through one of the following methods:
    1. Provide ADT table
        i. Filter patients by MRN
       ii. Filter temporally
            a. By encounter (CSN)
            b. By hospital admission time (between a start and end date)
    2. Provide departments
        i. Filter patients by MRN
       ii. Filter temporally
            a. By encounter (CSN)
            b. By hospital admission time (between a start and end date)
    3. Provide MRNs
        i. Filter temporally
            a. By encounter (CSN)
            b. By hospital admission time (between a start and end date)
    4. Provide cohort query
        i. Filter temporally
            a. By encounter (CSN)
            b. By hospital admission time (between a start and end date)
    """
    if args.path_adt is None:
        raise ValueError("Must provide path to save or load ADT table.")

    csns = None

    # Load initial ADT table
    # 1. by using the path to an existing ADT table
    if os.path.isfile(args.path_adt):
        adt = pd.read_csv(args.path_adt, low_memory=False)
        adt = _format_adt_table(adt)

        # Restrict ADT table to MRNs found in patient_csv
        if args.patient_csv is not None:
            patient_df = pd.read_csv(args.patient_csv, low_memory=False)
            mrns, csns = _mrns_csns_from_df(df=patient_df)
            adt = adt[adt["MRN"].isin(mrns)]
            logging.info(f"Filtered existing ADT table to MRNs in {args.patient_csv}")

        logging.info(f"Loaded existing ADT table: {args.path_adt}")

    # 2. by department
    elif args.departments is not None:
        # Run query to get patients by department
        dept_query = os.path.join(query_directory, "department.sql")
        dept_string = ", ".join([DEPT_ID[dept] for dept in args.departments])
        dept_df = query_to_dataframe(
            edw=edw,
            query_path=dept_query,
            strings_to_insert=[dept_string],
        )
        dept_df["MRN"] = dept_df["MRN"].astype(int).astype(str)
        dept_df["PatientEncounterID"] = (
            dept_df["PatientEncounterID"].astype(int).astype(str)
        )

        # Restrict to MRNs/CSNs found in patient_csv
        if args.patient_csv is not None:
            patient_df = pd.read_csv(args.patient_csv, low_memory=False)
            mrns, csns = _mrns_csns_from_df(df=patient_df)
            dept_df = dept_df[dept_df["MRN"].isin(mrns)]
            logging.info(f"Filtered departments to MRNs in {args.patient_csv}")
            if csns is not None:
                dept_df = dept_df[dept_df["PatientEncounterID"].isin(csns)]
                logging.info(f"Filtered departments to CSNs in {args.patient_csv}")

        # Process MRNs/CSNs from department query
        mrns, dept_csns = _mrns_csns_from_df(dept_df)
        mrns_string = ", ".join(mrns)
        if len(mrns) == 0:
            raise ValueError("No patients found by department query.")

        # Save results of department query
        result_name = "-".join(args.departments)
        result_path = os.path.join(args.output_folder, args.id, f"{result_name}.csv")
        dept_df.to_csv(result_path, index=False)
        logging.info(f"Saved cohort by departments to {result_path}")

        # Run query for ADT table
        adt_query = os.path.join(query_directory, "adt.sql")
        adt = query_to_dataframe(
            edw=edw,
            query_path=adt_query,
            strings_to_insert=[mrns_string],
        )
        adt = _format_adt_table(adt)
        adt = adt[adt["PatientEncounterID"].isin(dept_csns)]
        logging.info(f"Pulled ADT table by departments: {', '.join(args.departments)}")

    # 3. by using patient_csv containing MRNs and optionally CSNs
    elif args.patient_csv is not None:
        # Load data from patient_csv
        patient_df = pd.read_csv(args.patient_csv, low_memory=False)
        mrns, csns = _mrns_csns_from_df(df=patient_df)
        mrns_string = ", ".join(mrns)

        # Run query for ADT table
        adt_query = os.path.join(query_directory, "adt.sql")
        adt = query_to_dataframe(
            edw=edw,
            query_path=adt_query,
            strings_to_insert=[mrns_string],
        )
        adt = _format_adt_table(adt)
        logging.info(f"Pulled ADT table by patient_csv: {args.patient_csv}")

    # 4. by using a query that gets MRNs and optionally CSNs
    elif args.cohort_query is not None:
        # Run query to obtain cohort
        cohort_df = query_to_dataframe(
            edw=edw,
            query_path=args.cohort_query,
        )
        mrns, csns = _mrns_csns_from_df(df=cohort_df)
        mrns_string = ", ".join(mrns)

        # Save results of cohort query
        cohort_name = os.path.splitext(os.path.basename(args.cohort_query))[0]
        cohort_path = os.path.join(args.output_folder, args.id, f"{cohort_name}.csv")
        cohort_df.to_csv(cohort_path, index=False)
        logging.info(f"Saved cohort by query to {cohort_path}")

        # Run query for ADT table
        adt_query = os.path.join(query_directory, "adt.sql")
        adt = query_to_dataframe(
            edw=edw,
            query_path=adt_query,
            strings_to_insert=[mrns_string],
        )
        adt = _format_adt_table(adt)
        logging.info(f"Pulled ADT table by cohort_query: {args.cohort_query}")

    else:
        raise ValueError(
            "No method to get ADT table. Specify a valid path to an existing ADT "
            "table, a patient_csv file containing MRNs, or an EDW query that gets MRNs.",
        )

    # Time filter in order of priority:
    # 1. filter by CSN
    if csns is not None:
        adt = adt[adt["PatientEncounterID"].isin(csns)]
        logging.info(f"Filtered ADT table by CSNs.")
    # 2. filter by time
    elif args.start_time is not None or args.end_time is not None:
        if args.start_time is None:
            args.start_time = "1800-01-01"
        if args.end_time is None:
            args.end_time = "2200-01-01"
        start = pd.to_datetime(args.start_time, format="%Y-%m-%d")
        end = pd.to_datetime(args.end_time, format="%Y-%m-%d")
        after_start = adt["HospitalAdmitDTS"] > start
        before_end = adt["HospitalAdmitDTS"] < end
        adt = adt[after_start & before_end]
        logging.info(f"Filtered ADT table by hospital admission time.")
    # 3. do not filter

    # Check that ADT table is not empty
    if len(adt) == 0:
        raise ValueError("ADT table is empty.")

    # Save ADT table
    if not os.path.isfile(args.path_adt):
        os.makedirs(os.path.dirname(args.path_adt), exist_ok=True)
        adt.to_csv(args.path_adt, index=False)
        logging.info(f"Saved ADT table to {args.path_adt}")

    return adt


def pull_edw_data(args: argparse.Namespace, only_adt: bool = False):
    edw = connect_edw()
    query_directory = os.path.join(os.path.dirname(__file__), "queries-pipeline")
    adt = get_adt_table(
        args=args,
        edw=edw,
        query_directory=query_directory,
    )
    if only_adt:
        return
    _obtain_data(
        args=args,
        edw=edw,
        query_directory=query_directory,
        adt=adt,
    )
    logging.info(f"Saved EDW data to {args.path_edw}")
