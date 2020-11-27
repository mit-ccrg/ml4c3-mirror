#!/usr/bin/env python

# Imports: standard library
import os
import getpass
import argparse
from timeit import default_timer as timer

# Imports: third party
import pandas as pd
import pyodbc


def load_query(query_file: str) -> str:
    with open(query_file, "r") as f:
        # Get SQL query from file as str
        sql_str = f.read()

    # Remove leading and trailing whitespaces
    sql_str = sql_str.strip()

    # Remove carriage return characters
    sql_str = sql_str.replace("\r", "")
    return sql_str


def run_sql_query(
    db: pyodbc.Connection,
    result_file: str,
    query_file: str,
    chunksize: int = 1000000,
):
    sql_str = load_query(query_file)
    print(f"\tLoaded query from {query_file}")

    # Execute query in chunks and append to output csv
    print(f"\tExecuting query:\n{sql_str}")
    count = 0
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    for chunk in pd.read_sql_query(sql_str, db, chunksize=chunksize):
        chunk.to_csv(result_file, index=False, mode="a", header=count == 0)
        count += len(chunk)
        print(f"\tSaved {count} rows")
    print(f"\tSaved query results to {result_file}")


def connect_edw(username: str, password: str) -> pyodbc.Connection:
    host = "phsedw.partners.org,1433"
    user = f"PARTNERS\\{username}"

    try:
        db = pyodbc.connect(
            driver="FreeTDS",
            TDS_version="8.0",
            host=host,
            user=user,
            pwd=password,
        )
        print(f"Connected to {host} using login {user}")
    except:
        raise Exception(f"Failed to connect to {host} using login {user}")
    return db


def run(args):
    # Connect to EDW
    username = input("Enter EDW username: ")
    password = getpass.getpass("Enter EDW password: ")
    db = connect_edw(username, password)

    run_sql_query(db, args.result_file, args.query_file)


if __name__ == "__main__":
    start_time = timer()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_file",
        required=True,
        help="Path to .csv output file",
    )
    parser.add_argument(
        "--query_file",
        required=True,
        help="Path to .sql query file",
    )
    args = parser.parse_args()

    run(args)

    end_time = timer()
    elapsed_time = end_time - start_time
    print(f"\nPulled EDW data in {elapsed_time:.2f} seconds")
