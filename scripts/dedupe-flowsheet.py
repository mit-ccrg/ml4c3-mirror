#!/usr/bin/env python3

# Imports: standard library
import os
import shutil
import argparse
from timeit import default_timer as timer
from multiprocessing import Pool, Value, cpu_count

# Imports: third party
import pandas as pd

csv = "flowsheet.csv"
count = Value("i", 0)
total = Value("i", 0)


def check_flowsheet_rows(depts):
    def good_row(row):
        try:
            matched_dept = depts.loc[row["RecordedDTS"]]
            if isinstance(matched_dept, pd.Series):
                # some transfer in/out times overlap, okay if for the same department
                matched_dept = matched_dept.drop_duplicates()[0]
                if isinstance(matched_dept, pd.Series):
                    print(
                        f"Check measurement MRN/CSN/Recorded {row['MRN']}/{row['PatientEncounterID']}/{row['RecordedDTS']}",
                    )
                    raise KeyError(
                        f"Measurement cannot be taken in more than 1 department",
                    )
            return row["DepartmentID"] == matched_dept
        except:
            # the measurement does not fall in a time window defined by the ADT table
            return False

    return good_row


def dedupe_flowsheet(mrn_csn, df):
    with count.get_lock():
        count.value += 1
        if count.value % 10 == 0:
            print(f"Tried {count.value} / {total.value}")
    mrn, csn = mrn_csn
    flowsheet_path = os.path.join(args.edw_source, str(mrn), str(csn), csv)
    cleaned_flowsheet_path = os.path.join(args.edw_cleaned, str(mrn), str(csn), csv)
    if not os.path.isfile(flowsheet_path):
        return 0

    shutil.copytree(
        src=os.path.dirname(flowsheet_path),
        dst=os.path.dirname(cleaned_flowsheet_path),
        dirs_exist_ok=True,
    )

    t_in = pd.to_datetime(df["TransferInDTS"])
    t_out = pd.to_datetime(df["TransferOutDTS"])
    depts = df["DepartmentID"].astype(int)
    try:
        depts.index = pd.IntervalIndex.from_arrays(t_in, t_out)
        depts = depts.reset_index().drop_duplicates().set_index("index")["DepartmentID"]
    except ValueError:
        return 0

    flowsheet = pd.read_csv(cleaned_flowsheet_path, low_memory=False)
    flowsheet["RecordedDTS"] = pd.to_datetime(flowsheet["RecordedDTS"])
    mask = flowsheet.apply(check_flowsheet_rows(depts), axis=1)
    flowsheet[mask].to_csv(cleaned_flowsheet_path, index=False)
    return 1


def main(args):
    start_time = timer()

    adt_grouped = pd.read_csv(args.adt, low_memory=False).groupby(
        ["MRN", "PatientEncounterID"],
    )
    total.value = len(adt_grouped)
    with Pool(processes=args.num_workers) as pool:
        cleaned = sum(pool.starmap(dedupe_flowsheet, adt_grouped))

    end_time = timer()
    elapsed_time = end_time - start_time
    print(f"Cleaned {cleaned} encounters in {elapsed_time:.2f} seconds")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adt",
        help="Path to ADT table.",
    )
    parser.add_argument(
        "--edw-source",
        help="Path to directory containing EDW flowsheet CSVs to deduplicate.",
    )
    parser.add_argument(
        "--edw-cleaned",
    )
    parser.add_argument(
        "--num_workers",
        default=(cpu_count() + 1) // 2,
        type=int,
        help="Number of worker processes to use if processing in parallel.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
