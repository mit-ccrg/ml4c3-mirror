#!/usr/bin/env python3

# Imports: standard library
import os
import argparse
from timeit import default_timer as timer
from multiprocessing import Pool, Value, cpu_count

# Imports: third party
import pandas as pd

mapping = {}
failed = []
count = Value("i", 0)
total = Value("i", 0)


def check_patient_rows(mrn):
    def good_row(row):
        try:
            _mapping = mapping[mrn].loc[row["RecordedDTS"]]
            x = row["DepartmentID"] == _mapping["DepartmentID"]
            y = row["PatientEncounterID"] == _mapping["PatientEncounterID"]
            return x & y
        except:
            # the measurement does not fall in a time window defined by the ADT table
            return False

    return good_row


def dedupe_flowsheet(flowsheet_path_mrn):
    with count.get_lock():
        count.value += 1
        if count.value % 10 == 0:
            print(f"Finished {count.value} / {total.value}")
    flowsheet_path, mrn = flowsheet_path_mrn
    flowsheet = pd.read_csv(flowsheet_path)
    flowsheet["RecordedDTS"] = pd.to_datetime(flowsheet["RecordedDTS"])
    mask = flowsheet.apply(check_patient_rows(mrn), axis=1)
    flowsheet[mask].to_csv(flowsheet_path, index=False)


def main(args):
    start_time = timer()
    flowsheet_paths = []
    for mrn in os.listdir(args.edw):
        mrn_dir = os.path.join(args.edw, mrn)
        for csn in os.listdir(mrn_dir):
            flowsheet_path = os.path.join(mrn_dir, csn, "flowsheet.csv")
            flowsheet_paths.append((flowsheet_path, int(mrn)))

    print(f"Found {len(flowsheet_paths)} flowsheets at {args.edw}")

    adt = pd.read_csv(args.adt)
    for mrn, group in adt.groupby("MRN"):
        t_in = pd.to_datetime(group["TransferInDTS"])
        t_out = pd.to_datetime(group["TransferOutDTS"])
        csn_dept = group[["PatientEncounterID", "DepartmentID"]]
        try:
            csn_dept.index = pd.IntervalIndex.from_arrays(t_in, t_out)
            csn_dept = (
                csn_dept.reset_index()
                .drop_duplicates()
                .set_index("index")[["PatientEncounterID", "DepartmentID"]]
            )
            mapping[mrn] = csn_dept
        except ValueError as e:
            failed.append(mrn)

    print(f"ADT for the following patients are malformed, skipping: {failed}")

    total.value = len(flowsheet_paths)
    with Pool(processes=args.num_workers) as pool:
        pool.map(dedupe_flowsheet, flowsheet_paths)

    end_time = timer()
    elapsed_time = end_time - start_time
    print(f"Cleaned {len(flowsheet_paths)} flowsheets in {elapsed_time:.2f} seconds")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adt",
        help="Path to ADT table.",
    )
    parser.add_argument(
        "--edw",
        help="Path to directory containing EDW flowsheet CSVs to deduplicate.",
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
