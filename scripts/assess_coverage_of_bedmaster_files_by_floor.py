# Imports: standard library
import os
import time
import argparse
from typing import Any, Set, Dict, List
from datetime import datetime

# Imports: third party
import h5py
import pandas as pd

# Imports: first party
from definitions.icu import MAPPING_DEPARTMENTS
from ingest.icu.match_patient_bedmaster import PatientBedmasterMatcher

# pylint: disable=redefined-outer-name, possibly-unused-variable, too-many-statements

POSSIBLE_DEPARTMENTS = {
    "blake8": "MGH BLAKE 8 CARD SICU",
    "ellison8": "MGH ELLISON 8 CARDSURG",
    "ellison9": "MGH ELLISON 9 MED\\CCU",
    "ellison10": "MGH ELLISON 10 STP DWN",
    "ellison11": "MGH ELLISON11 CARD\\INT",
}


class AssessBedmasterCoverage:
    """
    Class to assess bedmaster files conversion coverage.
    """

    @staticmethod
    def count_department_coverage(
        tensors: str,
        path_bedmaster: str,
        path_adt: str,
        path_xref: str,
        path_coverage_statistics: str,
        desired_department: str,
    ):
        """
        Counts MRNs in ADT table with 1+ associated Bedmaster file.

        :param tensors: <str> Directory with .hd5 files.
        :param path_bedmaster: <str> Directory with Bedmaster .mat files.
        :param path_adt: <str> Path to ADT table.
        :param path_xref: <str> Path to xref table.
        :param path_coverage_statistics: <str> Path to save the resulting
               coverage-$department.csv file.
        :param desired_department: <str> Desired department to assess coverage.
        """

        bedmaster = pd.read_csv(path_xref)
        bedmaster = bedmaster[bedmaster["Department"] == desired_department]

        data: Dict[str, Dict[str, Any]] = {
            "mrn": {},
            "csn": {},
            "bedmaster_file": {},
            "unique_mrn": {},
            "size": {},
            "a_size": {},
        }
        bedmaster_files: Dict[str, Set[str]] = {}

        # Total number of Bedmaster files
        bedmaster_files["bedmaster"] = set(bedmaster["fileID"])
        bedmaster_un_first = datetime.fromtimestamp(
            int(bedmaster["transferIn"].min()),
        ).strftime("%Y-%m-%d %H:%M:%S")
        bedmaster_un_last = datetime.fromtimestamp(
            int(bedmaster["transferIn"].max()),
        ).strftime("%Y-%m-%d %H:%M:%S")

        # Take just crossreferenced files and get time of first and last Bedmaster file
        bedmaster = bedmaster.dropna(subset=["MRN", "PatientEncounterID"])
        bedmaster_first = datetime.fromtimestamp(
            int(bedmaster["transferIn"].min()),
        ).strftime("%Y-%m-%d %H:%M:%S")
        bedmaster_last = datetime.fromtimestamp(
            int(bedmaster["transferIn"].max()),
        ).strftime("%Y-%m-%d %H:%M:%S")

        # Read adt and filter
        adt = pd.read_csv(path_adt)
        adt = adt[adt["DepartmentDSC"] == desired_department]
        adt_first = adt["TransferInDTS"].min()[:-8]
        adt_last = adt["TransferInDTS"].max()[:-8]
        adt_before = adt[adt["TransferInDTS"] < bedmaster_first]
        adt_after = adt[adt["TransferInDTS"] > bedmaster_last]
        adt_filt = adt[adt["TransferInDTS"] >= bedmaster_first]
        adt_filt = adt_filt[adt_filt["TransferInDTS"] <= bedmaster_last]

        csns_out = set(adt_before["PatientEncounterID"]).union(
            set(adt_after["PatientEncounterID"]),
        )
        bedmaster_u = bedmaster[~bedmaster["PatientEncounterID"].isin(csns_out)]
        adt_before_u = adt_before[
            ~adt_before["PatientEncounterID"].isin(
                set(adt_filt["PatientEncounterID"]),
            )
        ]
        adt_after_u = adt_after[
            ~adt_after["PatientEncounterID"].isin(
                set(adt_filt["PatientEncounterID"]),
            )
        ]
        adt_filt_u = adt_filt[~adt_filt["PatientEncounterID"].isin(csns_out)]

        hd5_dic: Dict[str, List[int]] = {"MRN": [], "CSN": []}
        for _, row in adt.iterrows():
            try:
                f = h5py.File(os.path.join(tensors, f"{row.MRN}.hd5"), "r")
                if str(row.PatientEncounterID) in f["edw"]:
                    hd5_dic["MRN"].append(row.MRN)
                    hd5_dic["CSN"].append(row.PatientEncounterID)
            except OSError:
                continue

        tables = [
            "adt",
            "adt_before",
            "adt_after",
            "adt_filt",
            "bedmaster",
            "adt_before_u",
            "adt_after_u",
            "adt_filt_u",
            "bedmaster_u",
        ]

        # Compute set of MRNs and CSNs for each table
        for name in tables:
            table = locals()[name]
            data["mrn"][name] = set(table["MRN"])
            data["csn"][name] = set(table["PatientEncounterID"])
            data["mrn"][f"hd5_{name}"] = set(table["MRN"]).intersection(
                set(hd5_dic["MRN"]),
            )
            data["csn"][f"hd5_{name}"] = set(table["PatientEncounterID"]).intersection(
                set(hd5_dic["CSN"]),
            )

        for name in tables:
            if name in ["adt", "bedmaster"]:
                data["unique_mrn"][name] = set(table["MRN"])
            if name.endswith("_u"):
                data["unique_mrn"][name] = data["mrn"][name].intersection(
                    data["unique_mrn"][name.replace("_u", "")],
                )
            elif name == "adt_before":
                data["unique_mrn"][name] = (
                    data["mrn"][name]
                    .difference(
                        data["mrn"]["adt_after"],
                    )
                    .difference(data["mrn"]["adt_filt"])
                )
            elif name == "adt_after":
                data["unique_mrn"][name] = (
                    data["mrn"][name]
                    .difference(
                        data["mrn"]["adt_before"],
                    )
                    .difference(data["mrn"]["adt_filt"])
                )
            elif name == "adt_filt":
                data["unique_mrn"][name] = (
                    data["mrn"][name]
                    .difference(
                        data["mrn"]["adt_after"],
                    )
                    .difference(data["mrn"]["adt_before"])
                )
            data["unique_mrn"][f"hd5_{name}"] = data["unique_mrn"][name].intersection(
                set(hd5_dic["MRN"]),
            )

        # Number of bedmaster files
        bedmaster_files["adt"] = set(bedmaster["fileID"])
        bedmaster_files["adt_filt_u"] = set(bedmaster_u["fileID"])

        # Reorganize results
        results: Dict[str, List[Any]] = {}
        for key in data["mrn"]:
            results[key] = [
                len(data["mrn"][key]),
                len(data["csn"][key]),
                len(data["unique_mrn"][key]),
            ]
        for key in bedmaster_files:
            results[key].append(len(bedmaster_files[key]))
            size = 0.0
            for path in bedmaster_files[key]:
                for department in MAPPING_DEPARTMENTS[desired_department]:
                    try:
                        size += os.path.getsize(
                            os.path.join(path_bedmaster, department, f"{path}.mat"),
                        )
                    except FileNotFoundError:
                        continue
            a_size = round(size / len(bedmaster_files[key]), 2)
            size = round(size / 1e6, 2)
            results[key].extend([size, a_size])
        for key in ["bedmaster", "adt"]:
            results[key].extend([locals()[f"{key}_first"], locals()[f"{key}_last"]])

        rows = [
            "ADT",
            "hd5",
            "ADT (before Bedmaster time window)",
            "hd5 (before Bedmaster time window)",
            "ADT (after Bedmaster time window)",
            "hd5 (after Bedmaster time window)",
            "ADT (Bedmaster time window)",
            "hd5 (Bedmaster time window)",
            "Bedmaster",
            "hd5 with Bedmaster",
            "ADT (before strict Bedmaster time window)",
            "hd5 (before strict Bedmaster time window)",
            "ADT (after strict Bedmaster time window)",
            "hd5 (after strict Bedmaster time window)",
            "ADT (strict Bedmaster time window)",
            "hd5 (strict Bedmaster time window)",
            "Bedmaster (strict Bedmaster time window)",
            "hd5 with Bedmaster (strict Bedmaster time window)",
        ]
        columns = [
            "MRNs",
            "CSNs",
            "Unique MRNs",
            "Bedmaster Files",
            "Total Bedmaster files size (MB)",
            "Average Bedmaster files size (MB)",
            "First",
            "Last",
        ]
        df = pd.DataFrame.from_dict(results, orient="index", columns=columns)
        new_index = pd.Index(rows, name=desired_department)
        df.index = new_index
        df.to_csv(path_coverage_statistics)
        print(f"Saved {path_coverage_statistics}.")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./results",
        help="Directory where the results are saved.",
    )
    parser.add_argument(
        "--path_bedmaster",
        type=str,
        default="/media/lm4-bedmaster/",
        help="Directory containing Bedmaster .mat files.",
    )
    parser.add_argument(
        "--tensors",
        type=str,
        default="/media/ml4c3/hd5",
        help="Directory containing .hd5 files.",
    )
    parser.add_argument(
        "--departments",
        nargs="+",
        help="List of department names for which to process patient data.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace):
    print("\n")

    departments = {}
    for department in args.departments:
        departments[department] = POSSIBLE_DEPARTMENTS[department]

    path_adt = os.path.join(args.output_folder, "adt.csv")
    # If ADT and xref .csv files exist, use them
    if os.path.exists(path_adt):
        print(f"{path_adt} exists.")
    # If these files do not exist, create them by pulling from EDW
    else:
        print(f"{path_adt} does not exist; pulling from EDW...")
        os.system(
            f"./scripts/run.sh $PWD/ml4c3/recipes.py pull_adt \
                --departments {' '.join(departments.keys())} \
                --output_folder {args.output_folder} \
                --path_adt {path_adt}",
        )

    path_xref = os.path.join(args.output_folder, "xref.csv")
    # Match the bedmaster files in bedmaster_dir with adt_file
    if os.path.exists(path_xref):
        print(f"{path_xref} exists.")
    else:
        print(f"{path_xref} does not exist; generating...")
        matcher = PatientBedmasterMatcher(
            path_bedmaster=args.path_bedmaster,
            path_adt=path_adt,
            desired_departments=list(departments.values()),
        )
        matcher.match_files(path_xref, True)

    # Assess coverage for each department
    for department in departments:
        edw_department_name = departments[department]
        path_coverage_statistics = os.path.join(
            args.output_folder,
            f"coverage-{department}.csv",
        )

        start_time = time.time()
        assessor = AssessBedmasterCoverage()
        assessor.count_department_coverage(
            tensors=args.tensors,
            path_bedmaster=args.path_bedmaster,
            path_adt=path_adt,
            path_xref=path_xref,
            path_coverage_statistics=path_coverage_statistics,
            desired_department=edw_department_name,
        )
        elapsed_time = time.time() - start_time
        print(f"Assessing coverage of {department} took {elapsed_time:0f} sec")


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
