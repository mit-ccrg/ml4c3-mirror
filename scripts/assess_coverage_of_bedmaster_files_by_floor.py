# Imports: standard library
import os
import time
import argparse
from typing import Any, Dict, List
from datetime import datetime

# Imports: third party
import pandas as pd

# Imports: first party
from ml4c3.definitions.icu import MAPPING_DEPARTMENTS
from ml4c3.ingest.icu.match_patient_bedmaster import PatientBedmasterMatcher

# pylint: disable=redefined-outer-name, possibly-unused-variable, too-many-statements


class AssessBedmasterCoverage:
    """
    Class to assess bedmaster files conversion coverage.
    """

    @staticmethod
    def count_department_coverage(
        path_bedmaster: str,
        path_adt: str,
        path_xref: str,
        path_coverage_statistics: str,
        desired_department: str,
    ):
        """
        Counts MRNs in ADT table with 1+ associated Bedmaster file.

        :param path_bedmaster: <str> Directory with Bedmaster .mat files.
        :param path_adt: <str> Path to ADT table.
        :param path_xref: <str> Path to xref table.
        :param path_coverage_statistics: <str> Path to save the resulting
               coverage-$department.csv file.
        :param desired_department: <str> Desired department to assess coverage.
        """
        # Match the bedmaster files in bedmaster_dir with adt_file
        if os.path.exists(path_xref):
            print("{path_xref} exists")
        else:
            print(f"{path_xref} does not exist; generating")
            matcher = PatientBedmasterMatcher(
                path_bedmaster=path_bedmaster,
                path_adt=path_adt,
                desired_departments=[desired_department],
            )
            matcher.match_files(path_xref, True)

        bedmaster_df = pd.read_csv(path_xref)

        data: Dict[str, Dict[str, Any]] = {
            "mrn": {},
            "csn": {},
            "bedmaster_file": {},
            "unique_mrn": {},
            "size": {},
            "a_size": {},
        }

        # Total number of Bedmaster files
        data["bedmaster_file"]["bedmaster"] = set(bedmaster_df["fileID"])
        data["bedmaster_file"]["bedmaster_u"] = set(bedmaster_df["fileID"])
        bedmaster_un_first = datetime.fromtimestamp(
            int(bedmaster_df["transferIn"].min()),
        ).strftime("%Y-%m-%d %H:%M:%S")
        bedmaster_un_last = datetime.fromtimestamp(
            int(bedmaster_df["transferIn"].max()),
        ).strftime("%Y-%m-%d %H:%M:%S")
        # Take just crossreferenced files and get time of first and last Bedmaster file
        bedmaster_df = bedmaster_df.dropna(subset=["MRN", "PatientEncounterID"])
        bedmaster_first = datetime.fromtimestamp(
            int(bedmaster_df["transferIn"].min()),
        ).strftime("%Y-%m-%d %H:%M:%S")
        bedmaster_last = datetime.fromtimestamp(
            int(bedmaster_df["transferIn"].max()),
        ).strftime("%Y-%m-%d %H:%M:%S")

        # Read adt and filter before first and after last Bedmaster files and in between
        adt_df = pd.read_csv(path_adt)
        # adt_df = adt_df[adt_df["DepartmentDSC"] == des_dept]
        adt_first = adt_df["TransferInDTS"].min()[:-8]
        adt_last = adt_df["TransferInDTS"].max()[:-8]
        adt_df_before = adt_df[adt_df["TransferInDTS"] < bedmaster_first]
        adt_df_after = adt_df[adt_df["TransferInDTS"] > bedmaster_last]
        adt_df_filt = adt_df[adt_df["TransferInDTS"] >= bedmaster_first]
        adt_df_filt = adt_df_filt[adt_df_filt["TransferInDTS"] <= bedmaster_last]

        tables = [
            ("adt", adt_df),
            ("adt_before", adt_df_before),
            ("adt_after", adt_df_after),
            ("adt_filt", adt_df_filt),
            ("bedmaster", bedmaster_df),
        ]

        # Compute set of MRNs and CSNs for each table
        for name, table in tables:
            data["mrn"][name] = set(table["MRN"])
            data["csn"][name] = set(table["PatientEncounterID"])

        # Compute DF for strict time window (CSNs including limit time are not included)
        bedmaster_df_u = bedmaster_df[
            ~bedmaster_df["PatientEncounterID"].isin(
                data["csn"]["adt_before"].union(data["csn"]["adt_after"]),
            )
        ]
        adt_df_before_u = adt_df_before[
            ~adt_df_before["PatientEncounterID"].isin(data["csn"]["adt_filt"])
        ]
        adt_df_after_u = adt_df_after[
            ~adt_df_after["PatientEncounterID"].isin(data["csn"]["adt_filt"])
        ]
        adt_df_filt_u = adt_df_filt[
            ~adt_df_filt["PatientEncounterID"].isin(
                data["csn"]["adt_before"].union(data["csn"]["adt_after"]),
            )
        ]
        strict_tables = [
            ("adt_before_u", adt_df_before_u),
            ("adt_after_u", adt_df_after_u),
            ("adt_u", adt_df_filt_u),
            ("bedmaster_u", bedmaster_df_u),
        ]

        # Compute set of MRNs and CSNs for each new table
        for name, table in strict_tables:
            data["mrn"][name] = set(table["MRN"])
            data["csn"][name] = set(table["PatientEncounterID"])

        # For each type of ADT table compute the number of unique MRNs
        adt_tables = [
            "adt_before",
            "adt_after",
            "adt_before_u",
            "adt_after_u",
        ]
        for name in adt_tables:
            key = "_filt" if name.endswith("_u") else "_u"
            data["unique_mrn"][name] = data["mrn"][name].difference(
                data["mrn"][f"adt{key}"],
            )
            if name.startswith("adt_after"):
                data["unique_mrn"][name] = data["unique_mrn"][name].difference(
                    data["unique_mrn"]["adt_before"],
                )

        # Number of bedmaster files
        data["bedmaster_file"]["adt"] = set(bedmaster_df["fileID"])
        data["bedmaster_file"]["adt_filt"] = set(bedmaster_df["fileID"])
        data["bedmaster_file"]["adt_u"] = set(bedmaster_df_u["fileID"])

        # Bedmaster files size
        for key in ["bedmaster", "adt", "adt_filt", "adt_u"]:
            data["size"][key] = 0
            for path in data["bedmaster_file"][key]:
                for department in MAPPING_DEPARTMENTS[desired_department]:
                    try:
                        data["size"][key] += os.path.getsize(
                            os.path.join(path_bedmaster, department, f"{path}.mat"),
                        )
                    except FileNotFoundError:
                        continue
            data["size"][key] = round(data["size"][key] / 1e6, 2)
            data["a_size"][key] = round(
                data["size"][key] / len(data["bedmaster_file"][key]),
                2,
            )

        # Reorganize results
        results: Dict[str, List[Any]] = {}
        for key in data["mrn"]:
            results[key] = [len(data["mrn"][key]), len(data["csn"][key])]
        for key in data["bedmaster_file"]:
            results[key].append(len(data["bedmaster_file"][key]))
        for key in ["", "_filt", "_u"]:
            key2 = "" if key == "_filt" else key
            key = "" if key2 == "_u" else key
            mrnsp = len(data["mrn"][f"bedmaster{key2}"]) / len(data["mrn"][f"adt{key}"])
            csnsp = len(data["csn"][f"bedmaster{key2}"]) / len(data["csn"][f"adt{key}"])
            bedmasterp = len(data["bedmaster_file"][f"adt{key}"]) / len(
                data["bedmaster_file"][f"bedmaster{key2}"],
            )
            mrns_un = data["mrn"][f"adt{key}"].difference(
                data["mrn"][f"bedmaster{key2}"],
            )
            csns_un = data["csn"][f"adt{key}"].difference(
                data["csn"][f"bedmaster{key2}"],
            )
            bedmaster_files_un = data["bedmaster_file"][f"bedmaster{key2}"].difference(
                data["bedmaster_file"][f"adt{key}"],
            )
            key = key2 if key2 == "_u" else key
            results["Remaining" + key] = [
                len(mrns_un),
                len(csns_un),
                len(bedmaster_files_un),
            ]
            results["%" + key] = [
                round(mrnsp * 100, 3),
                round(csnsp * 100, 3),
                round(bedmasterp * 100, 3),
            ]
        for key in data["unique_mrn"]:
            results[key].extend(
                [None, None, None, None, None, len(data["unique_mrn"][key])],
            )
        for key in data["size"]:
            results[key].extend([data["size"][key], data["a_size"][key]])
        for key in ["bedmaster", "adt"]:
            results[key].extend([locals()[f"{key}_first"], locals()[f"{key}_last"]])
        results["Remaining"].extend([None, None, bedmaster_un_first, bedmaster_un_last])
        rows = [
            "ADT",
            "ADT (before Bedmaster time window)",
            "ADT (after Bedmaster time window)",
            "ADT (Bedmaster time window)",
            "Bedmaster",
            "ADT (before strict Bedmaster time window)",
            "ADT (after strict Bedmaster time window)",
            "ADT (strict Bedmaster time window)",
            "Bedmaster (strict Bedmaster time window)",
            "Remaining",
            "%",
            "Remaining (Bedmaster time window)",
            "% (Bedmaster time window)",
            "Remaining (strict Bedmaster time window)",
            "% (strict Bedmaster time window)",
        ]
        columns = [
            "MRNs",
            "CSNs",
            "Bedmaster Files",
            "Total Bedmaster files size (MB)",
            "Average Bedmaster files size (MB)",
            "First",
            "Last",
            "Unique MRNs",
        ]
        df = pd.DataFrame.from_dict(results, orient="index", columns=columns)
        new_index = pd.Index(rows, name=desired_department)
        df.index = new_index
        df.to_csv(path_coverage_statistics)
        order = [2, 12, 13, 5, 1, 14, 15, 8, 9, 3, 4, 6, 7, 10, 11]
        df.insert(1, "Order", order)
        df = df.sort_values(by=["Order"])
        df = df.drop(columns=["Order"])
        df.to_csv(path_coverage_statistics)
        print(f"Saved {path_coverage_statistics}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_pipeline_script",
        type=str,
        default="/$HOME/edw/icu/pipeline.py",
        help="Path to EDW pipeline script.",
    )
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
        help="Directory where the results are saved.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace):
    print("\n")

    departments = {
        "blake8": "MGH BLAKE 8 CARD SICU",
        "ellison8": "MGH ELLISON 8 CARDSURG",
        "ellison9": "MGH ELLISON 9 MED\\CCU",
        "ellison10": "MGH ELLISON 10 STP DWN",
        "ellison11": "MGH ELLISON11 CARD\\INT",
    }

    for department in departments:
        path_adt = os.path.join(args.output_folder, f"adt-{department}.csv")
        path_xref = os.path.join(args.output_folder, f"xref-{department}.csv")

        # If ADT and xref .csv files exist, use them
        if os.path.exists(path_adt):
            print(f"{path_adt} exists")

        # If these files do not exist, create them by pulling from EDW
        else:
            print(f"{path_adt} does not exist; pulling from EDW")
            os.system(
                f"python {args.path_to_pipeline_script} obtain_cohort \
                --department {department} \
                --output_folder {args.output_folder}",
            )

        edw_department_name = departments[department]
        path_coverage_statistics = os.path.join(
            args.output_folder,
            f"coverage-{department}.csv",
        )

        start_time = time.time()
        assessor = AssessBedmasterCoverage()
        assessor.count_department_coverage(
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
