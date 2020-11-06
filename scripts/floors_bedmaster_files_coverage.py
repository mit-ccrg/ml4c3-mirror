# Imports: standard library
import os
from typing import Any, Dict, List
from datetime import datetime

# Imports: third party
import pandas as pd

# Imports: first party
from ml4c3.definitions.icu import MAPPING_DEPARTMENTS
from ml4c3.ingest.icu.matchers.match_patient_bedmaster import PatientBedmasterMatcher

# pylint: disable=possibly-unused-variable, too-many-statements


class AssessBedmasterCoverage:
    """
    Class to assess bedmaster files conversion coverage.
    """

    @staticmethod
    def department_coverage(
        bedmaster_dir: str,
        edw_dir: str,
        adt_file: str,
        des_dept: str,
        output_dir: str = "./results",
    ):
        """
        It checks how many MRN of the ADT table has at least one Bedmaster file
        associated.

        :param bedmaster_dir: <str> Directory with Bedmaster .mat files.
        :param edw_dir: <str> Directory with the adt_file.
        :param adt_file: <str> File containing the admission, transfer and
                                 discharge from patients (.csv).
        :param des_dept: <str> Desired department to assess coverage.
        :param output_dir: <str> Directory where the output file will be saved.
        """
        # Match the bedmaster files in bedmaster_dir with adt_file
        matcher = PatientBedmasterMatcher(
            flag_lm4=False,
            bedmaster_dir=bedmaster_dir,
            edw_dir=edw_dir,
            adt_file=adt_file,
            des_depts=[des_dept],
        )
        matcher.match_files(
            os.path.join(output_dir, f"match_{des_dept.replace(' ', '')}.csv"),
            True,
        )
        bedmaster_df = pd.DataFrame(matcher.table_dic)

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
        adt_df = pd.read_csv(os.path.join(edw_dir, adt_file))
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
                data["size"][key] += os.path.getsize(
                    os.path.join(bedmaster_dir, f"{path}.mat"),
                )
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
        data_frame = pd.DataFrame.from_dict(results, orient="index", columns=columns)
        new_index = pd.Index(rows, name=des_dept)
        data_frame.index = new_index
        order = [2, 12, 13, 5, 1, 14, 15, 8, 9, 3, 4, 6, 7, 10, 11]
        data_frame.insert(1, "Order", order)
        data_frame = data_frame.sort_values(by=["Order"])
        data_frame = data_frame.drop(columns=["Order"])
        output_path = os.path.join(output_dir, f"bedmaster_files_coverage_{adt_file}")
        data_frame.to_csv(output_path)


dept = [
    "MGH BLAKE 8 CARD SICU",
    "MGH ELLISON 8 CARDSURG",
    "MGH ELLISON 9 MED\\CCU",
    "MGH ELLISON 10 STP DWN",
    "MGH ELLISON11 CARD\\INT",
]
adt_dept = ["blake8", "ellison8", "ellison9", "ellison10", "ellison11"]

for i, _ in enumerate(dept):
    DEPT = MAPPING_DEPARTMENTS[dept[i]][0]
    DES_DEPT = dept[i]
    Bedmaster_DIR = f"/media/lm4-bedmaster/{DEPT}"
    EDW_DIR = "./depts_adts"
    ADT_TABLE = f"adt_{adt_dept[i]}.csv"

    os.system(
        f"python3 /home/$USER/repos/edw/icu/pipeline.py \
        --cohort_query {adt_dept[i]} \
        --destination {EDW_DIR} \
        --compute_adt",
    )
    os.rename(os.path.join(EDW_DIR, "adt.csv"), os.path.join(EDW_DIR, ADT_TABLE))

    assesser = AssessBedmasterCoverage()
    assesser.department_coverage(Bedmaster_DIR, EDW_DIR, ADT_TABLE, DES_DEPT)
