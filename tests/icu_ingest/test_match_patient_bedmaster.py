# Imports: standard library
import os
from typing import List, Optional

# Imports: third party
import numpy as np
import pandas as pd
import pytest

# Imports: first party
# pylint: disable=no-member
from ingest.icu.match_patient_bedmaster import PatientBedmasterMatcher


def get_patient_bedmaster_matcher(
    bedmaster: str,
    desired_departments: Optional[List[str]] = None,
) -> PatientBedmasterMatcher:
    matcher = PatientBedmasterMatcher(
        bedmaster=bedmaster,
        adt=os.path.join(pytest.edw_dir, "adt.csv"),
        desired_departments=desired_departments,
    )
    return matcher


def test_file_structure_and_atributes(temp_dir):
    matcher = get_patient_bedmaster_matcher(pytest.bedmaster_matching)
    matcher2 = get_patient_bedmaster_matcher(
        bedmaster=pytest.bedmaster_matching,
        desired_departments=["MGH BLAKE 8 CARD SICU"],
    )
    assert not matcher.desired_departments
    assert matcher2.desired_departments == ["MGH BLAKE 8 CARD SICU"]

    # Check if columns of obtained xref table matches expected columns
    cross_ref_file_matched = os.path.join(temp_dir, "xref_file_matched.csv")
    matcher.match_files(cross_ref_file_matched)
    expected_df = pd.read_csv(os.path.join(pytest.datadir, "xref_file_matched_exp.csv"))
    obt_df = pd.read_csv(cross_ref_file_matched)
    assert set(expected_df.keys()) == set(obt_df.keys())


def test_xref_file_generation(temp_dir):
    def test_xref_file_results(
        temp_dir,
        bedmaster_dir,
    ):
        matcher = get_patient_bedmaster_matcher(
            bedmaster=bedmaster_dir,
        )
        expected_df = pd.read_csv(
            os.path.join(pytest.datadir, "xref_file_matched_exp.csv"),
        )
        cross_ref_file_matched = os.path.join(temp_dir, "xref_file_matched.csv")

        matcher.match_files(cross_ref_file_matched)
        obt_df = pd.read_csv(cross_ref_file_matched)

        # Remove the path prefix to leave just the file name
        obt_df["path"] = obt_df["path"].apply(lambda path: os.path.split(path)[1])
        assert (expected_df == obt_df).all().all()

        matcher2 = get_patient_bedmaster_matcher(
            bedmaster=bedmaster_dir,
            desired_departments=["MGH BLAKE 8 CARD SICU"],
        )
        matcher2.match_files(cross_ref_file_matched)
        obt_df2 = pd.read_csv(cross_ref_file_matched)

        # Remove the path prefix to leave just the file name
        obt_df2["path"] = obt_df2["path"].apply(lambda path: os.path.split(path)[1])

        assert (
            expected_df[expected_df["department"] == "BLK08"]
            .reset_index(drop=True)
            .equals(obt_df2.reset_index(drop=True))
        )

    folder = pytest.bedmaster_matching
    test_xref_file_results(temp_dir, folder)
