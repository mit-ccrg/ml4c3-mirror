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
    path_bedmaster: str,
    desired_departments: Optional[List[str]] = None,
) -> PatientBedmasterMatcher:
    matcher = PatientBedmasterMatcher(
        path_bedmaster=path_bedmaster,
        path_adt=os.path.join(pytest.edw_dir, "adt.csv"),
        desired_departments=desired_departments,
    )
    return matcher


def test_file_structure_and_atributes(temp_dir):
    matcher = get_patient_bedmaster_matcher(pytest.bedmaster_matching)
    matcher2 = get_patient_bedmaster_matcher(
        path_bedmaster=pytest.bedmaster_matching,
        desired_departments=["MGH BLAKE 8 CARD SICU"],
    )
    assert not matcher.desired_departments
    assert matcher2.desired_departments == ["MGH BLAKE 8 CARD SICU"]

    cross_ref_file_matched = os.path.join(temp_dir, "xref_file_matched.csv")
    matcher.match_files(cross_ref_file_matched)
    expected_df = pd.read_csv(os.path.join(pytest.datadir, "xref_file_matched_exp.csv"))
    obt_df = pd.read_csv(cross_ref_file_matched)
    assert np.array_equal(expected_df.keys(), obt_df.keys())


def test_xref_file_generation(temp_dir):
    def test_xref_file_results(
        temp_dir,
        bedmaster_dir,
    ):
        matcher = get_patient_bedmaster_matcher(
            path_bedmaster=bedmaster_dir,
        )
        matcher2 = get_patient_bedmaster_matcher(
            path_bedmaster=bedmaster_dir,
            desired_departments=["MGH BLAKE 8 CARD SICU"],
        )

        expected_df = pd.read_csv(
            os.path.join(pytest.datadir, "xref_file_matched_exp.csv"),
        )
        cross_ref_file_matched = os.path.join(temp_dir, "xref_file_matched.csv")
        matcher.match_files(cross_ref_file_matched)
        obt_df = pd.read_csv(cross_ref_file_matched)

        assert (expected_df == obt_df).all().all()

        matcher2.match_files(cross_ref_file_matched)
        obt_df = pd.read_csv(cross_ref_file_matched)

        assert (expected_df[:4] == obt_df).all().all()

    folder = pytest.bedmaster_matching
    test_xref_file_results(temp_dir, folder)
