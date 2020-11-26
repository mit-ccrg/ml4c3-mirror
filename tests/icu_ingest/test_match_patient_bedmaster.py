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

        assert np.array_equal(
            np.sort(np.nan_to_num(expected_df["MRN"].values)),
            np.sort(np.nan_to_num(obt_df["MRN"].values)),
        )

        assert np.array_equal(
            np.sort(np.nan_to_num(expected_df["PatientEncounterID"].values)),
            np.sort(np.nan_to_num(obt_df["PatientEncounterID"].values)),
        )

        ts_exp = np.sort(expected_df["transferIn"].values)
        ts_obt = np.sort(obt_df["transferIn"].values)
        assert ((ts_exp == ts_obt) | (np.isnan(ts_exp) & np.isnan(ts_obt))).all()
        tf_exp = np.sort(expected_df["transferOut"].values)
        tf_obt = np.sort(obt_df["transferOut"].values)
        assert ((tf_exp == tf_obt) | (np.isnan(tf_exp) & np.isnan(tf_obt))).all()
        ufst_exp = np.sort(expected_df["unixFileStartTime"].values)
        ufst_obt = np.sort(obt_df["unixFileStartTime"].values)
        assert (
            (ufst_exp == ufst_obt) | (np.isnan(ufst_exp) & np.isnan(ufst_obt))
        ).all()
        ufet_exp = np.sort(expected_df["unixFileEndTime"].values)
        ufet_obt = np.sort(obt_df["unixFileEndTime"].values)
        assert (
            (ufet_exp == ufet_obt) | (np.isnan(ufet_exp) & np.isnan(ufet_obt))
        ).all()
        assert np.array_equal(
            np.sort(np.nan_to_num(expected_df["Department"].values)),
            np.sort(np.nan_to_num(obt_df["Department"].values)),
        )

        matcher2.match_files(cross_ref_file_matched)
        obt_df = pd.read_csv(cross_ref_file_matched)

        assert np.array_equal(
            np.nan_to_num(expected_df["MRN"].values[:-1]),
            np.nan_to_num(obt_df["MRN"].values),
        )
        assert np.array_equal(
            np.nan_to_num(expected_df["PatientEncounterID"].values[:-1]),
            np.nan_to_num(obt_df["PatientEncounterID"].values),
        )
        ts_exp = expected_df["transferIn"].values[:-1]
        ts_obt = obt_df["transferIn"].values
        assert ((ts_exp == ts_obt) | (np.isnan(ts_exp) & np.isnan(ts_obt))).all()
        tf_exp = expected_df["transferOut"].values[:-1]
        tf_obt = obt_df["transferOut"].values
        assert ((tf_exp == tf_obt) | (np.isnan(tf_exp) & np.isnan(tf_obt))).all()
        ufst_exp = expected_df["unixFileStartTime"].values[:-1]
        ufst_obt = obt_df["unixFileStartTime"].values
        assert (
            (ufst_exp == ufst_obt) | (np.isnan(ufst_exp) & np.isnan(ufst_obt))
        ).all()
        ufet_exp = expected_df["unixFileEndTime"].values[:-1]
        ufet_obt = obt_df["unixFileEndTime"].values
        assert (
            (ufet_exp == ufet_obt) | (np.isnan(ufet_exp) & np.isnan(ufet_obt))
        ).all()
        assert np.array_equal(
            np.nan_to_num(expected_df["Department"].values[:-1]),
            np.nan_to_num(obt_df["Department"].values),
        )

    folder = pytest.bedmaster_matching
    test_xref_file_results(temp_dir, folder)
