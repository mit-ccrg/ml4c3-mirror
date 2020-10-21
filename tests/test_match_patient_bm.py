# Imports: standard library
import os

# Imports: third party
import numpy as np
import pandas as pd
import pytest

# pylint: disable=no-member


def test_file_structure_and_atributes(get_patientbm_matcher, temp_dir):
    matcher = get_patientbm_matcher(pytest.bm_matching)
    matcher2 = get_patientbm_matcher(
        pytest.bm_matching, departments=["MGH BLAKE 8 CARD SICU"],
    )
    expect_des_departments = [
        "MGH BIGELOW 6 PICU",
        "MGH BIGELOW 7",
        "MGH BIGELOW 9 MED",
        "MGH BIGELOW 11 MED",
        "MGH BIGELOW 12",
        "MGH BIGELOW 13 RACU",
        "MGH BIGELOW 14 MED",
        "MGH BLAKE 4 ENDO DEPT",
        "MGH BLAKE 6 TRANSPLANT",
        "MGH BLAKE 7 MICU",
        "MGH BLAKE 8 CARD SICU",
        "MGH BLAKE 10 NICU",
        "MGH BLAKE 12 ICU",
        "MGH BLAKE 13 OB",
        "MGH ELLISON 4 SICU",
        "MGH ELLISON 6 ORTH\\URO",
        "MGH ELLISON 7 SURG\\URO",
        "MGH ELLISON 8 CARDSURG",
        "MGH ELLISON 9 MED\\CCU",
        "MGH ELLISON 10 STP DWN",
        "MGH ELLISON11 CARD\\INT",
        "MGH ELLISON 12 MED",
        "MGH ELLISON13A OB-ANTE",
        "MGH ELLISON 14 BRN ICU",
        "MGH ELLISON 14 PLASTCS",
        "MGH ELLISON 16 MED ONC",
        "MGH ELLISON 17 PEDI",
        "MGH ELLISON 18 PEDI",
        "MGH ELLISON19 THOR/VAS",
        "MGH LUNDER 6 NEURO ICU",
        "MGH LUNDER 7 NEURO",
        "MGH LUNDER 8 NEURO",
        "MGH LUNDER 9 ONCOLOGY",
        "MGH LUNDER 10 ONCOLOGY",
        "MGH WHITE 6 ORTHO\\OMF",
        "MGH WHITE 7 GEN SURG",
        "MGH WHITE 8 MEDICINE",
        "MGH WHITE 9 MEDICINE",
        "MGH WHITE 10 MEDICINE",
        "MGH WHITE 11 MEDICINE",
        "MGH WHITE 12",
        "MGH WHITE 13 PACU",
        "MGH CARDIAC CATH LAB",
        "MGH EMERGENCY",
        "MGH PERIOPERATIVE DEPT",
        "MGH EP PACER LAB",
        "MGH PHILLIPS 20 MED",
        "MGH CPC",
    ]

    assert expect_des_departments == matcher.des_depts
    assert matcher2.des_depts == ["MGH BLAKE 8 CARD SICU"]

    cross_ref_file_matched = os.path.join(temp_dir, "xref_file_matched.csv")
    matcher.match_files(cross_ref_file_matched)
    expected_df = pd.read_csv(os.path.join(pytest.datadir, "xref_file_matched_exp.csv"))
    obt_df = pd.read_csv(cross_ref_file_matched)
    assert np.array_equal(expected_df.keys(), obt_df.keys())


def test_xref_file_generation(get_patientbm_matcher, temp_dir):
    def test_xref_file_results(get_patientbm_matcher, temp_dir, lm4, bm_dir):
        matcher = get_patientbm_matcher(bm_dir, lm4_flag=lm4)
        matcher2 = get_patientbm_matcher(
            bm_dir, lm4_flag=lm4, departments=["MGH BLAKE 8 CARD SICU"],
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

    lm4 = False
    folder = pytest.bm_matching
    test_xref_file_results(get_patientbm_matcher, temp_dir, lm4, folder)
    lm4 = True
    folder = pytest.lm4_matching
    test_xref_file_results(get_patientbm_matcher, temp_dir, lm4, folder)
