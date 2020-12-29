# Imports: standard library
import os

# Imports: third party
import pytest

# Imports: first party
from ingest.icu.assess_coverage import ICUCoverageAssesser

# pylint: disable=no-member


def test_assesser(temp_dir):
    assesser = ICUCoverageAssesser(
        output_dir=temp_dir,
        cohort_csv=os.path.join(pytest.edw_dir, "adt.csv"),
        adt_csv=os.path.join(pytest.edw_dir, "adt.csv"),
    )
    assesser.assess_coverage(
        bedmaster=pytest.bedmaster_matching,
        edw=pytest.edw_dir,
        hd5=temp_dir,
    )
    expected_files = [
        "adt.csv",
        "xref.csv",
        "coverage.csv",
        "remaining_patients.csv",
    ]
    output_files = [
        file_name for file_name in os.listdir(temp_dir) if file_name.endswith(".csv")
    ]
    assert sorted(expected_files) == sorted(output_files)
