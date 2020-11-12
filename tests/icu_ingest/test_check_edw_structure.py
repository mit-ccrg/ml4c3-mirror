# Imports: standard library
import os
import logging
import unittest.mock

# Imports: third party
import pytest

# Imports: first party
from ml4c3.ingest.icu.check_icu_structure import EDWChecker


def get_edw_checker(directory: str) -> EDWChecker:
    if directory == "edw":
        checker = EDWChecker(pytest.edw_dir)
    else:
        checker = EDWChecker(pytest.bedmaster_dir)
    return checker


def test_check_edw_structure():
    test_dir = os.path.join(os.path.dirname(__file__), "data")
    expected_warning_message = [
        "ERROR:root:Wrong folder format: "
        f"{test_dir}/edw/456 doesn't contain any folder.",
        "ERROR:root:Wrong folder format: the files ['diagnosis.csv'] were not found "
        f"in the input folder {test_dir}/edw/123/345.",
        "WARNING:root:Unexpected files: "
        f"['{test_dir}/edw/123/345/flowsheet_v2.csv']. "
        "Just the specific .csv files should be saved in csns folders.",
        f"WARNING:root:Unexpected files: ['{test_dir}/edw/456/empty.csv']. "
        "Just folders should be stored inside mrns folders.",
    ]
    log = unittest.TestCase()
    try:
        with log.assertLogs() as log_messages:
            edw_checker = get_edw_checker("edw")
            logging.getLogger(edw_checker.check_structure())
    except AssertionError:
        pass
    assert sorted(log_messages.output) == sorted(expected_warning_message)

    files = [
        os.path.join(test_dir, "bedmaster", file)
        for file in os.listdir(f"{test_dir}/bedmaster")
    ]
    expected_warning_message = [
        "ERROR:root:Wrong folder format: adt.csv was not found in the input "
        f"directory {test_dir}/bedmaster.",
        f"WARNING:root:Unexpected files: {str(sorted(files))}. "
        "Just an adt table and mrns folders "
        f"should be stored in {test_dir}/bedmaster.",
    ]
    log = unittest.TestCase()
    try:
        with log.assertLogs() as log_messages:
            edw_checker = get_edw_checker("bedmaster")
            logging.getLogger(edw_checker.check_structure())
    except AssertionError:
        pass
    assert sorted(log_messages.output) == sorted(expected_warning_message)
