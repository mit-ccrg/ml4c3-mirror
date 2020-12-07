# Imports: standard library
import os
import logging
import unittest.mock

# Imports: third party
# pylint: disable=no-member
import pytest

# Imports: first party
from definitions.icu import MAPPING_DEPARTMENTS
from ingest.icu.check_structure import BedmasterChecker

DEPARTMENTS = set()
for department in MAPPING_DEPARTMENTS:
    DEPARTMENTS.update(MAPPING_DEPARTMENTS[department])
alarms_files = os.listdir(pytest.alarms_dir)
for alarm_file in alarms_files:
    department = alarm_file.replace("bedmaster_alarms_", "").replace(".csv", "")
    DEPARTMENTS.remove(department)


def get_bedmaster_checker(directory: str) -> BedmasterChecker:
    if directory == "bedmaster":
        checker = BedmasterChecker(pytest.bedmaster_dir, pytest.alarms_dir)
    else:
        checker = BedmasterChecker(pytest.edw_dir, pytest.alarms_dir)
    return checker


def test_check_bedmaster_structure():
    test_dir = os.path.join(os.path.dirname(__file__))
    expected_warning_message = [
        "ERROR:root:Wrong file format: the groups ['vs', 'wv'] were not found in the "
        f"input file {test_dir}/data/bedmaster/error_file.mat.",
    ]
    for dept in DEPARTMENTS:
        expected_warning_message.append(
            f"WARNING:root:Missing file: the mapping name {dept} in "
            "ALARMS_FILES['names'] doesn't have its corresponding file "
            f"{test_dir}/data/bedmaster_alarms/bedmaster_alarms_{dept}.csv.",
        )
    log = unittest.TestCase()
    try:
        with log.assertLogs() as log_messages:
            bedmaster_checker = get_bedmaster_checker("bedmaster")
            logging.getLogger(bedmaster_checker.check_mat_files_structure())
            logging.getLogger(bedmaster_checker.check_alarms_files_structure())
    except AssertionError:
        pass
    assert sorted(log_messages.output) == sorted(expected_warning_message)
    edw_files = []
    for root, _, files in os.walk(os.path.join(test_dir, "data", "edw")):
        for file in files:
            if not file.endswith(".mat"):
                edw_files.append(os.path.join(root, file))
    expected_warning_message = []
    for dept in DEPARTMENTS:
        expected_warning_message.append(
            f"WARNING:root:Missing file: the mapping name {dept} in "
            "ALARMS_FILES['names'] doesn't have its corresponding file "
            f"{test_dir}/data/bedmaster_alarms/bedmaster_alarms_{dept}.csv.",
        )
    expected_warning_message.append(
        f"WARNING:root:Unexpected files: {sorted(edw_files)}. "
        f"Just .mat files should be stored in {test_dir}/data/edw.",
    )
    log = unittest.TestCase()
    try:
        with log.assertLogs() as log_messages:
            bedmaster_checker = get_bedmaster_checker("edw")
            logging.getLogger(bedmaster_checker.check_mat_files_structure())
            logging.getLogger(bedmaster_checker.check_alarms_files_structure())
    except AssertionError:
        pass
    assert sorted(log_messages.output) == sorted(expected_warning_message)
