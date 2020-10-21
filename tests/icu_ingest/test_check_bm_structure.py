# Imports: standard library
import os
import logging
import unittest.mock

# pylint: disable=no-member


def test_check_bm_structure(get_bm_checker):
    test_dir = os.path.join(os.path.dirname(__file__))
    expected_warning_message = [
        "ERROR:root:Wrong file format: the grups ['vs', 'wv'] were not found in the "
        f"input file {test_dir}/data/bedmaster/error_file.mat.",
        "WARNING:root:Missing file: the mapping name BIG06 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BIG06.csv.",
        "WARNING:root:Missing file: the mapping name BIG09 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BIG09.csv.",
        "WARNING:root:Missing file: the mapping name BIG09PU in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BIG09PU.csv.",
        "WARNING:root:Missing file: the mapping name BIG11 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BIG11.csv.",
        "WARNING:root:Missing file: the mapping name BIG12 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BIG12.csv.",
        "WARNING:root:Missing file: the mapping name BIG13 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BIG13.csv.",
        "WARNING:root:Missing file: the mapping name BIG14 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BIG14.csv.",
        "WARNING:root:Missing file: the mapping name BIG7 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BIG7.csv.",
        "WARNING:root:Missing file: the mapping name BIG9 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BIG9.csv.",
        "WARNING:root:Missing file: the mapping name BLAKE6 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BLAKE6.csv.",
        "WARNING:root:Missing file: the mapping name BLK07 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BLK07.csv.",
        "WARNING:root:Missing file: the mapping name BLK10 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BLK10.csv.",
        "WARNING:root:Missing file: the mapping name BLK12 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BLK12.csv.",
        "WARNING:root:Missing file: the mapping name BLK13 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BLK13.csv.",
        "WARNING:root:Missing file: the mapping name ELL-7 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL-7.csv.",
        "WARNING:root:Missing file: the mapping name ELL04 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL04.csv.",
        "WARNING:root:Missing file: the mapping name ELL09 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL09.csv.",
        "WARNING:root:Missing file: the mapping name ELL10 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL10.csv.",
        "WARNING:root:Missing file: the mapping name ELL11 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL11.csv.",
        "WARNING:root:Missing file: the mapping name ELL12 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL12.csv.",
        "WARNING:root:Missing file: the mapping name ELL13 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL13.csv.",
        "WARNING:root:Missing file: the mapping name ELL14 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL14.csv.",
        "WARNING:root:Missing file: the mapping name ELL16 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL16.csv.",
        "WARNING:root:Missing file: the mapping name ELL17 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL17.csv.",
        "WARNING:root:Missing file: the mapping name ELL18 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL18.csv.",
        "WARNING:root:Missing file: the mapping name ELL19 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL19.csv.",
        "WARNING:root:Missing file: the mapping name ELL4 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL4.csv.",
        "WARNING:root:Missing file: the mapping name ELL6 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL6.csv.",
        "WARNING:root:Missing file: the mapping name ELL9 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL9.csv.",
        "WARNING:root:Missing file: the mapping name LUN06 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_LUN06.csv.",
        "WARNING:root:Missing file: the mapping name LUN06IM in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_LUN06IM.csv.",
        "WARNING:root:Missing file: the mapping name LUN07 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_LUN07.csv.",
        "WARNING:root:Missing file: the mapping name LUN08 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_LUN08.csv.",
        "WARNING:root:Missing file: the mapping name LUN09 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_LUN09.csv.",
        "WARNING:root:Missing file: the mapping name LUN10 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_LUN10.csv.",
        "WARNING:root:Missing file: the mapping name WHITE10 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_WHITE10.csv.",
        "WARNING:root:Missing file: the mapping name WHITE11 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_WHITE11.csv.",
        "WARNING:root:Missing file: the mapping name WHT06 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_WHT06.csv.",
        "WARNING:root:Missing file: the mapping name WHT07 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_WHT07.csv.",
        "WARNING:root:Missing file: the mapping name WHT08 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_WHT08.csv.",
        "WARNING:root:Missing file: the mapping name WHT09 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_WHT09.csv.",
        "WARNING:root:Missing file: the mapping name WHT12 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_WHT12.csv.",
        "WARNING:root:Missing file: the mapping name WHT13 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_WHT13.csv.",
        "WARNING:root:Missing file: the mapping name None in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_None.csv.",
    ]
    log = unittest.TestCase()
    try:
        with log.assertLogs() as log_messages:
            bm_checker = get_bm_checker("bm")
            logging.getLogger(bm_checker.check_mat_files_structure())
            logging.getLogger(bm_checker.check_alarms_files_structure())
    except AssertionError:
        pass
    assert sorted(log_messages.output) == sorted(expected_warning_message)

    expected_warning_message = [
        "WARNING:root:Missing file: the mapping name BIG06 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BIG06.csv.",
        "WARNING:root:Missing file: the mapping name BIG09 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BIG09.csv.",
        "WARNING:root:Missing file: the mapping name BIG09PU in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BIG09PU.csv.",
        "WARNING:root:Missing file: the mapping name BIG11 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BIG11.csv.",
        "WARNING:root:Missing file: the mapping name BIG12 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BIG12.csv.",
        "WARNING:root:Missing file: the mapping name BIG13 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BIG13.csv.",
        "WARNING:root:Missing file: the mapping name BIG14 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BIG14.csv.",
        "WARNING:root:Missing file: the mapping name BIG7 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BIG7.csv.",
        "WARNING:root:Missing file: the mapping name BIG9 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BIG9.csv.",
        "WARNING:root:Missing file: the mapping name BLAKE6 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BLAKE6.csv.",
        "WARNING:root:Missing file: the mapping name BLK07 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BLK07.csv.",
        "WARNING:root:Missing file: the mapping name BLK10 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BLK10.csv.",
        "WARNING:root:Missing file: the mapping name BLK12 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BLK12.csv.",
        "WARNING:root:Missing file: the mapping name BLK13 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_BLK13.csv.",
        "WARNING:root:Missing file: the mapping name ELL-7 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL-7.csv.",
        "WARNING:root:Missing file: the mapping name ELL04 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL04.csv.",
        "WARNING:root:Missing file: the mapping name ELL09 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL09.csv.",
        "WARNING:root:Missing file: the mapping name ELL10 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL10.csv.",
        "WARNING:root:Missing file: the mapping name ELL11 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL11.csv.",
        "WARNING:root:Missing file: the mapping name ELL12 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL12.csv.",
        "WARNING:root:Missing file: the mapping name ELL13 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL13.csv.",
        "WARNING:root:Missing file: the mapping name ELL14 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL14.csv.",
        "WARNING:root:Missing file: the mapping name ELL16 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL16.csv.",
        "WARNING:root:Missing file: the mapping name ELL17 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL17.csv.",
        "WARNING:root:Missing file: the mapping name ELL18 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL18.csv.",
        "WARNING:root:Missing file: the mapping name ELL19 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL19.csv.",
        "WARNING:root:Missing file: the mapping name ELL4 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL4.csv.",
        "WARNING:root:Missing file: the mapping name ELL6 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL6.csv.",
        "WARNING:root:Missing file: the mapping name ELL9 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_ELL9.csv.",
        "WARNING:root:Missing file: the mapping name LUN06 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_LUN06.csv.",
        "WARNING:root:Missing file: the mapping name LUN06IM in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_LUN06IM.csv.",
        "WARNING:root:Missing file: the mapping name LUN07 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_LUN07.csv.",
        "WARNING:root:Missing file: the mapping name LUN08 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_LUN08.csv.",
        "WARNING:root:Missing file: the mapping name LUN09 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_LUN09.csv.",
        "WARNING:root:Missing file: the mapping name LUN10 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_LUN10.csv.",
        "WARNING:root:Missing file: the mapping name WHITE10 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_WHITE10.csv.",
        "WARNING:root:Missing file: the mapping name WHITE11 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_WHITE11.csv.",
        "WARNING:root:Missing file: the mapping name WHT06 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_WHT06.csv.",
        "WARNING:root:Missing file: the mapping name WHT07 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_WHT07.csv.",
        "WARNING:root:Missing file: the mapping name WHT08 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_WHT08.csv.",
        "WARNING:root:Missing file: the mapping name WHT09 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_WHT09.csv.",
        "WARNING:root:Missing file: the mapping name WHT12 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_WHT12.csv.",
        "WARNING:root:Missing file: the mapping name WHT13 in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_WHT13.csv.",
        f"WARNING:root:Unexpected files: ['{test_dir}/data/edw/123', "
        f"'{test_dir}/data/edw/456', "
        f"'{test_dir}/data/edw/adt.csv']. Just .mat files should be stored in "
        f"{test_dir}/data/edw.",
        "WARNING:root:Missing file: the mapping name None in ALARMS_FILES['names'] "
        "(ml4icu/globals.py) doesn't have its corresponding file "
        f"{test_dir}/data/bedmaster_alarms/bm_alarms_None.csv.",
    ]
    log = unittest.TestCase()
    try:
        with log.assertLogs() as log_messages:
            bm_checker = get_bm_checker("edw")
            logging.getLogger(bm_checker.check_mat_files_structure())
            logging.getLogger(bm_checker.check_alarms_files_structure())
    except AssertionError:
        pass
    assert sorted(log_messages.output) == sorted(expected_warning_message)
