# Imports: standard library
import os
import logging
import unittest.mock


def test_get_xref_files(cross_referencer):
    logging.disable(logging.CRITICAL)

    test_dir = os.path.dirname(__file__)
    expected_dic = {
        "123": {
            "345": [
                f"{test_dir}/data/bedmaster/file1_1_v4.mat",
                f"{test_dir}/data/bedmaster/file1_2_v4.mat",
                f"{test_dir}/data/bedmaster/file1_10_v4.mat",
            ],
            "890": [f"{test_dir}/data/bedmaster/file2_1_v4.mat"],
        },
        "456": {"980": [f"{test_dir}/data/bedmaster/file3_1_v4.mat"]},
    }
    expected_dic2 = {"456": {"980": [f"{test_dir}/data/bedmaster/file3_1_v4.mat"]}}
    expected_dic3 = {
        "123": {
            "345": [
                f"{test_dir}/data/bedmaster/file1_1_v4.mat",
                f"{test_dir}/data/bedmaster/file1_2_v4.mat",
                f"{test_dir}/data/bedmaster/file1_10_v4.mat",
            ],
            "890": [f"{test_dir}/data/bedmaster/file2_1_v4.mat"],
        },
    }
    assert cross_referencer.get_xref_files() == expected_dic
    assert cross_referencer.get_xref_files(mrns=["456"]) == expected_dic2
    assert (
        cross_referencer.get_xref_files(starting_time=280, ending_time=420)
        == expected_dic2
    )
    assert (
        cross_referencer.get_xref_files(
            overwrite_hd5=False,
            tensors=f"{test_dir}/data/",
        )
        == expected_dic2
    )
    assert cross_referencer.get_xref_files(n_patients=4) == expected_dic
    assert cross_referencer.get_xref_files(n_patients=1) == expected_dic3
    assert cross_referencer.get_xref_files(mrns=["456"], n_patients=1) == expected_dic2
    assert cross_referencer.get_xref_files(mrns=["456"], n_patients=0) == {}
    assert (
        cross_referencer.get_xref_files(
            starting_time=280,
            ending_time=420,
            flag_one_source=True,
        )
        == {}
    )


def test_stats(cross_referencer):
    test_dir = os.path.dirname(__file__)
    expected_message = [
        f"INFO:root:MRNs in {test_dir}/data/edw: 2\n"
        f"MRNs in {test_dir}/data/xref_file.csv: 2\n"
        f"Union MRNs: 2\n"
        f"Intersect MRNs: 2\n"
        f"CSNs in {test_dir}/data/edw: 1\n"
        f"CSNs in {test_dir}/data/xref_file.csv: 3\n"
        f"Union CSNs: 1\n"
        f"Intersect CSNs: 3\n"
        f"Bedmaster files IDs in {test_dir}/data/xref_file.csv: 3\n"
        f"Intersect Bedmaster files: 5\n",
    ]
    log = unittest.TestCase()
    cross_referencer.get_xref_files()
    logging.disable(logging.NOTSET)
    try:
        with log.assertLogs() as log_messages:
            logging.getLogger(cross_referencer.stats())
    except AssertionError:
        pass
    assert sorted(log_messages.output) == sorted(expected_message)
    logging.disable(logging.CRITICAL)
