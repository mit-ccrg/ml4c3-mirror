# Imports: standard library
import os
import sys
import logging
from shutil import copy2

# Imports: third party
import pandas as pd
import pytest

# Imports: first party
from ml4c3.recipes import run
from ml4c3.arguments import parse_args

# pylint: disable=no-member


def test_pre_tensorize_explorations(temp_dir):
    # logging.disable(logging.CRITICAL)

    for file in os.listdir(pytest.bedmaster_dir):
        if file.startswith("bedmaster_file-"):
            copy2(os.path.join(pytest.bedmaster_dir, file), temp_dir)

    sample_csv = pd.DataFrame.from_dict({"MRN": [123, 456]})
    sample_csv.to_csv(os.path.join(temp_dir, "sample_csv.csv"))
    sys.argv = [
        ".",
        "pre_tensorize_explore",
        "--path_bedmaster",
        temp_dir,
        "--path_edw",
        pytest.edw_dir,
        "--path_xref",
        pytest.cross_ref_file,
        "--output_folder",
        temp_dir,
        "--detailed_bedmaster",
        "--sample_csv",
        os.path.join(temp_dir, "sample_csv.csv"),
    ]
    args = parse_args()
    run(args)

    expected_files = [
        f"{args.summary_stats_base_name}_edw_demographics.csv",
        f"{args.summary_stats_base_name}_coverage.csv",
        f"{args.summary_stats_base_name}_signals_summary.csv",
        f"{args.summary_stats_base_name}_bedmaster_signal_stats.csv",
        f"{args.summary_stats_base_name}_bedmaster_files_stats.csv",
        "sample_csv.csv",
    ]

    output_files = [
        file_name
        for file_name in os.listdir(args.output_folder)
        if file_name.endswith(".csv")
    ]
    assert sorted(expected_files) == sorted(output_files)

    edw_df = pd.read_csv(os.path.join(args.output_folder, expected_files[0]))
    cross_ref_df = pd.read_csv(os.path.join(args.output_folder, expected_files[1]))
    signals_df = pd.read_csv(os.path.join(args.output_folder, expected_files[2]))
    bedmaster_signals_df = pd.read_csv(
        os.path.join(args.output_folder, expected_files[3]),
        index_col=0,
    )
    bedmaster_files_df = pd.read_csv(
        os.path.join(args.output_folder, expected_files[4]),
    )

    assert len(edw_df.index) == 9
    assert sorted(edw_df.columns) == sorted(
        ["field", "count", "min", "max", "mean", "total", "%"],
    )

    assert len(cross_ref_df.index) == 8
    assert sorted(cross_ref_df.columns) == sorted(["field", "count"])

    assert len(signals_df.index) == len(signals_df["signal"].unique())
    assert sorted(signals_df.columns) == sorted(
        ["signal", "count", "source", "total", "%"],
    )

    assert len(edw_df.index) == 9
    assert sorted(edw_df.columns) == sorted(
        ["field", "count", "min", "max", "mean", "total", "%"],
    )

    assert len(bedmaster_signals_df.index) == 11
    assert sorted(bedmaster_signals_df.columns) == sorted(
        [
            "channel",
            "total_overlap_bundles",
            "total_overlap_bundles_%",
            "files",
            "files_%",
            "source",
            "points",
            "min",
            "mean",
            "max",
            "dataevents",
            "sample_freq",
            "multiple_freq",
            "units",
            "scale_factor",
            "nan_on_time",
            "nan_on_time_%",
            "nan_on_values",
            "nan_on_values_%",
            "overlapped_points",
            "overlapped_points_%",
            "string_value_bundles",
            "defective_signal",
        ],
    )

    assert len(bedmaster_files_df.index) == 5
    assert sorted(bedmaster_files_df.columns) == sorted(["issue", "count", "count_%"])


def test_independent_args(temp_dir):
    for file in os.listdir(pytest.bedmaster_dir):
        if file.startswith("bedmaster_file-"):
            copy2(os.path.join(pytest.bedmaster_dir, file), temp_dir)

    def _get_output_files():
        return [
            file_name
            for file_name in os.listdir(temp_dir)
            if file_name.endswith(".csv")
        ]

    def _reset_dir():
        for file in os.listdir(temp_dir):
            if file.endswith(".csv"):
                os.remove(os.path.join(temp_dir, file))
        sample_csv = pd.DataFrame.from_dict({"MRN": [123, 456]})
        sample_csv.to_csv(os.path.join(temp_dir, "sample_csv.csv"))

    base_name = "pre_tensorize"
    expected_files = [
        f"{base_name}_edw_demographics.csv",
        f"{base_name}_coverage.csv",
        f"{base_name}_signals_summary.csv",
        "sample_csv.csv",
        f"{base_name}_bedmaster_signal_stats.csv",
        f"{base_name}_bedmaster_files_stats.csv",
    ]

    # Test standard
    sys.argv = [
        ".",
        "pre_tensorize_explore",
        "--path_edw",
        pytest.edw_dir,
        "--path_xref",
        pytest.cross_ref_file,
        "--path_bedmaster",
        temp_dir,
        "--output_folder",
        temp_dir,
        "--sample_csv",
        os.path.join(temp_dir, "sample_csv.csv"),
    ]
    _reset_dir()
    parsed_args = parse_args()
    run(parsed_args)
    assert sorted(expected_files[:4]) == sorted(_get_output_files())

    # Test Bedmaster detailed
    sys.argv.append("--detailed_bedmaster")
    parsed_args = parse_args()
    _reset_dir()
    run(parsed_args)

    assert sorted(expected_files) == sorted(_get_output_files())

    # Test just Bedmaster
    sys.argv.append("--no_xref")
    parsed_args = parse_args()
    _reset_dir()
    run(parsed_args)

    assert sorted(expected_files[3:]) == sorted(_get_output_files())

    # Test no_xref and no detailed
    sys.argv.remove("--detailed_bedmaster")
    _reset_dir()
    parsed_args = parse_args()
    run(parsed_args)

    assert sorted(expected_files[:4]) == sorted(_get_output_files())

    logging.disable(logging.NOTSET)
