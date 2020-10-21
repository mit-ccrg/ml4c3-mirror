# Imports: standard library
import os
import sys
import logging

# Imports: third party
import pytest

# Imports: first party
from ml4c3.recipes import run
from ml4c3.arguments import parse_args

# pylint: disable=no-member


def test_summary_writer(temp_dir):
    logging.disable(logging.CRITICAL)

    test_dir = os.path.join(os.path.dirname(__file__), "icu_ingest")
    sys.argv = f"""
    .
    tensorize_icu
    --path_xref {test_dir}/data/xref_file_tensorize.csv
    --path_bedmaster {test_dir}/data/bedmaster
    --path_edw {test_dir}/data/edw
    --path_alarms {test_dir}/data/bedmaster_alarms
    --output_folder {temp_dir}
    --id {pytest.run_id}
    --tensors {os.path.join(temp_dir, pytest.run_id)}
    """.split()

    args = parse_args()
    run(args)

    sys.argv = [
        ".",
        "explore_icu",
        "--tensors",
        os.path.join(temp_dir, pytest.run_id),
    ]
    args = parse_args()
    run(args)

    expected_files = [
        f"{args.output_files_prefix}_hd5_demographics.csv",
    ]
    output_files = [
        file_name
        for file_name in os.listdir(args.tensors)
        if file_name.endswith(".csv")
    ]

    assert sorted(expected_files) == sorted(output_files)

    sys.argv = [
        ".",
        "explore_icu",
        "--tensors",
        os.path.join(temp_dir, pytest.run_id),
        "--input_tensors",
        "blood_pressure_systolic_timeseries",
        "ii_value",
        "code_start_start_date",
        "colonoscopy_start_date",
    ]
    args = parse_args()
    run(args)

    expected_files = [
        f"{args.output_files_prefix}_hd5_demographics.csv",
        f"{args.output_files_prefix}_hd5_intersection_demographics.csv",
        f"{args.output_files_prefix}_hd5_union_demographics.csv",
        f"{args.output_files_prefix}_continuous_signals_union.csv",
        f"{args.output_files_prefix}_continuous_signals_intersection.csv",
        f"{args.output_files_prefix}_timeseries_signals_union.csv",
        f"{args.output_files_prefix}_timeseries_signals_intersection.csv",
        f"{args.output_files_prefix}_event_signals_union.csv",
        f"{args.output_files_prefix}_event_signals_intersection.csv",
        f"{args.output_files_prefix}_mrns_list_union.csv",
        f"{args.output_files_prefix}_mrns_list_intersection.csv",
    ]
    output_files = [
        file_name
        for file_name in os.listdir(args.tensors)
        if file_name.endswith(".csv")
    ]

    assert sorted(expected_files) == sorted(output_files)
