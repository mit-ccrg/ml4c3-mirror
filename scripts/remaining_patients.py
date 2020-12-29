# Imports: standard library
import os
import argparse

# Imports: first party
from ml4c3.ingest.icu.assess_coverage import ICUCoverageAssesser

# pylint: disable=redefined-outer-name


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patient_csv",
        type=str,
        default="/media/ml4c3/adt.csv",
        help="File with a list of MRNs and CSNs to check their existance.",
    )
    parser.add_argument(
        "--edw",
        type=str,
        default="/media/ml4c3/edw/",
        help="Directory where the analysis is performed.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./results",
        help="Directory where the results are saved.",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="If the parameter is set, the unfinished MRNs and CSNs in "
        "--input_dir are deleted.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace):
    remaining_patients = ICUCoverageAssesser._compare(
        ICUCoverageAssesser,
        args.patient_csv,
        args.edw,
        args.remove,
    )
    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)
    remaining_patients.to_csv(
        os.path.join(args.output_folder, "remaining_patients.csv"),
        index=False,
    )


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
