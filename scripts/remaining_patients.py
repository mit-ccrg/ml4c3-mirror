# Imports: standard library
import os
import argparse

# Imports: first party
from ml4c3.assess_icu_coverage import ICUCoverageAssesser

# pylint: disable=redefined-outer-name


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort_csv",
        type=str,
        default="/media/mad3/adt.csv",
        help="File with a list of MRNs and CSNs to check their existance.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/media/mad3/edw/",
        help="Directory where the analysis is performed.",
    )
    parser.add_argument(
        "--output_dir",
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
        args.cohort_csv,
        args.input_dir,
        args.remove,
    )
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    remaining_patients.to_csv(
        os.path.join(args.output_dir, "remaining_patients.csv"),
        index=False,
    )


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
