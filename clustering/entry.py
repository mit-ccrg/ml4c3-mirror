# Imports: standard library
import os
import logging

# Imports: first party
from clustering.objects.finder import Explorer, RequestReport


def find(args):
    finder = Explorer(args.tensors)
    report = finder.find(
        args.signals,
        cell=args.cell_value,
        stay_length=args.max_stay_length,
    )
    report.to_csv(os.path.join(args.output_folder, f"{args.csv_name}.csv"))


def extract(args):
    report_path = args.report_path
    hd5_path = args.tensors
    output_path = os.path.join(args.output_folder, args.output_file)

    logging.debug(f"Reading report from {report_path}...")
    report = RequestReport.from_csv(report_path)
    logging.debug(f"Report loaded!")

    logging.debug(f"Extracting data from {hd5_path}...")
    explorer = Explorer(hd5_path)
    bundle = explorer.extract_data(report)
    logging.debug(f"Data extracted and bundle created!")

    logging.debug(f"Saving bundle on {output_path}...")
    bundle.store(output_path)
    logging.debug(f"Bundle saved!")


def cluster_ui(args):
    # run_cluster_ui(args)
    raise ValueError(
        "Clustering UI is disabled for the moment! But we are working on it",
    )
