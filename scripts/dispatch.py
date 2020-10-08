#!/bin/python3

# Imports: standard library
import os
import re
import sys
import time
import socket
import logging
import argparse
import datetime
import itertools
import subprocess
from typing import List

env = os.environ.copy()


def _get_path_to_ecgs() -> str:
    """Check the hostname of the machine and return the appropriate path.
    If there is no match found, this function does not return anything, and
    the script ends up with a non-viable path prefix to HD5 files and will fail."""
    if "anduril" == socket.gethostname():
        path = "/storage/shared/ecg"
    elif "mithril" == socket.gethostname():
        path = "/storage/shared/ecg"
    elif "stultzlab" in socket.gethostname():
        path = "/storage/shared/ecg"
    else:
        path = ""
    return path


def _get_path_to_bootstraps() -> str:
    """Check the hostname of the machine and return the appropriate path.
    If there is no match found, this function does not return anything, and
    the script ends up with a non-viable path prefix to HD5 files and will fail."""
    if "anduril" == socket.gethostname():
        path = "~/dropbox/sts-data/bootstraps"
    elif "mithril" == socket.gethostname():
        path = "~/dropbox/sts-data/bootstraps"
    elif "stultzlab" in socket.gethostname():
        path = "/storage/shared/sts-data-deid/bootstraps"
    else:
        path = ""
    return os.path.expanduser(path)


def _get_path_to_results() -> str:
    """Check the hostname of the machine and return the appropriate path."""
    if "anduril" == socket.gethostname():
        path = "~/dropbox/sts-ecg/results"
    elif "mithril" == socket.gethostname():
        path = "~/dropbox/sts-ecg/results"
    elif "stultzlab" in socket.gethostname():
        path = "~/"
    else:
        path = "~/"
    return os.path.expanduser(path)


def setup_job(
    script: str,
    bootstrap: str,
    gpu: str,
    cohort: str,
    output_tensor: str,
    outcome_short_name: str,
) -> subprocess.Popen:
    """
    Setup environment variables, launch job, and return job object.
    """
    env["GPU"] = gpu
    env["BOOTSTRAP"] = bootstrap
    env["PATH_TO_ECGS"] = _get_path_to_ecgs()
    env["PATH_TO_BOOTSTRAPS"] = _get_path_to_bootstraps()
    env["PATH_TO_RESULTS"] = _get_path_to_results()
    env["COHORT"] = cohort
    env["OUTPUT_TENSOR"] = output_tensor
    env["OUTCOME_SHORT_NAME"] = outcome_short_name

    job = subprocess.Popen(
        f"bash {script}_temp".split(),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    append_str = ""
    if cohort != "":
        append_str += f", cohort {cohort}"
    if output_tensor != "":
        append_str += f", output tensor {output_tensor}"
    if outcome_short_name != "":
        append_str += f", outcome short name {outcome_short_name}"
    logging.info(
        f"Dispatched {script} with bootstrap {bootstrap} on gpu {gpu}{append_str}",
    )
    return job


def run(args: argparse.Namespace):
    start = time.time()

    gpu_jobs = {gpu: None for gpu in args.gpus}
    num_dispatched = 0

    try:
        for script in args.scripts:
            with open(script, "r") as original, open(f"{script}_temp", "w") as temp:
                temp.write(original.read().replace("-t ", ""))
            for (bootstrap, cohort, output_tensor_and_short_name) in itertools.product(
                args.bootstraps,
                args.cohorts,
                zip(args.output_tensors, args.outcome_short_names),
            ):
                output_tensor = output_tensor_and_short_name[0]
                outcome_short_name = output_tensor_and_short_name[1]
                assigned = False
                while not assigned:
                    for gpu, job in gpu_jobs.items():
                        if job is not None and job.poll() is None:
                            continue
                        if job is not None:
                            job.kill()
                        job = setup_job(
                            script=script,
                            bootstrap=bootstrap,
                            gpu=gpu,
                            cohort=cohort,
                            output_tensor=output_tensor,
                            outcome_short_name=outcome_short_name,
                        )
                        gpu_jobs[gpu] = job
                        assigned = True
                        num_dispatched += 1
                        break
                    time.sleep(1)
        for job in gpu_jobs.values():
            if job is not None:
                job.wait()
    finally:
        for job in gpu_jobs.values():
            if job is not None:
                job.kill()
        for script in args.scripts:
            try:
                os.remove(f"{script}_temp")
            except FileNotFoundError:
                continue
        logging.info(f"Cleaned up jobs")
    logging.info(
        f"Dispatched {num_dispatched} jobs in {time.time() - start:.0f} seconds",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        nargs="+",
        required=True,
        help="List of gpu devices to run on, specified by indices or intervals. For example, to use gpu 0, 3, 4, and 5: --gpus 0 3-5",
    )
    parser.add_argument(
        "--bootstraps",
        nargs="+",
        default=["0-9"],
        help="List of bootstraps to run on; same specification as gpus. default: 0-9",
    )
    parser.add_argument(
        "--scripts", nargs="+", required=True, help="list of paths to scripts to run",
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=[""],
        help="List of cohort (strings) to iterate over in bash script",
    )
    parser.add_argument(
        "--output_tensors",
        nargs="+",
        default=[""],
        help="List of output tensors to iterate over in bash script",
    )
    parser.add_argument(
        "--outcome_short_names",
        nargs="+",
        default=[""],
        help="List of outcome short names to iterate over in bash script",
    )

    args = parser.parse_args()

    log_formatter = logging.Formatter(
        "%(asctime)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(log_formatter)
    logger.addHandler(stdout_handler)

    interval = re.compile(r"^\d+-\d+$")
    if args.gpus is None or args.bootstraps is None or args.scripts is None:
        raise ValueError(f"Missing arguments")

    def _parse_index_list(idxs: List[str], name: str) -> List[str]:
        """
        Given a list of intervals, return a list of all values within all intervals.
        For example:
            _parse_index_list(["0-2", "5"], "bootstraps") -> ["0", "1", "2", "5"]
        """
        _idxs = set()
        for idx in idxs:
            if idx.isdigit():
                start = int(idx)
                end = start + 1
            elif interval.match(idx):
                start, end = map(int, idx.split("-"))
                end += 1
            else:
                raise ValueError(f"Invalid {name}: {idx}")
            _idxs = _idxs.union(set(range(start, end)))
        return list(map(str, _idxs))

    args.gpus = _parse_index_list(idxs=args.gpus, name="gpu")
    args.bootstraps = _parse_index_list(idxs=args.bootstraps, name="bootstrap")

    for script in args.scripts:
        if not os.path.isfile(script):
            raise ValueError(f"No script found at: {script}")

    return args


if __name__ == "__main__":
    try:
        args = parse_args()
        run(args)
    except Exception as e:
        logging.exception(e)
    finally:
        logging.warning(
            "If dispatched jobs launched docker containers, containers need to be manually cleaned up",
        )
