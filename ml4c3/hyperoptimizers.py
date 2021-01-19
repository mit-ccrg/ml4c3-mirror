# Imports: standard library
import os
import copy
import json
import time
import logging
import argparse
import datetime
import itertools
import subprocess
from multiprocessing import Queue, Process

# Imports: third party
import numpy as np
import pandas as pd


def train_model_worker(
    args: argparse.Namespace,
    gpu: int,
    trial: int,
    result_q: Queue,
):
    performance_metrics = {}
    try:
        if args.recipe != "train":
            gpu = ""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # Imports: first party
        from ml4c3.logger import load_config
        from ml4c3.recipes import train_model
        from ml4c3.arguments import _load_tensor_maps

        now_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        load_config(
            log_level=args.logging_level,
            log_dir=args.output_folder,
            log_file_basename="log_" + now_string,
            log_name=f"hyperoptimization_trial_{trial}",
            log_console=False,
        )
        _load_tensor_maps(args)
        performance_metrics = train_model(args)
    finally:
        result_q.put((trial, performance_metrics))


def collect_results(
    result_df: pd.DataFrame,
    result_q: Queue,
    base_output_folder: str,
    n_permutations: int,
):
    result_path = os.path.join(base_output_folder, "metrics-and-hyperparameters.csv")
    for i in range(n_permutations):
        trial, performance_metrics = result_q.get()
        for key, value in performance_metrics.items():
            result_df.loc[trial, key] = f"{value:.3}"

        # Incrementally save results
        result_df.to_csv(result_path)


def hyperoptimize(args: argparse.Namespace):
    # Import hyperoptimization config file and select N random permutations
    with open(args.hyperoptimize_config_file, "r") as file:
        hyperparameter_options = json.load(file)
    keys, values = zip(*hyperparameter_options.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    logging.info(
        f"Generated {len(permutations)} hyperparameter combinations from "
        f"{args.hyperoptimize_config_file}",
    )

    np.random.shuffle(permutations)
    permutations = permutations[: args.max_evals]
    logging.info(
        f"Randomly selected {len(permutations)} hyperparameter combinations to try",
    )

    # Infer the number of hyperoptimize workers as the number of available gpus
    if args.hyperoptimize_workers is None:
        # Cannot rely on built-in tensorflow methods because importing tensorflow
        # makes all gpus visible and prevents setting visible devices within workers
        n_gpus = str(subprocess.check_output(["nvidia-smi", "-L"])).count("UUID")
        args.hyperoptimize_workers = n_gpus
    logging.info(f"Using {args.hyperoptimize_workers} GPUs for hyperoptimization")

    # Setup workers, prepopulate dataframe to circumvent concurrent access by result
    # worker and start result collection worker
    workers = {i: None for i in range(args.hyperoptimize_workers)}
    result_q = Queue(maxsize=len(permutations))
    result_df = pd.DataFrame()
    result_df.index.name = "trial"
    for trial, permutation in enumerate(permutations):
        for key, value in permutation.items():
            result_df.loc[trial, key] = str(value)
    base_output_folder = args.output_folder
    result_worker = Process(
        target=collect_results,
        args=(result_df, result_q, base_output_folder, len(permutations)),
    )
    result_worker.start()

    # Dispatch trials to workers
    for trial, permutation in enumerate(permutations):
        _args = copy.deepcopy(args)
        # Each permutation is a dictionary mapping parameter name to parameter value
        for key, value in permutation.items():
            vars(_args)[key] = value
        _args.output_folder = os.path.join(base_output_folder, "trials", str(trial))
        os.makedirs(_args.output_folder, exist_ok=True)

        assigned = False
        while not assigned:
            for idx, worker in workers.items():
                # try to clear prior jobs
                if isinstance(worker, Process) and worker.exitcode is not None:
                    worker.join()
                    worker = None
                # try to assign the next permutation to a free worker
                if worker is None:
                    assigned = True
                    worker = Process(
                        target=train_model_worker,
                        args=(_args, idx, trial, result_q),
                    )
                    worker.start()
                    workers[idx] = worker
                    logging.info(
                        f"Dispatched trial {trial} ({trial+1} / {len(permutations)})",
                    )
                    break
            time.sleep(1)

    # Cleanup workers
    for idx, worker in workers.items():
        if isinstance(worker, Process):
            worker.join()
    result_worker.join()
