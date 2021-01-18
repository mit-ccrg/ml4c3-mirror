# Imports: standard library
import os
import copy
import json
import time
import logging
import argparse
import datetime
import itertools
from multiprocessing import Queue, Process

# Imports: third party
import numpy as np
import pandas as pd


def train_model_worker(
    args: argparse.Namespace,
    gpu: int,
    trial: int,
    results: Queue,
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
        results.put((trial, performance_metrics))


def hyperoptimize(args: argparse.Namespace):
    with open(args.hyperoptimize_config_file, "r") as file:
        hyperparameter_options = json.load(file)
    keys, values = zip(*hyperparameter_options.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    logging.info(
        f"Generated {len(permutations)} hyperparameter combinations from {args.hyperoptimize_config_file}",
    )

    np.random.shuffle(permutations)
    permutations = permutations[: args.max_evals]
    logging.info(
        f"Randomly selected {len(permutations)} hyperparameter combinations to try",
    )

    workers = {i: None for i in range(args.hyperoptimize_workers)}
    results = Queue(maxsize=len(permutations))
    df = pd.DataFrame()
    df.index.name = "trial"
    base_output_folder = args.output_folder
    for trial, permutation in enumerate(permutations):
        _args = copy.deepcopy(args)
        # each permutation is a dictionary mapping parameter name to parameter value
        for key, value in permutation.items():
            df.loc[trial, key] = str(value)
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
                        args=(_args, idx, trial, results),
                    )
                    worker.start()
                    workers[idx] = worker
                    logging.info(
                        f"Dispatched trial {trial} ({trial+1} / {len(permutations)})",
                    )
                    break
            time.sleep(1)

    for idx, worker in workers.items():
        if isinstance(worker, Process):
            worker.join()

    for i in range(len(permutations)):
        trial, performance_metrics = results.get()
        for key, value in performance_metrics.items():
            df.loc[trial, key] = value

    for i in range(len(permutations)):
        logging.info(f"Trial {i} completed with results:\n\n{df.loc[i]}\n\n")

    df = df.reset_index()
    df.to_csv(
        os.path.join(base_output_folder, "metrics-and-hyperparameters.csv"),
        index=False,
    )
