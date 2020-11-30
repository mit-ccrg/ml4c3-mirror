# pylint: disable=wrong-import-position, disable=wrong-import-order
# pylint: disable=dangerous-default-value
# Imports: standard library
import gc
import os
import json
import logging
import argparse
from typing import Dict, List
from collections import defaultdict

# Imports: third party
import numpy as np
import pandas as pd
import hyperopt
from hyperopt import hp, tpe, fmin
from skimage.filters import threshold_otsu

# Imports: first party
from ml4c3.plots import plot_metric_history
from ml4c3.models import (
    make_shallow_model,
    make_sklearn_model,
    train_model_from_datasets,
    make_multimodal_multitask_model,
)
from ml4c3.datasets import train_valid_test_datasets
from definitions.types import Arguments
from ml4c3.evaluations import predict_and_evaluate
from ml4c3.tensormap.TensorMap import TensorMap, update_tmaps

# fmt: off
# need matplotlib -> Agg -> pyplot
import matplotlib                       # isort:skip
matplotlib.use("Agg")                   # isort:skip
from matplotlib import pyplot as plt    # isort:skip
# fmt: on


MAX_LOSS = 9e9


def hyperoptimize(args: argparse.Namespace):
    """
    hyperopt is a Python library that performs Bayesian optimization over
    hyperparameters to minimize an objective function. Here, the objective function
    is loss_from_model. Hyperparameter combinations are randomly
    chosen and non-unique choices are skipped before model compilation. The
    computation to skip repeated combinations is fast and inexpensive. However, each
    non-unique combination counts towards the maximum number of models to evaluate.
    If a grid search over a relatively small search space is desired, set max_evals
    >> size of search space. In this case, it is likely, but not guaranteed, that
    all combinations will be seen.
    """
    with open(args.hyperoptimize_config_file, "r") as file:
        param_lists = json.load(file)
    space = {}
    keys = list(param_lists.keys())
    for key in keys:
        if not isinstance(param_lists[key], list):
            value = param_lists.pop(key)
            vars(args)[key] = value
            continue
        if len(param_lists[key]) == 1:
            value = param_lists.pop(key)
            vars(args)[key] = value[0]
            continue
        space.update({key: hp.choice(key, param_lists[key])})
    hyperparameter_optimizer(args=args, space=space, param_lists=param_lists)


def hyperparameter_optimizer(
    args: argparse.Namespace,
    space: Dict[str, hyperopt.pyll.base.Apply],
    param_lists: Arguments,
):
    histories = []
    aucs = []
    results_path = os.path.join(args.output_folder, args.id)
    i = 0
    seen_combinations = set()

    def loss_from_model(x: Arguments):
        model = None
        history = None
        auc = None
        nonlocal i
        i += 1

        try:
            trial_id = f"{i - 1}"
            trials_path = os.path.join(results_path, "trials")

            # only try unique parameter combinations
            params = str(x)
            if params in seen_combinations:
                raise ValueError(
                    f"Trial {trial_id}: hyperparameter combination is non-unique: "
                    f"{params}",
                )
            seen_combinations.add(params)

            set_args_from_x(args, x)
            if args.mode == "train":
                model = make_multimodal_multitask_model(**args.__dict__)
            elif args.mode == "train_keras_logreg":
                model = make_shallow_model(**args.__dict__)
            else:
                hyperparameters = {}
                if args.mode == "train_sklearn_logreg":
                    if args.l1 == 0 and args.l2 == 0:
                        args.c = 1e7
                    else:
                        args.c = 1 / (args.l1 + args.l2)
                    hyperparameters["c"] = args.c
                    hyperparameters["l1_ratio"] = args.c * args.l1
                elif args.mode == "train_sklearn_svm":
                    hyperparameters["c"] = args.c
                elif args.mode == "train_sklearn_randomforest":
                    hyperparameters["n_estimators"] = args.n_estimators
                    hyperparameters["max_depth"] = args.max_depth
                    hyperparameters["min_samples_split"] = args.min_samples_split
                    hyperparameters["min_samples_leaf"] = args.min_samples_leaf
                elif args.mode == "train_sklearn_xgboost":
                    hyperparameters["n_estimators"] = args.n_estimators
                    hyperparameters["max_depth"] = args.max_depth
                else:
                    raise ValueError("Uknown train mode: ", args.mode)
                # SKLearn only works with one output tmap
                assert len(args.tensor_maps_out) == 1
                model = make_sklearn_model(
                    model_type=args.sklearn_model_type,
                    hyperparameters=hyperparameters,
                )

            if model.count_params() > args.max_parameters:
                logging.info(
                    f"Model too big, max parameters is:{args.max_parameters}, model"
                    f" has:{model.count_params()}. Return max loss.",
                )
                return MAX_LOSS

            datasets, _, cleanups = train_valid_test_datasets(
                tensor_maps_in=args.tensor_maps_in,
                tensor_maps_out=args.tensor_maps_out,
                tensors=args.tensors,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                patient_csv=args.patient_csv,
                valid_ratio=args.valid_ratio,
                test_ratio=args.test_ratio,
                train_csv=args.train_csv,
                valid_csv=args.valid_csv,
                test_csv=args.test_csv,
                output_folder=args.output_folder,
                run_id=args.id,
                cache_off=args.cache_off,
                mixup_alpha=args.mixup_alpha,
            )
            train_dataset, valid_dataset, test_dataset = datasets
            model, history = train_model_from_datasets(
                model=model,
                tensor_maps_in=args.tensor_maps_in,
                tensor_maps_out=args.tensor_maps_out,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                epochs=args.epochs,
                patience=args.patience,
                learning_rate_patience=args.learning_rate_patience,
                learning_rate_reduction=args.learning_rate_reduction,
                output_folder=trials_path,
                run_id=trial_id,
                image_ext=args.image_ext,
                return_history=True,
                plot=args.make_training_plots,
            )
            history.history["parameter_count"] = [model.count_params()]
            histories.append(history.history)
            train_auc = predict_and_evaluate(
                model=model,
                data=train_dataset,
                tensor_maps_in=args.tensor_maps_in,
                tensor_maps_out=args.tensor_maps_out,
                plot_path=os.path.join(trials_path, trial_id),
                data_split="train",
                image_ext=args.image_ext,
            )
            test_auc = predict_and_evaluate(
                model=model,
                data=test_dataset,
                tensor_maps_in=args.tensor_maps_in,
                tensor_maps_out=args.tensor_maps_out,
                plot_path=os.path.join(trials_path, trial_id),
                data_split="test",
                image_ext=args.image_ext,
            )
            auc = {"train": train_auc, "test": test_auc}
            aucs.append(auc)
            plot_metric_history(
                history,
                None,
                "",
                os.path.join(trials_path, trial_id),
            )
            logging.info(
                f"Current architecture:\n{_string_from_architecture_dict(x)}\nCurrent"
                f" model size: {model.count_params()}.",
            )

            logging.info(f"Iteration {i} / {args.max_evals} max evaluations")

            loss_and_metrics = model.evaluate(train_dataset)
            logging.info(f"Train loss: {loss_and_metrics[0]:0.3f}")

            loss_and_metrics = model.evaluate(test_dataset)
            logging.info(f"Test loss: {loss_and_metrics[0]:0.3f}")

            logging.info(f"Train AUC(s): {train_auc}")
            logging.info(f"Test AUC(s): {test_auc}")

            for cleanup in cleanups:
                cleanup()
            return loss_and_metrics[0]

        except ValueError:
            logging.exception(
                "ValueError trying to make a model for hyperparameter optimization."
                " Returning max loss.",
            )
            return MAX_LOSS
        except:
            logging.exception(
                "Error trying hyperparameter optimization. Returning max loss.",
            )
            return MAX_LOSS
        finally:
            del model
            gc.collect()
            if auc is None:
                aucs.append({"train": {"BAD_MODEL": -1}, "test": {"BAD_MODEL": -1}})
            if history is None:
                histories.append(
                    {
                        "loss": [MAX_LOSS],
                        "val_loss": [MAX_LOSS],
                        "parameter_count": [0],
                    },
                )

    trials = hyperopt.Trials()
    fmin(
        fn=loss_from_model,
        space=space,
        algo=tpe.suggest,
        max_evals=args.max_evals,
        trials=trials,
    )
    plot_trials(
        trials=trials,
        histories=histories,
        aucs=aucs,
        figure_path=results_path,
        image_ext=args.image_ext,
        param_lists=param_lists,
    )


def set_args_from_x(args: argparse.Namespace, x: Arguments):
    for k in args.__dict__:
        if k in x:
            logging.info(f"arg: {k}")
            logging.info(f"value from hyperopt: {x[k]}")
            logging.info(f"original value in args: {args.__dict__[k]}")
            if isinstance(args.__dict__[k], int):
                args.__dict__[k] = int(x[k])
            elif isinstance(args.__dict__[k], float):
                v = float(x[k])
                if v == int(v):
                    v = int(v)
                args.__dict__[k] = v
            elif isinstance(args.__dict__[k], list):
                if isinstance(x[k], tuple):
                    args.__dict__[k] = list(x[k])
                else:
                    args.__dict__[k] = [x[k]]
            else:
                args.__dict__[k] = x[k]
            logging.info(f"value in args is now: {args.__dict__[k]}\n")
    logging.info(f"Set arguments to: {args}")
    tmaps: Dict[str, TensorMap] = {}
    for tm in args.input_tensors + args.output_tensors:
        tmaps = update_tmaps(tm, tmaps)
    args.tensor_maps_in = [tmaps[it] for it in args.input_tensors]
    args.tensor_maps_out = [tmaps[ot] for ot in args.output_tensors]


def _string_from_architecture_dict(x: Arguments):
    return "\n".join([f"{k} = {x[k]}" for k in x])


def _trial_metric_and_param_label(
    i: int,
    all_losses: np.array,
    histories: List[Dict],
    trials: hyperopt.Trials,
    param_lists: Dict,
    aucs: List[Dict[str, Dict]],
) -> str:
    label = f"Trial {i}\n"
    for split, split_auc in aucs[i].items():
        no_idx = 0
        for idx, channel in enumerate(split_auc):
            if "no_" in channel:
                no_idx = idx

        for idx, (channel, auc) in enumerate(split_auc.items()):
            if len(split_auc) == 2 and no_idx == idx:
                continue
            label += f"{split.title()} {channel} AUC: {auc:.3f}\n"
    # fmt: off
    label += (
        f"Test Loss: {all_losses[i]:.3f}\n"
        f"Train Loss: {histories[i]['loss'][-1]:.3f}\n"
        f"Validation Loss: {histories[i]['val_loss'][-1]:.3f}\n"
        f"Model Parameter Count: {histories[i]['parameter_count'][-1]}\n"
    )
    # fmt: on
    label += _trial_parameter_string(trials, i, param_lists)
    return label


def _trial_parameter_string(
    trials: hyperopt.Trials,
    index: int,
    param_lists: Dict,
) -> str:
    label = ""
    params = trials.trials[index]["misc"]["vals"]
    for param in params:
        label += f"{param} = "
        value = params[param][0]
        if param in param_lists:
            label += str(param_lists[param][int(value)])
        elif param in ["num_layers", "layer_width"]:
            label += str(int(value))
        elif value < 1:
            label += f"{value:.2E}"
        else:
            label += f"{value:.2f}"
        label += "\n"
    return label


def _trial_metrics_and_params_to_df(
    all_losses: np.array,
    histories: List[Dict],
    trials: hyperopt.Trials,
    param_lists: Dict,
    trial_aucs: List[Dict[str, Dict]],
) -> pd.DataFrame:
    data: defaultdict = defaultdict(list)
    trial_aucs_test = []
    trial_aucs_train = []
    for trial_auc in trial_aucs:
        for split, split_auc in trial_auc.items():
            no_idx = 0
            for i, label in enumerate(split_auc):
                if "no_" in label:
                    no_idx = i

            for i, (label, auc) in enumerate(split_auc.items()):
                if len(split_auc) == 2 and no_idx == i:
                    continue

                if split == "test":
                    trial_aucs_test.append(auc)
                elif split == "train":
                    trial_aucs_train.append(auc)

    data.update(
        {
            "test_loss": all_losses,
            "train_loss": [history["loss"][-1] for history in histories],
            "valid_loss": [history["val_loss"][-1] for history in histories],
            "parameter_count": [
                history["parameter_count"][-1] for history in histories
            ],
            "test_auc": trial_aucs_test,
            "train_auc": trial_aucs_train,
        },
    )
    data.update(_trial_parameters_to_dict(trials, param_lists))
    df = pd.DataFrame(data)
    df.index.name = "Trial"
    return df


def _trial_parameters_to_dict(trials: hyperopt.Trials, param_lists: Dict) -> Dict:
    data = defaultdict(list)
    for trial in trials.trials:
        params = trial["misc"]["vals"]
        for param in params:
            value = params[param][0]
            if param in param_lists:
                value = param_lists[param][int(value)]
            elif param in ["num_layers", "layer_width"]:
                value = int(value)
            data[param].append(value)
    return data


def plot_trials(
    trials: hyperopt.Trials,
    histories: List[Dict],
    aucs: List[Dict[str, Dict]],
    figure_path: str,
    image_ext: str,
    param_lists: Dict = {},
):
    if not os.path.isdir(figure_path):
        os.makedirs(figure_path)
    all_losses = np.array(trials.losses())  # the losses we will put in the text
    real_losses = all_losses[all_losses != MAX_LOSS]
    cutoff = MAX_LOSS
    try:
        cutoff = threshold_otsu(real_losses)
    except ValueError:
        logging.info("Otsu thresholding failed. Using MAX_LOSS for threshold.")
    lplot = np.clip(all_losses, a_min=-np.inf, a_max=cutoff)  # the losses we will plot
    plt.figure(figsize=(64, 64))
    matplotlib.rcParams.update({"font.size": 9})
    colors = ["r" if x == cutoff else "b" for x in lplot]
    plt.plot(lplot)
    trial_metrics_and_params_df = _trial_metrics_and_params_to_df(
        all_losses,
        histories,
        trials,
        param_lists,
        aucs,
    )
    for col, dtype in trial_metrics_and_params_df.dtypes.items():
        if dtype == float:
            trial_metrics_and_params_df[col] = trial_metrics_and_params_df[col].apply(
                lambda x: f"{x:.3}",
            )
    metric_and_param_path = os.path.join(
        figure_path,
        "metrics_and_hyperparameters.csv",
    )
    trial_metrics_and_params_df.to_csv(metric_and_param_path)
    logging.info(f"Saved metric and hyperparameter table to {metric_and_param_path}")
    labels = [
        _trial_metric_and_param_label(
            i,
            all_losses,
            histories,
            trials,
            param_lists,
            aucs,
        )
        for i in range(len(trials.trials))
    ]
    for i, label in enumerate(labels):
        plt.text(i, lplot[i], label, color=colors[i])
    plt.xlabel("Iterations")
    plt.ylabel("Losses")
    plt.ylim(min(lplot) * 0.95, max(lplot) * 1.05)
    plt.title("Hyperparameter Optimization\n")
    plt.axhline(
        cutoff,
        label=f"Loss display cutoff at {cutoff:.3f}",
        color="r",
        linestyle="--",
    )
    loss_path = os.path.join(figure_path, "loss_per_trial" + image_ext)
    plt.legend()
    plt.savefig(loss_path)
    logging.info(f"Saved loss plot to {loss_path}")

    _, [ax1, ax3, ax2] = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(60, 20),
        sharey="all",
        gridspec_kw={"width_ratios": [2, 1, 2]},
    )
    cm = plt.get_cmap("gist_rainbow")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")
    linestyles = "solid", "dotted", "dashed", "dashdot"
    for i, history in enumerate(histories):
        color = cm(i / len(histories))
        training_loss = np.clip(history["loss"], a_min=-np.inf, a_max=cutoff)
        val_loss = np.clip(history["val_loss"], a_min=-np.inf, a_max=cutoff)
        label = labels[i]
        ax1.plot(training_loss, label=label, linestyle=linestyles[i % 4], color=color)
        ax1.text(len(training_loss) - 1, training_loss[-1], str(i))
        ax2.plot(val_loss, label=label, linestyle=linestyles[i % 4], color=color)
        ax2.text(len(val_loss) - 1, val_loss[-1], str(i))
    ax1.axhline(
        cutoff,
        label=f"Loss display cutoff at {cutoff:.3f}",
        color="k",
        linestyle="--",
    )
    ax1.set_title("Training Loss")
    ax2.axhline(
        cutoff,
        label=f"Loss display cutoff at {cutoff:.3f}",
        color="k",
        linestyle="--",
    )
    ax2.set_title("Validation Loss")
    ax3.legend(
        *ax2.get_legend_handles_labels(),
        loc="upper center",
        fontsize="x-small",
        mode="expand",
        ncol=5,
    )
    ax3.axis("off")
    learning_path = os.path.join(figure_path, "learning_curves_all_trials" + image_ext)
    plt.tight_layout()
    plt.savefig(learning_path)
    logging.info(f"Saved learning curve plot to {learning_path}")
