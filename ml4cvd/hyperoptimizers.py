# Imports: standard library
import gc
import os
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
from ml4cvd.plots import plot_metric_history
from ml4cvd.models import train_model_from_datasets, make_multimodal_multitask_model
from ml4cvd.datasets import train_valid_test_datasets
from ml4cvd.definitions import Arguments
from ml4cvd.evaluations import predict_and_evaluate
from ml4cvd.tensormap.TensorMap import TensorMap, update_tmaps

# fmt: off
# need matplotlib -> Agg -> pyplot
import matplotlib                       # isort:skip
matplotlib.use("Agg")                   # isort:skip
from matplotlib import pyplot as plt    # isort:skip
# fmt: on


MAX_LOSS = 9e9


def hyperoptimize(args: argparse.Namespace):
    """
    hyperopt is a Python library that performs Bayesian optimization over hyperparameters
    to minimize an objective function. Here, the objective function is
    loss_from_multimodal_multitask.
    Hyperparameter combinations are randomly chosen and non-unique choices are skipped
    before model compilation. The computation to skip repeated combinations is fast and
    inexpensive. However, each non-unique combination counts towards the maximum number
    of models to evaluate. If a grid search over a relatively small search space is
    desired, set max_evals >> size of search space. In this case, it is likely, but not
    guaranteed, that all combinations will be seen.
    """
    block_size_sets = [2, 3, 4]
    conv_layers_sets = [[32]]
    conv_normalize_sets = [""]
    dense_layers_sets = [
        [10, 5],
        [40, 20],
        [30, 30],
        [64, 16],
        [100, 50],
    ]
    dense_blocks_sets = [
        [32, 24, 16],
    ]
    pool_types = ["max", "average"]
    conv_regularize_sets = ["spatial_dropout"]
    conv_dropout_sets = [0.5]
    dropout_sets = [0, 0.1, 0.2, 0.3]
    conv_x_sets = _generate_conv1D_filter_widths(
        num_unique_filters=6,
        list_len_bounds=[1, 1],
        first_filter_width_bounds=[6, 200],
        probability_vary_filter_width=0,
    )
    learning_rate_sets = [0.001, 0.005, 0.01]

    # Initialize empty dict of tmaps
    tmaps: Dict[str, TensorMap] = {}

    # Generate weighted loss tmaps for STS death
    weighted_losses = [val for val in range(1, 10, 2)]
    output_tensors_sets = _generate_weighted_loss_tmaps(
        base_tmap_name="sts_death", weighted_losses=weighted_losses,
    )
    for tmap_name in output_tensors_sets:
        tmaps = update_tmaps(tmap_name=tmap_name, tmaps=tmaps)

    input_tmap_sets = [
        "ecg_2500_std_preop_newest",
        "ecg_age_std_preop_newest",
        "ecg_sex_preop_newest",
    ]
    for tmap_name_or_list in input_tmap_sets:
        if isinstance(tmap_name_or_list, list):
            for tmap_name in tmap_name_or_list:
                tmaps = update_tmaps(tmap_name=tmap_name, tmaps=tmaps)
        elif isinstance(tmap_name_or_list, str):
            tmaps = update_tmaps(tmap_name=tmap_name_or_list, tmaps=tmaps)

    space = {
        # "block_size": hp.choice("block_size", block_size_sets),
        # "conv_dropout": hp.choice("conv_dropout", conv_dropout_sets),
        # "conv_normalize": hp.choice("conv_normalize", conv_normalize_sets),
        # "conv_regularize": hp.choice("conv_regularize", conv_regularize_sets),
        # "conv_x": hp.choice("conv_x", conv_x_sets),
        # "dense_blocks": hp.choice("dense_blocks", dense_blocks_sets),
        "dense_layers": hp.choice("dense_layers", dense_layers_sets),
        "dropout": hp.choice("dropout", dropout_sets),
        "output_tensors": hp.choice("output_tensors", output_tensors_sets),
        # "input_tensors": hp.choice("input_tensors", input_tmap_sets),
        "learning_rate": hp.choice("learning_rate", learning_rate_sets),
        # "pool_type": hp.choice("pool_type", pool_types),
    }
    param_lists = {
        # "block_size": block_size_sets,
        # "conv_x": conv_x_sets,
        # "conv_dropout": conv_dropout_sets,
        # "conv_normalize": conv_normalize_sets,
        # "conv_regularize": conv_regularize_sets,
        # "dense_blocks": dense_blocks_sets,
        "dense_layers": dense_layers_sets,
        "dropout": dropout_sets,
        "output_tensors": output_tensors_sets,
        # "input_tensors": input_tmap_sets,
        "learning_rate": learning_rate_sets,
        # "pool_type": pool_types,
    }
    hyperparameter_optimizer(args=args, space=space, param_lists=param_lists)


def hyperparameter_optimizer(
    args: argparse.Namespace,
    space: Dict[str, hyperopt.pyll.base.Apply],
    param_lists: Arguments,
):
    args.keep_paths = False
    args.keep_paths_test = False
    histories = []
    aucs = []
    results_path = os.path.join(args.output_folder, args.id)
    i = 0
    seen_combinations = set()

    def loss_from_multimodal_multitask(x: Arguments):
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
                    f"Trial {trial_id}: hyperparameter combination is non-unique: {params}",
                )
            seen_combinations.add(params)

            set_args_from_x(args, x)
            model = make_multimodal_multitask_model(**args.__dict__)

            if model.count_params() > args.max_parameters:
                logging.info(
                    f"Model too big, max parameters is:{args.max_parameters}, model"
                    f" has:{model.count_params()}. Return max loss.",
                )
                return MAX_LOSS

            datasets, stats, cleanups = train_valid_test_datasets(
                tensor_maps_in=args.tensor_maps_in,
                tensor_maps_out=args.tensor_maps_out,
                tensors=args.tensors,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sample_csv=args.sample_csv,
                valid_ratio=args.valid_ratio,
                test_ratio=args.test_ratio,
                train_csv=args.train_csv,
                valid_csv=args.valid_csv,
                test_csv=args.test_csv,
                output_folder=args.output_folder,
                run_id=args.id,
            )
            train_dataset, valid_dataset, test_dataset = datasets
            model, history = train_model_from_datasets(
                model=model,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                epochs=args.epochs,
                patience=args.patience,
                learning_rate_patience=args.learning_rate_patience,
                learning_rate_reduction=args.learning_rate_reduction,
                output_folder=trials_path,
                run_id=trial_id,
                return_history=True,
                plot=False,
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
                history, None, "", os.path.join(trials_path, trial_id),
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
        fn=loss_from_multimodal_multitask,
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
    tmaps = {}
    for tm in args.input_tensors + args.output_tensors:
        tmaps = update_tmaps(tm, tmaps)
    args.tensor_maps_in = [tmaps[it] for it in args.input_tensors]
    args.tensor_maps_out = [tmaps[ot] for ot in args.output_tensors]


def _ensure_even_number(num: int) -> int:
    if num % 2 == 1:
        num += 1
    return num


def _generate_conv1D_filter_widths(
    num_unique_filters: int = 25,
    list_len_bounds: List[int] = [5, 5],
    first_filter_width_bounds: List[int] = [50, 150],
    probability_vary_filter_width: float = 0.5,
    vary_filter_scale_bounds: List[float] = [1.25, 1.75],
) -> List[List[int]]:
    """Generate a list of 1D convolutional filter widths that are lists of even ints.

    :param num_unique_filters: number of unique lists of filters to generate,
        e.g. 10 will result in a list of 10 lists of filter widths.

    :param list_len_bounds: bounds of the number of elements in each list of filters;
        the number of elements is a randomly selected integer in these bounds.
        e.g. [1, 4] will choose a random int from among 1, 2, 3, or 4.

    :param first_filter_width_bounds: bounds of the first filter width; randomly
        selected integer in these bounds similar to 'list_len_bounds'.

    :param probability_vary_filter_width: probability of choosing to vary filter size;
        a randomly generated float between 0-1 is compared to this value. If <=, then
        the filter size is varied.

    :param vary_filter_scale_bounds: bounds of the scale factor for decreasing filter
        width in subsequent layers; the scale is a randomly selected float in these
        bounds. The filter width of the next layer equals the filter width of the prior
        layer divided by the filter_scale. This scale factor is applied to all layers:
           ```
           list_len = 4
           first_filter_width = 100
           filter_scale = 1.5
           ```
        These settings would result in the following filter widths: [100, 66, 44, 30]
    """
    list_of_filters = []

    while len(list_of_filters) < num_unique_filters:

        # Generate length of filter sizes
        list_len = np.random.randint(
            low=list_len_bounds[0], high=list_len_bounds[1] + 1, size=1, dtype=int,
        )[0]

        # Generate first filter size
        first_filter_width = np.random.randint(
            low=first_filter_width_bounds[0],
            high=first_filter_width_bounds[1] + 1,
            size=1,
            dtype=int,
        )[0]
        first_filter_width = _ensure_even_number(first_filter_width)

        # Randomly determine if filter size varies or not
        if probability_vary_filter_width >= np.random.rand():

            # Randomly generate filter scale value by which to divide subsequent filter sizes
            vary_filter_scale = np.random.uniform(
                low=vary_filter_scale_bounds[0], high=vary_filter_scale_bounds[1],
            )

            # Iterate through list of filter sizes
            this_filter = []

            for i in range(list_len):
                this_filter.append(first_filter_width)

                # Check if we want to vary filter size
                current_filter_width = first_filter_width
                first_filter_width = int(first_filter_width / vary_filter_scale)
                first_filter_width = _ensure_even_number(first_filter_width)

                # If reducing filter size makes it 0, reset to prior filter size
                if first_filter_width == 0:
                    first_filter_width = current_filter_width

            if this_filter not in list_of_filters:
                list_of_filters.append(this_filter)

        # Else the filter size is constant
        else:
            list_of_filters.append([first_filter_width])

    return list_of_filters


def _generate_weighted_loss_tmaps(
    base_tmap_name: str, weighted_losses: List[int],
) -> List[str]:
    new_tmap_names = [
        base_tmap_name + "_weighted_loss_" + str(weight) for weight in weighted_losses
    ]
    return new_tmap_names


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
    trials: hyperopt.Trials, index: int, param_lists: Dict,
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
    data = defaultdict(list)
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
        all_losses, histories, trials, param_lists, aucs,
    )
    for col, dtype in trial_metrics_and_params_df.dtypes.items():
        if dtype == float:
            trial_metrics_and_params_df[col] = trial_metrics_and_params_df[col].apply(
                lambda x: f"{x:.3}",
            )
    metric_and_param_path = os.path.join(
        figure_path, "metrics_and_hyperparameters.csv",
    )
    trial_metrics_and_params_df.to_csv(metric_and_param_path)
    logging.info(f"Saved metric and hyperparameter table to {metric_and_param_path}")
    labels = [
        _trial_metric_and_param_label(
            i, all_losses, histories, trials, param_lists, aucs,
        )
        for i in range(len(trials.trials))
    ]
    for i, label in enumerate(labels):
        plt.text(i, lplot[i], label, color=colors[i])
    plt.xlabel("Iterations")
    plt.ylabel("Losses")
    plt.ylim(min(lplot) * 0.95, max(lplot) * 1.05)
    plt.title(f"Hyperparameter Optimization\n")
    plt.axhline(
        cutoff, label=f"Loss display cutoff at {cutoff:.3f}", color="r", linestyle="--",
    )
    loss_path = os.path.join(figure_path, "loss_per_trial" + image_ext)
    plt.legend()
    plt.savefig(loss_path)
    logging.info(f"Saved loss plot to {loss_path}")

    fig, [ax1, ax3, ax2] = plt.subplots(
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
        cutoff, label=f"Loss display cutoff at {cutoff:.3f}", color="k", linestyle="--",
    )
    ax1.set_title("Training Loss")
    ax2.axhline(
        cutoff, label=f"Loss display cutoff at {cutoff:.3f}", color="k", linestyle="--",
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


def sample_random_hyperparameter(
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    scaling: str = "linear",
    primitive_type: np.dtype = float,
):
    if scaling not in ["linear", "logarithmic"]:
        raise ValueError(f"Invalid scaling parameter: {scaling}")

    if lower_bound >= upper_bound:
        logging.warning(
            f"Lower ({lower_bound}) and upper ({upper_bound}) bounds overlap",
        )

    if scaling == "logarithmic":
        lower_bound = np.log(lower_bound)
        upper_bound = np.log(upper_bound)

    if primitive_type == float:
        sampled_value = np.random.uniform(low=lower_bound, high=upper_bound, size=1)
    elif primitive_type == int:
        sampled_value = np.random.randint(
            low=lower_bound, high=upper_bound + 1, size=1,
        )
    else:
        raise ValueError(f"Invalid primitive type: {primitive_type}")

    if scaling == "logarithmic":
        sampled_value = np.exp(sampled_value)

    # Return the value inside numpy array
    return sampled_value.item()
