# Imports: standard library
import json
import argparse
from typing import Any, Dict, List

# Imports: third party
import numpy as np

# Imports: first party
from tensormap.TensorMap import TensorMap, update_tmaps

# pylint: disable=dangerous-default-value


def generate_hyperoptimize_json(parameters: Dict[str, List[Any]], path_json: str):
    """
    Generates and saves a dictionary with parameter values to hyperoptimize as a
    .json file. This file is used by the hyperoptimize recipe.

    :param parameters: <Dict[str, List[Any]]> Dictionary with model parameters.
                       Keys are the name of the input parameters.
                       Items are list of possible parameter values.
    :param path_json: <str> Full path where the .json file will be saved.

    >>> parameters = {"dense_layers": [[10, 5], [40, 20]], "dropout": [0, 0.1]}
    >>> path_json = "./hyperoptimize_parameters.json"
    >>> generate_hyperoptimize_json(parameters=parameters, path_json=path_json)
    """
    if not path_json.endswith(".json"):
        path_json += ".json"
    with open(path_json, "w") as file:
        json.dump(parameters, file, indent=2)

    print("\n======= JSON string of hyperparameters =======\n")
    json_formatted_str = json.dumps(parameters, indent=2)
    print(json_formatted_str)
    print("\n======= End JSON string =======\n")
    print(f"Saved hyperoptimize json file at: {path_json}")


def generate_hyperoptimize_json_arrest(path_json: str, model_type: str):
    """
    Generates and saves a dictionary with parameter values to hyperoptimize as a
    .json file. This file is used by the hyperoptimize recipe.

    :param path_json: <str> Full path where the .json file will be saved.
    :param model_type: <str> Model type to create the set of hyperparameters.
    """
    tmaps: Dict[str, TensorMap] = {}

    output_tensors_set: List[List[str]] = [["arrest_double"]]
    for tmap_list in output_tensors_set:
        for tmap_name in tmap_list:
            tmaps = update_tmaps(tmap_name=tmap_name, tmaps=tmaps)

    windows = [
        (1, 24, 24),
        (1, 24, 48),
        (1, 24, 72),
        (1, 48, 24),
        (1, 48, 48),
        (1, 48, 72),
    ]
    signals = [
        "blood_pressure_systolic_value",
        "blood_pressure_diastolic_value",
        "respirations_value",
        "pulse_value",
        "temperature_value",
        "urine_output_value",
        "calcium_value",
        "sodium_value",
        "potassium_value",
        "chloride_value",
        "bun_value",
        "creatinine_value",
        "wbc_value",
        "hgb_value",
        "glucose_value",
        "anion_gap_value",
        "ppi_value",
        "ppt_value",
    ]
    features = [
        "min",
        "max",
        "mean",
        "std",
        "first",
        "last",
        "count",
        # "mean_slope",
        # "mean_crossing_rate",
    ]
    input_tensors_set: List[List[str]] = []
    for T1, T2, T3 in windows:
        input_tmaps = []
        window = (
            f"{T1}_hrs_pre_arrest_start_date_{T2}_hrs_post_admin_date_{T3}_hrs_window"
        )
        for signal in signals:
            for feature in features:
                input_tmaps.append(f"{signal}_{window}_{feature}")
        input_tmaps.append("age_first_visit_arrest_start_date_double")
        input_tmaps.append(f"length_of_stay_{T1}_hrs_pre_arrest_start_date")
        input_tensors_set.append(input_tmaps)
    for tmap_list in input_tensors_set:
        for tmap_name in tmap_list:
            tmaps = update_tmaps(tmap_name=tmap_name, tmaps=tmaps)

    parameters: Dict[str, List[Any]] = generate_model_dict(model_type)
    parameters.update(
        {
            "input_tensors": input_tensors_set,
            "output_tensors": output_tensors_set,
            "patient_csv": [
                "/media/ml4c3/cohorts_lists/rr-and-codes.csv",
                "/media/ml4c3/cohorts_lists/rr-and-codes-non-icu.csv",
            ],
        },
    )

    generate_hyperoptimize_json(parameters=parameters, path_json=path_json)


def generate_model_dict(model_type: str):
    if model_type == "deep_nn":
        parameters = generate_nn_dict()
    elif model_type == "keras_logreg":
        parameters = generate_keras_logreg_dict()
    elif model_type == "slearn_logreg":
        parameters = generate_sklearn_logreg_dict()
    elif model_type == "svm":
        parameters = generate_svm_dict()
    elif model_type == "randomforest":
        parameters = generate_randomforest_dict()
    elif model_type == "xgboost":
        parameters = generate_xgboost_dict()
    else:
        raise ValueError(f"Wrong model type: {model_type}")
    return parameters


def generate_sklearn_logreg_dict():
    parameters: Dict[str, List[Any]] = {
        "mode": ["train_sklearn_logreg"],
        # Model Architecture Parameters
        "l1": [0.0, 0.01, 0.02],
        "l2": [0.0, 0.01, 0.02],
        # Training Parameters
        "valid_ratio": [0.2],
        "test_ratio": [0.1],
    }
    return parameters


def generate_svm_dict():
    parameters: Dict[str, List[Any]] = {
        "mode": ["train_sklearn_svm"],
        # Model Architecture Parameters
        "c": [0.0, 0.01, 0.04, 0.08, 0.10],
        # Training Parameters
        "valid_ratio": [0.2],
        "test_ratio": [0.1],
    }
    return parameters


def generate_randomforest_dict():
    parameters: Dict[str, List[Any]] = {
        "mode": ["train_sklearn_randomforest"],
        # Model Architecture Parameters
        "n_estimators": [40, 60, 80, 100, 120, 140],
        "max_depth": [3, 5, 7, 9, 12],
        "min_samples_split": [5],
        "min_samples_leaf": [8],
        # Training Parameters
        "valid_ratio": [0.2],
        "test_ratio": [0.1],
    }
    return parameters


def generate_xgboost_dict():
    parameters: Dict[str, List[Any]] = {
        "mode": ["train_sklearn_xgboost"],
        # Model Architecture Parameters
        "gamma": [0.0, 0.02, 0.06, 0.1, 0.2],
        "l1": [0.0, 0.01, 0.02],
        "l2": [0.0, 0.01, 0.02],
        "n_estimators": [40, 60, 80, 100, 120, 140],
        "max_depth": [3, 5, 7, 9, 12],
        # Training Parameters
        "valid_ratio": [0.2],
        "test_ratio": [0.1],
    }
    return parameters


def generate_nn_dict():
    parameters: Dict[str, List[Any]] = {
        "mode": ["train"],
        # Model Architecture Parameters
        "activation": ["relu"],
        "block_size": [3],
        "conv_layers": [[32]],
        "conv_x": [[3]],
        "conv_y": [[3]],
        "conv_z": [[2]],
        "conv_dilate": [False],
        "conv_dropout": [0.0],
        "conv_regularize": ["dropout"],  # dropout, spatial_dropout
        "conv_type": ["conv"],  # conv, separable, depth
        "dense_blocks": [[32, 24, 16]],
        "dense_layers": [[16, 64]],
        "directly_embed_and_repeat": [None],  # None or int
        "dropout": [0.0, 0.05],
        "layer_normalization": [None],
        "layer_order": [["normalization", "activation", "regularization"]],
        "pool_after_final_dense_bloack": [True],
        "pool_type": ["max"],  # max, average
        "pool_x": [2],
        "pool_y": [2],
        "pool_z": [1],
        # Training Parameters
        "epochs": [200],
        "batch_size": [64],
        "valid_ratio": [0.2],
        "test_ratio": [0.1],
        "learning_rate": [0.0002],
        "learning_rate_patience": [8],
        "learning_rate_reduction": [0.5],
        "mixup_alpha": [0],
        "patience": [24],
        "optimizer": ["adam"],
        "learning_rate_schedule": [None],  # None, triangular, triangular2
        "anneal_rate": [0.0],
        "anneal_shift": [0.0],
        "anneal_max": [2.0],
    }
    return parameters


def generate_keras_logreg_dict():
    # TODO
    parameters = {}
    return parameters


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
    list_of_filters: List[List[int]] = []

    while len(list_of_filters) < num_unique_filters:

        # Generate length of filter sizes
        list_len = np.random.randint(
            low=list_len_bounds[0],
            high=list_len_bounds[1] + 1,
            size=1,
            dtype=int,
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

            # Randomly generate filter scale value by which to divide subsequent
            # filter sizes
            vary_filter_scale = np.random.uniform(
                low=vary_filter_scale_bounds[0],
                high=vary_filter_scale_bounds[1],
            )

            # Iterate through list of filter sizes
            this_filter = []

            for _ in range(list_len):
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
    base_tmap_name: str,
    weighted_losses: List[int],
) -> List[str]:
    new_tmap_names = [
        base_tmap_name + "_weighted_loss_" + str(weight) for weight in weighted_losses
    ]
    return new_tmap_names


def sample_random_hyperparameter(
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    scaling: str = "linear",
    primitive_type: np.dtype = float,
):
    if scaling not in ["linear", "logarithmic"]:
        raise ValueError(f"Invalid scaling parameter: {scaling}")

    if lower_bound >= upper_bound:
        print(f"Lower ({lower_bound}) and upper ({upper_bound}) bounds overlap")

    if scaling == "logarithmic":
        lower_bound = np.log(lower_bound)
        upper_bound = np.log(upper_bound)

    if primitive_type == float:
        sampled_value = np.random.uniform(low=lower_bound, high=upper_bound, size=1)
    elif primitive_type == int:
        sampled_value = np.random.randint(
            low=lower_bound,
            high=upper_bound + 1,
            size=1,
        )
    else:
        raise ValueError(f"Invalid primitive type: {primitive_type}")

    if scaling == "logarithmic":
        sampled_value = np.exp(sampled_value)

    # Return the value inside numpy array
    return sampled_value.item()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate hyperoptimize config json file.",
    )
    parser.add_argument(
        "--hyperoptimize_config_file",
        type=str,
        default="~/Dropbox/ml4c3_run_scripts/hyperoptimize.json",
        help="Full path to to .json file with the parameters to hyperoptimize. You can "
        "use the script generate_hyperoptimize_json.py to create it.",
    )
    parser.add_argument(
        "--cohort",
        type=str,
        choices=["arrest"],
        help="Cohort of patients that you want to create the json file for.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "deep_nn",
            "keras_logreg",
            "sklrean_logreg",
            "svm",
            "randomforest",
            "xgboost",
        ],
        help="Model type.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.cohort == "arrest":
        generate_hyperoptimize_json_arrest(args.hyperoptimize_config_file, args.model)
