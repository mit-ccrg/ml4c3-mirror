# Imports: standard library
import json
import logging
from typing import Any, Dict, List

# Imports: third party
import numpy as np

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
        path_json += "json"
    with open(path_json, "w") as file:
        json.dump(parameters, file)


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
