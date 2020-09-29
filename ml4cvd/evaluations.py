# Imports: standard library
import os
import logging
from typing import Dict, List, Tuple, Union, Optional
from collections import OrderedDict

# Imports: third party
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model

# Imports: first party
from ml4cvd.plots import subplot_rocs, subplot_scatters, evaluate_predictions
from ml4cvd.definitions import CSV_EXT, Path, Paths, Inputs, Outputs, Predictions
from ml4cvd.tensor_generators import (
    BATCH_INPUT_INDEX,
    BATCH_PATHS_INDEX,
    BATCH_OUTPUT_INDEX,
    TensorGenerator,
)
from ml4cvd.tensormap.TensorMap import TensorMap, find_negative_label_and_channel


def predict_and_evaluate(
    model: Model,
    data: Union[TensorGenerator, Tuple[Inputs, Outputs], Tuple[Inputs, Outputs, Paths]],
    tensor_maps_in: List[TensorMap],
    tensor_maps_out: List[TensorMap],
    plot_path: Path,
    data_split: str,
    save_coefficients: bool = False,
    steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    save_predictions: bool = False,
) -> Dict:
    """
    Evaluate model on dataset, save plots, and return performance metrics

    :param model: Model
    :param data: TensorGenerator or tuple of inputs, outputs, and optionally paths
    :param tensor_maps_in: Input maps
    :param tensor_maps_out: Output maps
    :param plot_path: Path to directory to save plots to
    :param data_split: Name of data split
    :param save_coefficients: Save model coefficients
    :param steps: Number of batches to use, required if data is a TensorGenerator
    :param batch_size: Number of samples to use in a batch, required if data is a tuple input and output numpy arrays
    :param save_predictions: If true, save predicted and actual output values to a csv

    :return: Dictionary of performance metrics
    """
    layer_names = [layer.name for layer in model.layers]
    performance_metrics = {}
    scatters = []
    rocs = []

    for tm in tensor_maps_out:
        if tm.output_name not in layer_names:
            raise ValueError(
                "Output tensor map name not found in layers of loaded model",
            )

    if save_coefficients:
        # Get coefficients from model layers
        coefficients = [c[0].round(3) for c in model.layers[-1].get_weights()[0]]

        # Get feature names from TMaps
        feature_names = []
        for tm in tensor_maps_in:
            # Use append to add single string to list
            if tm.channel_map is None:
                feature_names.append(tm.name)
            # Use extend to add list items to list
            else:
                feature_names.extend(tm.channel_map)

        if len(coefficients) != len(feature_names):
            raise ValueError("Number of coefficient values and names differ!")

        # Create dataframe of features
        df = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
        df = df.iloc[(-df["coefficient"]).argsort()].reset_index(drop=True)

        # Save dataframe
        fname = os.path.join(plot_path, "coefficients" + ".csv")
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        df.round(3).to_csv(path_or_buf=fname, index=False)

    y_predictions, output_data, data_paths = _get_predictions_from_data(
        model=model, data=data, steps=steps, batch_size=batch_size,
    )

    if save_predictions:
        save_data = OrderedDict()
        if data_paths is not None:
            save_data["sample_id"] = [
                os.path.splitext(os.path.basename(p))[0] for p in data_paths
            ]
        for y_prediction, tm in zip(y_predictions, tensor_maps_out):
            if tm.axes != 1:
                continue

            y_actual = tm.rescale(output_data[tm.output_name])
            y_prediction = tm.rescale(y_prediction)

            if tm.channel_map is not None:
                negative_label_idx = -1
                if len(tm.channel_map) == 2:
                    _, negative_label_idx = find_negative_label_and_channel(
                        tm.channel_map,
                    )
                for cm, idx in tm.channel_map.items():
                    if idx == negative_label_idx:
                        continue
                    save_data[f"{tm.name}_{cm}_actual"] = y_actual[..., idx]
                    save_data[f"{tm.name}_{cm}_predicted"] = y_prediction[..., idx]
            else:
                save_data[f"{tm.name}_actual"] = y_actual.flatten()
                save_data[f"{tm.name}_predicted"] = y_prediction.flatten()
        path = os.path.join(plot_path, f"predictions_{data_split}{CSV_EXT}")
        pd.DataFrame(save_data).round(6).to_csv(path, index=False)
        logging.info(f"Saved predictions at: {path}")

    for y, tm in zip(y_predictions, tensor_maps_out):
        if tm.output_name not in layer_names:
            continue
        y_truth = np.array(output_data[tm.output_name])
        performance_metrics.update(
            evaluate_predictions(
                tm=tm,
                y_predictions=y,
                y_truth=y_truth,
                title=tm.name,
                folder=plot_path,
                test_paths=data_paths,
                rocs=rocs,
                scatters=scatters,
                data_split=data_split,
            ),
        )

    if len(rocs) > 1:
        subplot_rocs(rocs, data_split, plot_path)
    if len(scatters) > 1:
        subplot_scatters(scatters, data_split, plot_path)

    return performance_metrics


def _get_predictions_from_data(
    model: Model,
    data: Union[TensorGenerator, Tuple[Inputs, Outputs], Tuple[Inputs, Outputs, Paths]],
    steps: Optional[int],
    batch_size: Optional[int],
) -> Tuple[Predictions, Outputs, Optional[Paths]]:
    """
    Get model predictions, output data, and paths from data source. If data source is a TensorGenerator, each sample
    in the dataset will be used no more than once. In the case where steps * batch_size > num_samples, each sample
    is used exactly once. If data source is a tuple of inputs and outputs, it is up to the user to provide the
    correct number of samples.

    :param model: Model
    :param data: TensorGenerator or tuple of inputs, outputs, and optionally paths
    :param steps: Number of batches to use, required if data is a TensorGenerator
    :param batch_size: Number of samples to use in a batch, required if data is a tuple input and output numpy arrays
    :return: Tuple of predictions as a list of numpy arrays, a dictionary of output data, and optionally paths
    """

    if isinstance(data, Tuple):
        if len(data) == 2:
            input_data, output_data = data
            paths = None
        elif len(data) == 3:
            input_data, output_data, paths = data
        else:
            raise ValueError(f"Expected 2 or 3 elements to data tuple, got {len(data)}")

        if batch_size is None:
            raise ValueError(
                f"When providing data as tuple of inputs and outputs, batch_size is required, got {batch_size}",
            )

        y_predictions = model.predict(input_data, batch_size=batch_size)
        if not isinstance(y_predictions, list):
            y_predictions = [y_predictions]
    elif isinstance(data, TensorGenerator):
        if steps is None:
            raise ValueError(
                f"When providing data as a generator, steps is required, got {steps}",
            )

        # no need for deterministic operation, we truncate after the first true epoch of
        # samples which is guaranteed to be unique, order within true epoch does not matter
        data.reset()
        batch_size = data.batch_size
        data_length = steps * batch_size
        y_predictions = [np.zeros((data_length,) + tm.shape) for tm in data.output_maps]
        output_data = {
            tm.output_name: np.zeros((data_length,) + tm.shape)
            for tm in data.output_maps
        }
        paths = [] if data.keep_paths else None
        for step in range(steps):
            start_idx = step * batch_size
            end_idx = start_idx + batch_size
            batch = next(data)

            # for single output models, prediction is an ndarray
            # for multi output models, predictions are a list of ndarrays
            batch_y_predictions = model.predict(batch[BATCH_INPUT_INDEX])
            if not isinstance(batch_y_predictions, list):
                batch_y_predictions = [batch_y_predictions]
            for i in range(len(y_predictions)):
                y_predictions[i][start_idx:end_idx] = batch_y_predictions[i]

            for output_name, output_tensor in batch[BATCH_OUTPUT_INDEX].items():
                output_data[output_name][start_idx:end_idx] = output_tensor

            if data.keep_paths:
                paths.extend(batch[BATCH_PATHS_INDEX])

            # truncate arrays to only use each sample exactly once
            if data.true_epoch_successful_samples is not None:
                num_samples = data.true_epoch_successful_samples
                if end_idx >= num_samples:
                    for i in range(len(y_predictions)):
                        y_predictions[i] = y_predictions[i][:num_samples]
                    for output_name in output_data:
                        # fmt: off
                        output_data[output_name] = output_data[output_name][:num_samples]
                        # fmt: on
                    if data.keep_paths:
                        paths = paths[:num_samples]
                    break
    else:
        raise NotImplementedError(
            f"Cannot get data for inference from data of type {type(data).__name__}: {data}",
        )

    # predictions returned by this function are lists of numpy arrays
    return y_predictions, output_data, paths
