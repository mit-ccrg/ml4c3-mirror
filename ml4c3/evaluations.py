# Imports: standard library
import os
import logging
from typing import Dict, List, Tuple, Union, Optional
from collections import OrderedDict, defaultdict

# Imports: third party
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model

# Imports: first party
from ml4c3.plots import subplot_rocs, subplot_scatters, evaluate_predictions
from ml4c3.models import SKLEARN_MODELS
from ml4c3.datasets import (
    BATCH_INPUT_INDEX,
    BATCH_PATHS_INDEX,
    BATCH_OUTPUT_INDEX,
    get_array_from_dict_of_arrays,
    get_dicts_of_arrays_from_dataset,
)
from ml4c3.definitions.types import Path, Paths, Inputs, Outputs, Predictions
from ml4c3.definitions.globals import CSV_EXT
from ml4c3.tensormap.TensorMap import TensorMap, find_negative_label_and_channel


def predict_and_evaluate(
    model: Union[Model, SKLEARN_MODELS],
    data: Union[
        tf.data.Dataset,
        Tuple[Inputs, Outputs],
        Tuple[Inputs, Outputs, Paths],
    ],
    tensor_maps_in: List[TensorMap],
    tensor_maps_out: List[TensorMap],
    plot_path: Path,
    data_split: str,
    image_ext: str,
    save_coefficients: bool = False,
    batch_size: Optional[int] = None,
    save_predictions: bool = False,
) -> Dict:
    """
    Evaluate trained model on dataset, save plots, and return performance metrics

    :param model: Model
    :param data: tensorflow Dataset or tuple of inputs, outputs, and optionally paths
    :param tensor_maps_in: Input maps
    :param tensor_maps_out: Output maps
    :param plot_path: Path to directory to save plots to
    :param data_split: Name of data split
    :param save_coefficients: Save model coefficients
    :param batch_size: Number of samples to use in a batch, required if data is a
                       tuple of input and output numpy arrays
    :param save_predictions: If true, save predicted and actual output values to a csv

    :return: Dictionary of performance metrics
    """
    performance_metrics = {}
    scatters = []
    rocs = []
    if isinstance(model, Model):
        layer_names = [layer.name for layer in model.layers]
        for tm in tensor_maps_out:
            if tm.output_name not in layer_names:
                raise ValueError(
                    "Output tensor map name not found in layers of loaded model",
                )
    if save_coefficients:
        if isinstance(model, Model):
            coefficients = [c[0].round(3) for c in model.layers[-1].get_weights()[0]]
        else:
            coefficients = _get_sklearn_model_coefficients(model=model)

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
        model=model,
        data=data,
        batch_size=batch_size,
        tensor_maps_in=tensor_maps_in,
        tensor_maps_out=tensor_maps_out,
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
        if isinstance(model, Model):
            if tm.output_name not in layer_names:
                continue
        y_truth = np.array(output_data[tm.output_name])

        performance_metrics.update(
            evaluate_predictions(
                tm=tm,
                y_predictions=y,
                y_truth=y_truth,
                title=tm.name,
                image_ext=image_ext,
                folder=plot_path,
                test_paths=data_paths,
                rocs=rocs,
                scatters=scatters,
                data_split=data_split,
            ),
        )
    if len(rocs) > 1:
        subplot_rocs(
            rocs=rocs,
            data_split=data_split,
            image_ext=image_ext,
            plot_path=plot_path,
        )
    if len(scatters) > 1:
        subplot_scatters(
            scatters=scatters,
            data_split=data_split,
            image_ext=image_ext,
            plot_path=plot_path,
        )
    return performance_metrics


def _get_sklearn_model_coefficients(model: SKLEARN_MODELS) -> np.array:
    if model.name == "logreg":
        return model.coef_.flatten()
    elif model.name == "svm":
        return model.LSVC.coef_.flatten()
    elif model.name == "randomforest" or model.name == "xgboost":
        return model.feature_importances_.flatten()
    else:
        raise ValueError(f"{model.name} lacks feature coefficients or importances")


def _get_predictions_from_data(
    model: Union[Model, SKLEARN_MODELS],
    data: Union[
        tf.data.Dataset,
        Tuple[Inputs, Outputs],
        Tuple[Inputs, Outputs, Paths],
    ],
    batch_size: Optional[int],
    tensor_maps_in: Optional[List[TensorMap]],
    tensor_maps_out: Optional[List[TensorMap]],
) -> Tuple[Predictions, Outputs, Optional[Paths]]:
    """
    Get model predictions, output data, and paths from data source. Data must not
    be infinite.

    :param model: Model
    :param data: finite tensorflow Dataset or tuple of inputs, outputs, and
                 optionally paths
    :param batch_size: Number of samples to use in a batch, required if data is a
                       tuple input and output numpy arrays
    :return: Tuple of predictions as a list of numpy arrays, a dictionary of
             output data, and optionally paths
    """
    if isinstance(data, Tuple):
        if len(data) == 2:
            input_data, output_data = data
            paths = None
        elif len(data) == 3:
            input_data, output_data, paths = data
        else:
            raise ValueError(
                f"Expected 2 or 3 elements to dataset tuple, got {len(data)}",
            )
        if batch_size is None:
            raise ValueError(
                "When providing dataset as tuple of inputs and outputs, batch_size "
                "is required, got {batch_size}",
            )
        y_predictions = model.predict(x=input_data, batch_size=batch_size)

    elif isinstance(data, tf.data.Dataset):
        y_prediction_batches = defaultdict(list)
        output_data_batches = defaultdict(list)
        path_batches = []

        if isinstance(model, Model):
            for batch in data:
                output_data_batch = batch[BATCH_OUTPUT_INDEX]
                for output_name, output_tensor in output_data_batch.items():
                    output_data_batches[output_name].append(output_tensor.numpy())

                batch_y_predictions = model.predict(batch[BATCH_INPUT_INDEX])
                if not isinstance(batch_y_predictions, list):
                    batch_y_predictions = [batch_y_predictions]

                for prediction_idx, batch_y_prediction in enumerate(
                    batch_y_predictions,
                ):
                    y_prediction_batches[prediction_idx].append(batch_y_prediction)

                if len(batch) == 3:
                    path_batches.append(batch[BATCH_PATHS_INDEX].numpy().astype(str))

            y_predictions = [
                np.concatenate(y_prediction_batches[prediction_idx])
                for prediction_idx in sorted(y_prediction_batches)
            ]

        elif isinstance(model, SKLEARN_MODELS.__args__):
            data = get_dicts_of_arrays_from_dataset(dataset=data)
            assert all(tm.axes == 1 for tm in tensor_maps_in + tensor_maps_out)
            assert len(tensor_maps_out) == 1

            # Isolate arrays from datasets for desired tensor maps
            X = get_array_from_dict_of_arrays(
                tensor_maps=tensor_maps_in,
                data=data[BATCH_INPUT_INDEX],
                drop_redundant_columns=False,
            )
            y_predictions = model.predict_proba(X)

            for output_name, output_tensor in data[BATCH_OUTPUT_INDEX].items():
                output_data_batches[output_name].append(output_tensor)

            if len(data) == 3:
                path_batches.append(data[BATCH_PATHS_INDEX])

        else:
            raise NotImplementedError(
                f"Cannot perform inference on model of type {type(model).__name}",
            )

        # Iterate over batches and concatenate into dict of arrays
        output_data = {
            output_name: np.concatenate(output_data_batches[output_name])
            for output_name in output_data_batches
        }
        paths = (
            None if len(path_batches) == 0 else np.concatenate(path_batches).tolist()
        )
    else:
        raise NotImplementedError(
            "Cannot get data for inference from data of type "
            "{type(data).__name__}: {data}",
        )

    if not isinstance(y_predictions, list):
        y_predictions = [y_predictions]

    return y_predictions, output_data, paths
