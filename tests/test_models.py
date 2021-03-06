# pylint: disable=no-member
# Imports: standard library
import os
from typing import Dict, List, Tuple, Iterator, Optional
from itertools import cycle

# Imports: third party
import numpy as np
import pytest
import tensorflow as tf

# Imports: first party
from ml4c3.models import MODEL_EXT, make_multimodal_multitask_model
from definitions.models import BottleneckType
from tensormap.TensorMap import TensorMap

MEAN_PRECISION_EPS = 0.02  # how much mean precision degradation is acceptable
DEFAULT_PARAMS = {
    "conv_type": "conv",
    "conv_blocks": [6],
    "conv_block_size": 1,
    "conv_block_layer_order": ["convolution", "activation", "dropout", "normalization"],
    "residual_blocks": [5, 3],
    "residual_block_size": 3,
    "residual_block_layer_order": [
        "convolution",
        "activation",
        "dropout",
        "normalization",
    ],
    "dense_blocks": [5, 3],
    "dense_block_size": 3,
    "dense_block_layer_order": [
        "normalization",
        "activation",
        "convolution",
        "dropout",
    ],
    "conv_x": [3],
    "conv_y": [3],
    "conv_z": [2],
    "conv_padding": "same",
    "pool_type": "max",
    "pool_x": 1,
    "pool_y": 1,
    "pool_z": 1,
    "bottleneck_type": BottleneckType.FlattenRestructure,
    "dense_layers": [4, 2],
    "activation_layer": "relu",
    "dense_layer_order": ["dense", "normalization", "activation", "dropout"],
    "optimizer": "adam",
    "learning_rate": 1e-3,
}

TrainType = Dict[str, np.ndarray]


def make_training_data(
    input_tmaps: List[TensorMap],
    output_tmaps: List[TensorMap],
) -> Iterator[Tuple[TrainType, TrainType, None]]:
    return cycle(
        [
            (
                {
                    tm.input_name: tf.random.normal((2,) + tm.shape)
                    for tm in input_tmaps
                },
                {tm.output_name: tf.zeros((2,) + tm.shape) for tm in output_tmaps},
                None,
            ),
        ],
    )


def assert_model_trains(
    input_tmaps: List[TensorMap],
    output_tmaps: List[TensorMap],
    m: Optional[tf.keras.Model] = None,
):
    if m is None:
        m = make_multimodal_multitask_model(
            tensor_maps_in=input_tmaps,
            tensor_maps_out=output_tmaps,
            **DEFAULT_PARAMS,
        )
    for tmap, tensor in zip(input_tmaps, m.inputs):
        assert tensor.shape[1:] == tmap.shape
    for tmap, tensor in zip(output_tmaps, m.outputs):
        assert tensor.shape[1:] == tmap.shape
    data = make_training_data(input_tmaps, output_tmaps)
    history = m.fit(
        x=data,
        steps_per_epoch=2,
        epochs=2,
        validation_data=data,
        validation_steps=2,
    )
    for tmap in output_tmaps:
        for metric in tmap.metrics:
            metric_name = metric if isinstance(metric, str) else metric.__name__
            name = (
                f"{tmap.output_name}_{metric_name}"
                if len(output_tmaps) > 1
                else metric_name
            )
            assert name in history.history


class TestMakeMultimodalMultitaskModel:
    @pytest.mark.parametrize(
        "input_output_tmaps",
        [
            (pytest.CONTINUOUS_TMAPS[:1], pytest.CONTINUOUS_TMAPS[1:2]),
            (pytest.CONTINUOUS_TMAPS[1:2], pytest.CONTINUOUS_TMAPS[:1]),
            (pytest.CONTINUOUS_TMAPS[:2], pytest.CONTINUOUS_TMAPS[:2]),
        ],
    )
    def test_multimodal_multitask_quickly(
        self,
        input_output_tmaps: Tuple[List[TensorMap], List[TensorMap]],
    ):
        """
        Tests 1d->2d, 2d->1d, (1d,2d)->(1d,2d)
        """
        assert_model_trains(input_output_tmaps[0], input_output_tmaps[1])

    @pytest.mark.parametrize(
        "input_output_tmaps",
        [
            (pytest.CONTINUOUS_TMAPS[:1], [pytest.SEGMENT_IN]),
            ([pytest.SEGMENT_IN], pytest.CONTINUOUS_TMAPS[:1]),
            ([pytest.SEGMENT_IN], [pytest.SEGMENT_IN]),
        ],
    )
    def test_multimodal_multitask_variational(
        self,
        input_output_tmaps: Tuple[List[TensorMap], List[TensorMap]],
        tmpdir,
    ):
        """
        Tests 1d->2d, 2d->1d, (1d,2d)->(1d,2d)
        """
        params = DEFAULT_PARAMS.copy()
        params["bottleneck_type"] = BottleneckType.Variational
        params["pool_x"] = params["pool_y"] = 2
        m = make_multimodal_multitask_model(
            tensor_maps_in=input_output_tmaps[0],
            tensor_maps_out=input_output_tmaps[1],
            **params,
        )
        assert_model_trains(input_output_tmaps[0], input_output_tmaps[1], m)
        m.save(os.path.join(tmpdir, "vae.h5"))
        path = os.path.join(tmpdir, f"m{MODEL_EXT}")
        m.save(path)
        make_multimodal_multitask_model(
            tensor_maps_in=input_output_tmaps[0],
            tensor_maps_out=input_output_tmaps[1],
            model_file=path,
            **DEFAULT_PARAMS,
        )

    def test_no_dense_layers(self):
        params = DEFAULT_PARAMS.copy()
        params["dense_layers"] = []
        inp, out = pytest.CONTINUOUS_TMAPS[:2], pytest.CATEGORICAL_TMAPS[:2]
        m = make_multimodal_multitask_model(
            tensor_maps_in=inp, tensor_maps_out=out, **DEFAULT_PARAMS
        )
        assert_model_trains(inp, out, m)
