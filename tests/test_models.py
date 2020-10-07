# Imports: standard library
import os
from typing import Dict, List, Tuple, Iterator, Optional
from itertools import cycle

# Imports: third party
import numpy as np
import pytest
import tensorflow as tf

# Imports: first party
from ml4cvd.models import (
    MODEL_EXT,
    ACTIVATION_FUNCTIONS,
    BottleneckType,
    make_multimodal_multitask_model,
)
from ml4cvd.tensormap.TensorMap import TensorMap

MEAN_PRECISION_EPS = 0.02  # how much mean precision degradation is acceptable
DEFAULT_PARAMS = {
    "activation": "relu",
    "dense_layers": [4, 2],
    "dense_blocks": [5, 3],
    "block_size": 3,
    "conv_width": 3,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "conv_type": "conv",
    "conv_layers": [6, 5, 3],
    "conv_x": [3],
    "conv_y": [3],
    "conv_z": [2],
    "padding": "same",
    "max_pools": [],
    "pool_type": "max",
    "pool_x": 1,
    "pool_y": 1,
    "pool_z": 1,
    "dropout": 0,
    "bottleneck_type": BottleneckType.FlattenRestructure,
    "layer_order": ["activation", "regularization", "normalization"],
}


TrainType = Dict[str, np.ndarray]  # TODO: better name


def make_training_data(
    input_tmaps: List[TensorMap], output_tmaps: List[TensorMap],
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
            input_tmaps, output_tmaps, **DEFAULT_PARAMS,
        )
    for tmap, tensor in zip(input_tmaps, m.inputs):
        assert tensor.shape[1:] == tmap.shape
        assert tensor.shape[1:] == tmap.shape
    for tmap, tensor in zip(output_tmaps, m.outputs):
        assert tensor.shape[1:] == tmap.shape
        assert tensor.shape[1:] == tmap.shape
    data = make_training_data(input_tmaps, output_tmaps)
    history = m.fit(
        x=data, steps_per_epoch=2, epochs=2, validation_data=data, validation_steps=2,
    )
    for tmap in output_tmaps:
        for metric in tmap.metrics:
            metric_name = metric if type(metric) == str else metric.__name__
            name = (
                f"{tmap.output_name}_{metric_name}"
                if len(output_tmaps) > 1
                else metric_name
            )
            assert name in history.history


def _rotate(a: List, n: int):
    return a[-n:] + a[:-n]


class TestMakeMultimodalMultitaskModel:
    @pytest.mark.parametrize(
        "input_output_tmaps",
        [
            (pytest.CONTINUOUS_TMAPS[:1], pytest.CONTINUOUS_TMAPS[1:2]),
            (pytest.CONTINUOUS_TMAPS[1:2], pytest.CONTINUOUS_TMAPS[:1]),
            (pytest.CONTINUOUS_TMAPS[:2], pytest.CONTINUOUS_TMAPS[:2]),
        ],
    )
    def test_multimodal_multitask_quickly(self, input_output_tmaps):
        """
        Tests 1d->2d, 2d->1d, (1d,2d)->(1d,2d)
        """
        assert_model_trains(input_output_tmaps[0], input_output_tmaps[1])

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "input_tmaps", pytest.MULTIMODAL_UP_TO_4D,
    )
    @pytest.mark.parametrize(
        "output_tmaps", pytest.MULTIMODAL_UP_TO_4D,
    )
    def test_multimodal(
        self, input_tmaps: List[TensorMap], output_tmaps: List[TensorMap],
    ):
        assert_model_trains(input_tmaps, output_tmaps)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "input_tmap", pytest.CONTINUOUS_TMAPS[:-1],
    )
    @pytest.mark.parametrize(
        "output_tmap", pytest.TMAPS_UP_TO_4D,
    )
    def test_unimodal_md_to_nd(self, input_tmap: TensorMap, output_tmap: TensorMap):
        assert_model_trains([input_tmap], [output_tmap])

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "input_tmap", pytest.TMAPS_UP_TO_4D,
    )
    @pytest.mark.parametrize(
        "output_tmap", pytest.TMAPS_UP_TO_4D,
    )
    def test_load_unimodal(self, tmpdir, input_tmap, output_tmap):
        m = make_multimodal_multitask_model(
            [input_tmap], [output_tmap], **DEFAULT_PARAMS,
        )
        path = os.path.join(tmpdir, f"m{MODEL_EXT}")
        m.save(path)
        make_multimodal_multitask_model(
            [input_tmap], [output_tmap], model_file=path, **DEFAULT_PARAMS,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "activation", ACTIVATION_FUNCTIONS.keys(),
    )
    def test_load_custom_activations(self, tmpdir, activation):
        inp, out = pytest.CONTINUOUS_TMAPS[:2], pytest.CATEGORICAL_TMAPS[:2]
        params = DEFAULT_PARAMS.copy()
        params["activation"] = activation
        m = make_multimodal_multitask_model(inp, out, **params)
        path = os.path.join(tmpdir, f"m{MODEL_EXT}")
        m.save(path)
        make_multimodal_multitask_model(
            inp, out, model_file=path, **params,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "input_tmaps", pytest.MULTIMODAL_UP_TO_4D,
    )
    @pytest.mark.parametrize(
        "output_tmaps", pytest.MULTIMODAL_UP_TO_4D,
    )
    def test_load_multimodal(
        self, tmpdir, input_tmaps: List[TensorMap], output_tmaps: List[TensorMap],
    ):
        m = make_multimodal_multitask_model(
            input_tmaps, output_tmaps, **DEFAULT_PARAMS,
        )
        path = os.path.join(tmpdir, f"m{MODEL_EXT}")
        m.save(path)
        make_multimodal_multitask_model(
            input_tmaps, output_tmaps, model_file=path, **DEFAULT_PARAMS,
        )

    @pytest.mark.parametrize(
        "input_output_tmaps",
        [
            (pytest.CONTINUOUS_TMAPS[:1], [pytest.SEGMENT_IN]),
            ([pytest.SEGMENT_IN], pytest.CONTINUOUS_TMAPS[:1]),
            ([pytest.SEGMENT_IN], [pytest.SEGMENT_IN]),
        ],
    )
    def test_multimodal_multitask_variational(self, input_output_tmaps, tmpdir):
        """
        Tests 1d->2d, 2d->1d, (1d,2d)->(1d,2d)
        """
        params = DEFAULT_PARAMS.copy()
        params["bottleneck_type"] = BottleneckType.Variational
        params["pool_x"] = params["pool_y"] = 2
        m = make_multimodal_multitask_model(
            input_output_tmaps[0], input_output_tmaps[1], **params
        )
        assert_model_trains(input_output_tmaps[0], input_output_tmaps[1], m)
        m.save(os.path.join(tmpdir, "vae.h5"))
        path = os.path.join(tmpdir, f"m{MODEL_EXT}")
        m.save(path)
        make_multimodal_multitask_model(
            input_output_tmaps[0],
            input_output_tmaps[1],
            model_file=path,
            **DEFAULT_PARAMS,
        )

    def test_no_dense_layers(self):
        params = DEFAULT_PARAMS.copy()
        params["dense_layers"] = []
        inp, out = pytest.CONTINUOUS_TMAPS[:2], pytest.CATEGORICAL_TMAPS[:2]
        m = make_multimodal_multitask_model(inp, out, **DEFAULT_PARAMS)
        assert_model_trains(inp, out, m)
