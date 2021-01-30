# Imports: standard library
import os
import tempfile

# Imports: third party
import numpy as np
from data import get_pretraining_tasks
from hyperoptimize import (
    load_model,
    save_model,
    build_downstream_model,
    build_pretraining_model,
    get_optimizer_iterations,
)


def _build_test_model():
    return build_pretraining_model(
        ecg_length=100,
        kernel_size=3,
        group_size=2,
        depth=10,
        initial_width=8,
        width_growth_rate=2,
        width_quantization=1.5,
        learning_rate=1e-5,
    )


def _test_build_pretraining_model():
    m = _build_test_model()
    m.summary()
    dummy_in = {m.input_name: np.zeros((10, 100, 12))}
    dummy_out = {
        tm.output_name: np.zeros((10,) + tm.shape) for tm in get_pretraining_tasks()
    }
    history = m.fit(
        dummy_in,
        dummy_out,
        batch_size=2,
        validation_data=(dummy_in, dummy_out),
    )
    assert "val_loss" in history.history
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_model(tmpdir, m)
        m2 = load_model(path, m)
    history = m2.fit(
        dummy_in,
        dummy_out,
        batch_size=2,
        validation_data=(dummy_in, dummy_out),
    )
    assert get_optimizer_iterations(m) == 10
    assert "val_loss" in history.history


def _test_build_pretraining_model_bad_group_size():
    m = build_pretraining_model(
        ecg_length=2250,
        kernel_size=3,
        group_size=32,
        depth=13,
        initial_width=28,
        width_growth_rate=2.11,
        width_quantization=2.6,
        learning_rate=1e-5,
    )
    m.summary()
    dummy_in = {m.input_name: np.zeros((10, 100, 12))}
    dummy_out = {
        tm.output_name: np.zeros((10,) + tm.shape) for tm in get_pretraining_tasks()
    }
    history = m.fit(
        dummy_in,
        dummy_out,
        batch_size=2,
        validation_data=(dummy_in, dummy_out),
    )
    assert "val_loss" in history.history


def _test_build_downstream_model():
    m = build_downstream_model(
        downstream_tmap_name="age",
        ecg_length=100,
        kernel_size=3,
        group_size=2,
        depth=10,
        initial_width=8,
        width_growth_rate=2,
        width_quantization=1.5,
        learning_rate=1e-5,
    )
    m.summary()
    dummy_in = {m.input_name: np.zeros((10, 100, 12))}
    dummy_out = {"output_age_continuous": np.random.randn(10, 1)}
    history = m.fit(
        dummy_in,
        dummy_out,
        batch_size=2,
        validation_data=(dummy_in, dummy_out),
    )
    assert "val_loss" in history.history


def _test_build_downstream_model_pretrained():
    m = _build_test_model()
    m.summary()
    dummy_in = {m.input_name: np.zeros((10, 100, 12))}
    dummy_out = {
        tm.output_name: np.zeros((10,) + tm.shape) for tm in get_pretraining_tasks()
    }
    history = m.fit(
        dummy_in,
        dummy_out,
        batch_size=2,
        validation_data=(dummy_in, dummy_out),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(save_model(tmpdir, m), "pretraining_model.h5")
        downstream = build_downstream_model(
            downstream_tmap_name="age",
            model_file=path,
            ecg_length=100,
            kernel_size=3,
            group_size=2,
            depth=10,
            initial_width=8,
            width_growth_rate=2,
            width_quantization=1.5,
            learning_rate=1e-5,
        )
    rand_in = np.random.randn(1, 100, 12)
    np.testing.assert_allclose(  # did the pretrained layers get loaded?
        m.layers[0](rand_in),
        downstream.layers[0](rand_in),
    )
    downstream.summary()
    dummy_in = {m.input_name: np.zeros((10, 100, 12))}
    dummy_out = {"output_age_continuous": np.zeros((10, 1))}
    history = downstream.fit(
        dummy_in,
        dummy_out,
        batch_size=2,
        validation_data=(dummy_in, dummy_out),
    )
    assert "val_loss" in history.history


def _test_build_downstream_model_pretrained_freeze_weights():
    m = _build_test_model()
    m.summary()
    dummy_in = {m.input_name: np.zeros((10, 100, 12))}
    dummy_out = {
        tm.output_name: np.zeros((10,) + tm.shape) for tm in get_pretraining_tasks()
    }
    history = m.fit(
        dummy_in,
        dummy_out,
        batch_size=2,
        validation_data=(dummy_in, dummy_out),
    )
    # can we load pretrained models that were not frozen?
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(save_model(tmpdir, m), "pretraining_model.h5")
        downstream = build_downstream_model(
            downstream_tmap_name="age",
            model_file=path,
            ecg_length=100,
            kernel_size=3,
            group_size=2,
            depth=10,
            initial_width=8,
            width_growth_rate=2,
            width_quantization=1.5,
            learning_rate=1e-5,
            freeze_weights=True,
        )
    rand_in = np.random.randn(1, 100, 12)
    np.testing.assert_allclose(  # did the pretrained layers get loaded?
        m.layers[0](rand_in),
        downstream.layers[0](rand_in),
    )
    downstream.summary()
    dummy_in = {m.input_name: np.zeros((10, 100, 12))}
    dummy_out = {"output_age_continuous": np.zeros((10, 1))}
    history = downstream.fit(
        dummy_in,
        dummy_out,
        batch_size=2,
        validation_data=(dummy_in, dummy_out),
    )
    np.testing.assert_allclose(  # did the pretrained layers stay the same?
        m.layers[0](rand_in),
        downstream.layers[0](rand_in),
    )
    assert "val_loss" in history.history

    # can we load pretrained models that were frozen?
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(save_model(tmpdir, downstream), "pretraining_model.h5")
        new_downstream = build_downstream_model(
            downstream_tmap_name="age",
            model_file=path,
            ecg_length=100,
            kernel_size=3,
            group_size=2,
            depth=10,
            initial_width=8,
            width_growth_rate=2,
            width_quantization=1.5,
            learning_rate=1e-5,
            freeze_weights=True,
        )
    new_downstream.summary()
    dummy_in = {m.input_name: np.zeros((10, 100, 12))}
    dummy_out = {"output_age_continuous": np.zeros((10, 1))}
    history = new_downstream.fit(
        dummy_in,
        dummy_out,
        batch_size=2,
        validation_data=(dummy_in, dummy_out),
    )
    np.testing.assert_allclose(  # did the pretrained layers stay the same?
        m.layers[0](rand_in),
        downstream.layers[0](rand_in),
    )
    assert "val_loss" in history.history


if __name__ == "__main__":
    print(100 * "~")
    _test_build_pretraining_model()
    print(100 * "~")
    _test_build_pretraining_model_bad_group_size()
    print(100 * "~")
    _test_build_downstream_model()
    print(100 * "~")
    _test_build_downstream_model_pretrained()
    print(100 * "~")
    _test_build_downstream_model_pretrained_freeze_weights()
