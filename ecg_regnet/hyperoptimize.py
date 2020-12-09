# Imports: standard library
import os
import pickle
import argparse
import tempfile
from typing import List, DefaultDict
from collections import defaultdict

# Imports: third party
import numpy as np
import pandas as pd
from ray import tune
from data import (
    get_ecg_tmap,
    augmentation_dict,
    get_pretraining_tasks,
    get_pretraining_datasets,
)
from regnet import ML4C3Regnet
from tensorflow import config as tf_config
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from tensorflow.keras.layers import Input
from tensorflow_addons.optimizers import SGDW
from tensorflow.keras.experimental import CosineDecay
from tensorflow.python.keras.utils.layer_utils import count_params

STEPS_PER_EPOCH = 100
VALIDATION_STEPS_PER_EPOCH = 10
BATCH_SIZE = 128  # following RegNet


class EarlyStopping(tune.stopper.Stopper):
    """Stop a trial early once its validation loss has plateaued"""

    def __init__(
        self,
        patience: int,  # number of epochs to wait before stopping
    ):
        self.patience = patience
        self._trial_to_loss: DefaultDict[str, List[float]] = defaultdict(list)

    def __call__(self, trial_id, result) -> bool:
        """
        If patience is 5, and the loss each epoch is [1, 2, 2, 2, 2],
        the model will stop training the __next__ epoch if there is still no improvement,
        since in the current iteration the following evaluates to false:
        (argmin(loss) = 0) + (patience = 5) < (len(losses) = 5)
        """
        if np.isnan(result["val_loss"]):
            print(f"Stopping {trial_id} for NaN loss")
            return True
        losses = self._trial_to_loss[trial_id]
        losses.append(result["val_loss"])
        best_loss_idx = np.argmin(losses)
        if best_loss_idx + self.patience < len(losses):
            print(
                f"Stopping {trial_id} for no improvement after {self.patience} trials.",
            )
            return True
        return False

    def stop_all(self) -> bool:
        return False


def build_pretraining_model(
    ecg_length: int,
    kernel_size: int,
    group_size: int,
    depth: int,
    initial_width: int,
    width_growth_rate: int,
    width_quantization: float,
    decay_steps: int,  # optimizer settings
    learning_rate: float,
    **kwargs,
):
    tmaps_out = get_pretraining_tasks()
    input_name = get_ecg_tmap(0, []).input_name
    output_name_to_shape = {tmap.output_name: tmap.shape[0] for tmap in tmaps_out}
    model = ML4C3Regnet(
        kernel_size,
        group_size,
        depth,
        initial_width,
        width_growth_rate,
        width_quantization,
        input_name,
        output_name_to_shape,
    )
    lr_schedule = CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
    )
    optimizer = SGDW(
        learning_rate=lr_schedule,
        momentum=0.9,
        weight_decay=5 * 5e-5,
    )  # following RegNet's setup
    model({input_name: Input(shape=(ecg_length, 12))})  # initialize model
    model.compile(loss=[tm.loss for tm in tmaps_out], optimizer=optimizer)
    return model


def save_optimizer(folder: str, optimizer):
    with open(os.path.join(folder, "optimizer.pkl"), "wb") as f:
        pickle.dump(optimizer, f)


def load_optimizer(folder: str, optimizer):
    with open(os.path.join(folder, "optimizer.pkl"), "rb") as f:
        return pickle.load(f)


def save_model(folder: str, model) -> str:
    checkpoint_path = os.path.join(folder, "pretraining_model.h5")
    model.save_weights(checkpoint_path)
    save_optimizer(folder, model.optimizer)
    return folder


def load_model(folder: str, model):
    checkpoint_path = os.path.join(folder, "pretraining_model.h5")
    model.load_weights(checkpoint_path)
    optimizer = load_optimizer(folder, model.optimizer)
    model.compile(optimizer=optimizer, loss=model.loss)
    return model


def get_optimizer_iterations(model) -> int:
    return model.optimizer.iterations.numpy()


def get_optimizer_lr(model) -> float:
    """Assumes optimizer is using a learning rate schedule"""
    iters = get_optimizer_iterations(model)
    return model.optimizer.learning_rate(iters).numpy()


def _build_test_model():
    return build_pretraining_model(
        decay_steps=10,
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
    assert np.isclose(get_optimizer_lr(m), 1e-5)
    history = m.fit(
        dummy_in,
        dummy_out,
        batch_size=2,
        validation_data=(dummy_in, dummy_out),
    )
    iters = get_optimizer_iterations(m)
    lr = get_optimizer_lr(m)
    assert "val_loss" in history.history
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_model(tmpdir, m)
        m2 = load_model(path, m)
    assert np.isclose(
        get_optimizer_lr(m),
        lr,
    )  # does the learning rate stay the same after loading?
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
        decay_steps=10,
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


class PretrainingTrainable(tune.Trainable):
    def setup(self, config):
        # Imports: third party
        import tensorflow as tf  # necessary for ray tune

        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(
                gpu,
                True,
            )  # do not allocate all memory right away

        augmentation_strengths = {
            name.replace("_strength", ""): strength
            for name, strength in config.items()
            if "strength" in name
        }

        datasets, stats, cleanups = get_pretraining_datasets(
            ecg_length=config["ecg_length"],
            augmentation_strengths=augmentation_strengths,
            num_augmentations=config["num_augmentations"],
            hd5_folder=config["hd5_folder"],
            num_workers=config["num_workers"],
            batch_size=BATCH_SIZE,
            train_csv=config["train_csv"],
            valid_csv=config["valid_csv"],
            test_csv=config["test_csv"],
        )
        self.cleanups = cleanups
        self.train_dataset, self.valid_dataset, _ = datasets

        self.model = build_pretraining_model(**config)
        print(
            f"Model has {count_params(self.model.trainable_weights)} trainable parameters",
        )

    def cleanup(self):
        for cleanup in self.cleanups:
            cleanup()

    def save_checkpoint(self, tmp_checkpoint_dir):
        return save_model(tmp_checkpoint_dir, self.model)

    def load_checkpoint(self, checkpoint):
        load_model(checkpoint, self.model)

    def step(self):
        history = self.model.fit(
            x=self.train_dataset,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=VALIDATION_STEPS_PER_EPOCH,
            epochs=self.iteration + 1,
            validation_data=self.valid_dataset,
            initial_epoch=self.iteration,
        )
        history_dict = {name: np.mean(val) for name, val in history.history.items()}
        if "val_loss" not in history_dict:
            raise ValueError(f"No val loss in epoch {self.iteration}")
        history_dict["epoch"] = self.iteration
        history_dict["optimizer_lr"] = get_optimizer_lr(self.model)
        history_dict["optimizer_iter"] = get_optimizer_iterations(self.model)
        print(f"Completed epoch {self.iteration}")
        return history_dict


def run(
    train_csv: str,
    valid_csv: str,
    test_csv: str,
    epochs: int,
    hd5_folder: str,
    cpus: int,
    cpus_per_model: int,
    gpus_per_model: float,
    output_folder: str,
    num_trials: int,
    patience: int,
):
    augmentation_params = {
        f"{aug_name}_strength": tune.uniform(0, 1)
        for aug_name in augmentation_dict()
        if "roll" not in aug_name
    }
    augmentation_params["roll_strength"] = 1.0
    augmentation_params["num_augmentations"] = tune.randint(0, len(augmentation_dict()))

    model_params = {
        "ecg_length": tune.qrandint(1250, 5000, 250),
        "kernel_size": tune.randint(3, 30),
        "group_size": tune.qrandint(1, 64, 8),
        "depth": tune.randint(12, 28),
        "initial_width": tune.qrandint(16, 64, 4),  # TODO: too small?
        "width_growth_rate": tune.uniform(0, 4),
        "width_quantization": tune.uniform(1.5, 3),
    }
    training_config = {
        "train_csv": train_csv,
        "valid_csv": valid_csv,
        "test_csv": test_csv,
        "hd5_folder": hd5_folder,
        "num_workers": cpus_per_model,
    }
    optimizer_params = {
        "learning_rate": tune.loguniform(1e-6, 1e-1),
        "decay_steps": STEPS_PER_EPOCH * epochs,
    }
    hyperparams = {**augmentation_params, **model_params, **optimizer_params}

    max_concurrent = min(
        cpus // cpus_per_model,
        len(tf_config.list_physical_devices("GPU")) // gpus_per_model
        if gpus_per_model
        else np.inf,
    )
    bohb_search = TuneBOHB(
        metric="val_loss",
        mode="min",
        max_concurrent=max_concurrent,
    )
    bohb_scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        metric="val_loss",
        mode="min",
        max_t=epochs,
    )

    print(
        f"Running BOHB tune for {num_trials} trials with {max_concurrent} maximum concurrent trials",
    )
    print(f"Results will appear in {output_folder}")

    stopper = EarlyStopping(patience=patience)
    analysis = tune.run(
        PretrainingTrainable,
        verbose=1,
        num_samples=num_trials,  # how many hyperparameter trials
        search_alg=bohb_search,
        scheduler=bohb_scheduler,
        resources_per_trial={
            "cpu": cpus_per_model,
            "gpu": gpus_per_model,
        },
        local_dir=output_folder,
        config={**hyperparams, **training_config},
        stop=stopper,
    )
    analysis.results_df.to_csv(os.path.join(output_folder, "results.tsv"), sep="\t")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_csv",
        help="Path to CSV with Sample IDs to reserve for training.",
        required=True,
    )
    parser.add_argument(
        "--valid_csv",
        help=("Path to CSV with Sample IDs to reserve for validation"),
        required=True,
    )
    parser.add_argument(
        "--test_csv",
        help=("Path to CSV with Sample IDs to reserve for testing."),
        required=True,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help=f"Number of training epochs of size 128 (batch size) * {STEPS_PER_EPOCH} (steps per epoch).",
        required=True,
    )
    parser.add_argument(
        "--patience",
        type=int,
        help=f"Number of epochs without progress before halting training of a model.",
        required=True,
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        help="Number of models to train.",
        required=True,
    )
    parser.add_argument(
        "--cpus",
        type=int,
        help="Number of cpus available.",
        required=True,
    )
    parser.add_argument(
        "--cpus_per_model",
        type=int,
        help="Number of cpus per model in hyperparameter optimization.",
        required=True,
    )
    parser.add_argument(
        "--gpus_per_model",
        type=float,
        help="Number of gpus per model in hyperparameter optimization.",
        required=True,
    )
    parser.add_argument(
        "--hd5_folder",
        help="Path to folder containing hd5s.",
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        default="./recipes-output",
        help="Path to output folder for recipes.py runs.",
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(**args.__dict__)
