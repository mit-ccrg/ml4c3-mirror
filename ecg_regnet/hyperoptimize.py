# Imports: standard library
import os
import argparse

# Imports: third party
import numpy as np
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


def build_pretraining_model(
    epochs: int,
    ecg_length: int,
    kernel_size: int,
    group_size: int,
    depth: int,
    initial_width: int,
    width_growth_rate: int,
    width_quantization: int,
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
        initial_learning_rate=0.005,
        decay_steps=epochs,
    )
    optimizer = SGDW(
        learning_rate=lr_schedule,
        momentum=0.9,
        weight_decay=5 * 5e-5,
    )  # following regnet's setup
    model({input_name: Input(shape=(ecg_length, 12))})  # initialize model
    model.compile(loss=[tm.loss for tm in tmaps_out], optimizer=optimizer)
    return model


def _test_build_pretraining_model():
    m = build_pretraining_model(
        epochs=10,
        ecg_length=100,
        kernel_size=3,
        group_size=2,
        depth=10,
        initial_width=32,
        width_growth_rate=3,
        width_quantization=1.5,
    )
    m.summary()
    dummy_in = {m.input_name: np.zeros((10, 100, 12))}
    dummy_out = {
        tm.output_name: np.zeros((10,) + tm.shape) for tm in get_pretraining_tasks()
    }
    m.fit(dummy_in, dummy_out, batch_size=2)


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
            batch_size=128,  # following RegNet
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
        self.model.save_weights(
            os.path.join(tmp_checkpoint_dir, "pretraining_model.h5"),
        )

    def load_checkpoint(self, checkpoint):
        self.model.load_weights(checkpoint)

    def step(self):
        history = self.model.fit(
            x=self.train_dataset,
            epochs=1,
            validation_data=self.valid_dataset,
        )
        history_dict = {name: np.mean(val) for name, val in history.history}
        history_dict["epoch"] = self.iteration
        return history_dict


def run(
    train_csv: str,
    valid_csv: str,
    test_csv: str,
    epochs: int,
    hd5_folder: str,
    cpus_per_model: int,
    gpus_per_model: float,
    output_folder: str,
    num_trials: int,
):
    cpus = 16  # TODO: should be an argument
    augmentation_params = {
        f"{aug_name}_strength": tune.uniform(0, 1) for aug_name in augmentation_dict()
    }
    augmentation_params["num_augmentations"] = tune.randint(0, len(augmentation_dict()))

    model_params = {
        "ecg_length": tune.qrandint(1250, 5000, 250),
        "kernel_size": tune.randint(3, 10),
        "group_size": tune.qrandint(1, 32, 8),
        "depth": tune.randint(12, 28),
        "initial_width": tune.qrandint(16, 64, 4),  # TODO: too small?
        "width_growth_rate": tune.uniform(0, 4),
        "width_quantization": tune.uniform(1.5, 3),
    }
    training_config = {
        "train_csv": train_csv,
        "valid_csv": valid_csv,
        "test_csv": test_csv,
        "epochs": epochs,
        "hd5_folder": hd5_folder,
        "num_workers": cpus_per_model,
    }
    hyperparams = {**augmentation_params, **model_params}

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
        reuse_actors=True,  # speed up trials by reusing resources
    )


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
        help="Number of training epochs.",
        required=True,
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        help="Number of training epochs.",
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
