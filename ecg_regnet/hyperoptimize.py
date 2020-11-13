import os
import argparse
import numpy as np
from ray import tune
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
from .regnet import ML4C3Regnet
from .data import get_pretraining_datasets, get_ecg_tmap, get_pretraining_tasks, augmentation_dict
from ml4c3.metrics import get_metric_dict
from tensorflow_addons.optimizers import SGDW
from tensorflow.keras.experimental import CosineDecay
from tensorflow import config as tf_config
from multiprocessing import cpu_count


def build_pretraining_model(
        epochs: int,
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
    output_name_to_shape = {
        tmap.output_name: tmap.shape[0]
        for tmap in tmaps_out
    }
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
    metrics = get_metric_dict(tmaps_out)
    lr_schedule = CosineDecay(
        initial_learning_rate=.005, decay_steps=epochs,
    )
    optimizer = SGDW(
        learning_rate=lr_schedule, momentum=.9, weight_decay=5 * 5e-5,
    )  # following regnet's setup
    model.compile(loss=metrics, optimizer=optimizer)
    return model


class PretrainingTrainable(tune.Trainable):

    def setup(self, config):
        import tensorflow as tf  # necessary for ray tune

        augmentation_strengths = {
            name: strength
            for name, strength in config.items() if 'strength' in name
        }

        datasets, stats, cleanups = get_pretraining_datasets(
            ecg_length=config['ecg_length'],
            augmentation_strengths=augmentation_strengths,
            hd5_folder=config['hd5_folder'],
            num_workers=config['workers'],
            batch_size=128,  # following RegNet
            train_csv=config['train_csv'],
            valid_csv=config['valid_csv'],
            test_csv=config['test_csv']
        )
        self.cleanups = cleanups
        self.train_dataset, self.valid_dataset, _ = datasets

        self.model = build_pretraining_model(**config)

    def cleanup(self):
        for cleanup in self.cleanups:
            cleanup()

    def save_checkpoint(self, tmp_checkpoint_dir):
        self.model.save_weights(os.path.join(tmp_checkpoint_dir, 'pretraining_model.h5'))

    def load_checkpoint(self, checkpoint):
        self.model.load_weights(checkpoint)

    def step(self):
        history = self.model.fit(
            batch_size=128, x=self.train_dataset,
            epochs=1, validation_data=self.valid_dataset,
        )
        history_dict = {
            name: np.mean(val) for name, val in history.history
        }
        history_dict['epoch'] = self.iteration
        return history_dict


def run(
        train_csv: str, valid_csv: str, test_csv: str,
        epochs: int, hd5_folder: str,
        cpus_per_model: int, gpus_per_model: float,
        output_folder: str,
        num_trials: int,
):
    augmentation_strengths = {
        f'{aug_name}_strength': tune.uniform(0, 1)
        for aug_name in augmentation_dict()
    }
    model_params = {
        'ecg_length': tune.randint(1250, 5001),
        'kernel_size': tune.randint(3, 10),
        'group_size': tune.qrandint(1, 32, 8),
        'depth': tune.randint(12, 28),
        'initial_width': tune.randint(16, 256),
        'width_growth_rate': tune.uniform(0, 4),
        'width_quantization': tune.uniform(1.5, 3),
    }
    training_config = {
        'train_csv': train_csv,
        'valid_csv': valid_csv,
        'test_csv': test_csv,
        'epochs': epochs,
        'hd5_folder': hd5_folder,
    }
    hyperparams = {**augmentation_strengths, **model_params}

    max_concurrent = min(
            cpu_count() // cpus_per_model,
            len(tf_config.list_physical_devices('GPU') // gpus_per_model)
        )
    bohb_search = TuneBOHB(
        hyperparams,
        metric='val_loss',
        max_concurrent=max_concurrent
    )
    bohb_scheduler = HyperBandForBOHB(
        time_attr='training_iteration',
        metric='val_loss',
        mode='min',
        max_t=epochs,
    )

    print(f'Running BOHB tune for {num_trials} trials with {max_concurrent} maximum concurrent trials')
    print(f'Results will appear in {output_folder}')
    analysis = tune.run(
        PretrainingTrainable,
        verbose=1,
        num_samples=num_trials,  # how many hyperparameter trials
        search_alg=bohb_search,
        scheduler=bohb_scheduler,
        resources_per_trial={
            'cpu': cpus_per_model,
            'gpu': gpus_per_model,
        },
        local_dir=output_folder,
        config={**hyperparams, **training_config}
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_csv",
        help="Path to CSV with Sample IDs to reserve for training.",
    )
    parser.add_argument(
        "--valid_csv",
        help=(
            "Path to CSV with Sample IDs to reserve for validation. Takes precedence"
            " over valid_ratio."
        ),
    )
    parser.add_argument(
        "--test_csv",
        help=(
            "Path to CSV with Sample IDs to reserve for testing. Takes precedence over"
            " test_ratio."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--cpus_per_model",
        type=int,
        help="Number of cpus per model in hyperparameter optimization.",
    )
    parser.add_argument(
        "--gpus_per_model",
        type=float,
        help="Number of gpus per model in hyperparameter optimization.",
    )
    parser.add_argument(
        "--hd5_folder",
        help="Path to folder containing tensors, or where tensors will be written.",
    )
    parser.add_argument(
        "--output_folder",
        default="./recipes-output",
        help="Path to output folder for recipes.py runs.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(**args.__dict__)
