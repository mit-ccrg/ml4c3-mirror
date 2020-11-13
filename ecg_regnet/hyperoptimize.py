import os
import argparse
import numpy as np
from ray import tune
from .regnet import ML4C3Regnet
from .data import get_pretraining_datasets, get_ecg_tmap, get_pretraining_tasks, augmentation_dict
from ml4c3.metrics import get_metric_dict
from tensorflow_addons.optimizers import SGDW
from tensorflow.keras.experimental import CosineDecay


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
        cpus_available: int, gpus_available: int,
        parallel_trials: int,
        output_folder: str,
):
    # TODO: consider population based optimization
    # TODO: consider bayesian optimization
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
        'width_growth_rate': tune.uniform(0, 256),
        'width_quantization': tune.uniform(1.5, 3),
    }
    training_config = {
        'train_csv': train_csv,
        'valid_csv': valid_csv,
        'test_csv': test_csv,
        'epochs': epochs,
        'hd5_folder': hd5_folder,
    }
    tune.run(
        PretrainingTrainable,
        stop={'training_iteration': epochs},
        verbose=1,
        num_samples=4,  # how many models to train
        resources_per_trial={
            'cpu': cpus_available / parallel_trials,
            'gpu': gpus_available / gpus_available,
        },
        metric='val_loss',
        mode='min',
        local_dir=output_folder,
        config={
            **augmentation_strengths, **model_params,
            **training_config,
        }
    )


if __name__ == '__main__':
    run(  # TODO: argparse for these parameters
        train_csv, valid_csv, test_csv,
        epochs, hd5_folder,
        cpus_available, gpus_available,
        parallel_trials,
        output_folder,
    )
