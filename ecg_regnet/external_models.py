from typing import Dict, List

import numpy as np
import tensorflow_hub as hub
from ray.tune import Trainable
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Reshape
from tensorflow_addons.optimizers import SGDW, RectifiedAdam
from tensorflow.python.keras.utils.layer_utils import count_params

from data import get_downstream_datasets, get_ecg_tmap, downstream_tmap_from_name
from hyperoptimize import BATCH_SIZE


class ML4C3EfficientNet(Model):
    """Compatible with ML4C3 TensorGenerator"""

    def __init__(
        self,
        input_name: str,
        output_name_to_shape: Dict[str, int],
        freeze: bool = False,
    ):
        super(ML4C3EfficientNet, self).__init__()
        self.body = hub.KerasLayer(
            "https://tfhub.dev/google/efficientnet/b0/feature-vector/1",
            input_shape=(100, 100)
        )
        if freeze:
            self.body.trainable = False
        self.input_name = input_name
        self.denses = {
            name: Dense(shape, name=name)
            for name, shape in output_name_to_shape.items()
        }
        self.reshape = Reshape((100, 100, 3))

    def call(self, inputs, training=None, mask=None):
        x = inputs[self.input_name]
        x = self.reshape(x)
        x = self.body(x)
        return {name: dense(x) for name, dense in self.denses.items()}


def _test_efficient_net():
    import numpy as np
    m = ML4C3EfficientNet("ecg", {"age": 1})
    x = m.predict({"ecg": np.zeros((5, 2500, 12), dtype=np.float32)})
    assert x["age"].shape == (5, 1)


class ML4C3Ribeiro(Model):
    """
    ml4c3 compatible model with weights from https://www.nature.com/articles/s41467-020-15432-4
    weights can be downloaded at https://zenodo.org/record/3765717#.YBiPzXdKhhE
    """
    def __init__(
            self,
            input_name: str,
            output_name_to_shape: Dict[str, int],
            ribeiro_weights_path: str,
            freeze: bool = False,
    ):
        super(ML4C3Ribeiro, self).__init__()
        self.input_name = input_name
        self.denses = {
            name: Dense(shape, name=name)
            for name, shape in output_name_to_shape.items()
        }
        self.body = load_model(ribeiro_weights_path)
        self.body.layers.pop()  # remove dense layer at end
        if freeze:
            self.body.trainable = False

    def call(self, inputs, training=None, mask=None):
        """note that ribeiro model requires input shape 4097, 12"""
        x = inputs[self.input_name]
        x = self.body(x)
        return {name: dense(x) for name, dense in self.denses.items()}


def transform_datasets_ribeiro(datasets: List[tf.data.Dataset]) -> List[tf.data.Dataset]:
    """Ribeiro et al. has units of 10^-4 volts vs. our dataset's milli volts"""
    name = get_ecg_tmap(0, []).input_name
    return [
        dataset.apply(lambda x: x[0][name] * 10) for dataset in datasets
    ]


def _test_ribeiro():
    import numpy as np
    m = ML4C3Ribeiro("ecg", {"age": 1}, "/Users/ndiamant/Downloads/model/model.hdf5")
    x = m.predict({"ecg": np.zeros((5, 4096, 12), dtype=np.float32)})
    assert x["age"].shape == (5, 1)


class EfficientNetTrainable(Trainable):
    def setup(self, config):
        import tensorflow as tf  # necessary for ray tune

        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(
                gpu,
                True,
            )  # do not allocate all memory right away

        # build model
        downstream_tmap_name = config.get("downstream_tmap_name", False)
        tmap = downstream_tmap_from_name(downstream_tmap_name)
        input_name = get_ecg_tmap(0, []).input_name
        output_name_to_shape = {tmap.output_name: tmap.shape[0]}
        self.model = ML4C3EfficientNet(
            input_name=input_name,
            output_name_to_shape=output_name_to_shape,
        )
        optimizer = RectifiedAdam(config["learning_rate"])
        self.model.compile(loss=tmap.loss, optimizer=optimizer)
        self.model.summary()
        print(
            f"Model has {count_params(self.model.trainable_weights)} trainable parameters",
        )

        # build data
        size = config.get("downstream_size", False)
        assert size
        datasets, stats, cleanups = get_downstream_datasets(
            downstream_tmap_name=downstream_tmap_name,
            downstream_size=size,
            ecg_length=2500,
            augmentation_strengths={},
            num_augmentations=0,
            hd5_folder=config["hd5_folder"],
            num_workers=config["num_workers"],
            batch_size=BATCH_SIZE,
            csv_folder=config["csv_folder"],
        )

        self.cleanups = cleanups
        self.train_dataset, self.valid_dataset, _ = datasets

    def save_checkpoint(self, tmp_checkpoint_dir):
        return self.model.save(tmp_checkpoint_dir)

    def load_checkpoint(self, checkpoint):
        self.model = load_model(checkpoint)

    def cleanup(self):
        for cleanup in self.cleanups:
            cleanup()

    def step(self):
        history = self.model.fit(
            x=self.train_dataset,
            epochs=self.iteration + 1,
            validation_data=self.valid_dataset,
            initial_epoch=self.iteration,
        )
        history_dict = {name: np.mean(val) for name, val in history.history.items()}
        if "val_loss" not in history_dict:
            raise ValueError(f"No val loss in epoch {self.iteration}")
        history_dict["epoch"] = self.iteration
        print(f"Completed epoch {self.iteration}")
        return history_dict


class RibeiroTrainable(Trainable):
    def setup(self, config):
        import tensorflow as tf  # necessary for ray tune

        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(
                gpu,
                True,
            )  # do not allocate all memory right away

        # build model
        downstream_tmap_name = config.get("downstream_tmap_name", False)
        tmap = downstream_tmap_from_name(downstream_tmap_name)
        input_name = get_ecg_tmap(0, []).input_name
        output_name_to_shape = {tmap.output_name: tmap.shape[0]}
        self.model = ML4C3Ribeiro(
            input_name=input_name,
            output_name_to_shape=output_name_to_shape,
            ribeiro_weights_path=config["model_file"]
        )
        optimizer = RectifiedAdam(config["learning_rate"])
        self.model.compile(loss=tmap.loss, optimizer=optimizer)
        self.model.summary()
        print(
            f"Model has {count_params(self.model.trainable_weights)} trainable parameters",
        )

        # build data
        size = config.get("downstream_size", False)
        assert size
        datasets, stats, cleanups = get_downstream_datasets(
            downstream_tmap_name=downstream_tmap_name,
            downstream_size=size,
            ecg_length=4096,
            augmentation_strengths={},
            num_augmentations=0,
            hd5_folder=config["hd5_folder"],
            num_workers=config["num_workers"],
            batch_size=BATCH_SIZE,
            csv_folder=config["csv_folder"],
        )
        self.cleanups = cleanups
        self.train_dataset, self.valid_dataset, _ = transform_datasets_ribeiro(datasets)

    def save_checkpoint(self, tmp_checkpoint_dir):
        return self.model.save(tmp_checkpoint_dir)

    def load_checkpoint(self, checkpoint):
        self.model = load_model(checkpoint)

    def cleanup(self):
        for cleanup in self.cleanups:
            cleanup()

    def step(self):
        history = self.model.fit(
            x=self.train_dataset,
            epochs=self.iteration + 1,
            validation_data=self.valid_dataset,
            initial_epoch=self.iteration,
        )
        history_dict = {name: np.mean(val) for name, val in history.history.items()}
        if "val_loss" not in history_dict:
            raise ValueError(f"No val loss in epoch {self.iteration}")
        history_dict["epoch"] = self.iteration
        print(f"Completed epoch {self.iteration}")
        return history_dict
