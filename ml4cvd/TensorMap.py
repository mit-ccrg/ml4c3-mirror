# Imports: standard library
import os
import logging
import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Tuple, Union, Callable, Optional

# Imports: third party
import h5py
import numpy as np
import numcodecs
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical

# Imports: first party
from ml4cvd.metrics import (
    pearson,
    cox_hazard_loss,
    per_class_recall,
    per_class_precision,
    per_class_recall_3d,
    per_class_recall_4d,
    per_class_recall_5d,
    sentinel_logcosh_loss,
    per_class_precision_3d,
    per_class_precision_4d,
    per_class_precision_5d,
    survival_likelihood_loss,
)
from ml4cvd.normalizer import Normalizer, Standardize, ZeroMeanStd1
from ml4cvd.definitions import JOIN_CHAR, STOP_CHAR, ECG_READ_TEXT, StorageType

MEAN_IDX = 0
STD_IDX = 1
DEFAULT_TIME_TO_EVENT_CHANNELS = {"event": 0, "follow_up_days": 1}


class Interpretation(Enum):
    """Interpretations give TensorMaps semantics encoded by the numpy array the TensorMap yields.
    Interpretations tell us the kind of thing stored but nothing about its size.
    For example, a binary label and 2D pixel mask for segmentation should both have interpretation CATEGORICAL.
    CONTINUOUS Interpretations are the default and make sense for scalar values like height and weight
    as well as multidimensional arrays of raw pixel or voxel values.
    Providing explicit interpretations in TensorMap constructors is encouraged.
    Interpretations are used to set reasonable defaults values when explicit arguments are not provided."""

    CONTINUOUS = auto()
    CATEGORICAL = auto()
    EMBEDDING = auto()
    LANGUAGE = auto()
    TIME_TO_EVENT = auto()
    SURVIVAL_CURVE = auto()
    DISCRETIZED = auto()
    MESH = auto()

    def __str__(self):
        """class Interpretation.FLOAT_ARRAY becomes float_array"""
        return str.lower(super().__str__().split(".")[1])


class TimeSeriesOrder(Enum):
    NEWEST = "NEWEST"
    OLDEST = "OLDEST"
    RANDOM = "RANDOM"


class TensorMap(object):
    """Tensor maps encode the semantics, shapes and types of tensors available

    The mapping can be to numpy nd arrays, categorical labels, or continuous values.
    The tensor shapes can be inferred for categorical TensorMaps which provide a channel mapping dictionary.
    The channel map is a dict mapping a description string to an index into a numpy array i.e the tensor.
    For categorical data the resulting tensor is a one hot vector with a 1 at the channel index and zeros elsewhere.
    In general, new data sources require new TensorMaps and new tensor writers.
    Input and output names are treated differently to allow self mappings, for example auto-encoders
    """

    def __hash__(self):
        return hash((self.name, self.shape, self.interpretation))

    def __init__(
        self,
        name: str,
        interpretation: Optional[Interpretation] = Interpretation.CONTINUOUS,
        loss: Optional[Union[str, Callable]] = None,
        shape: Optional[Tuple[int, ...]] = None,
        model: Optional[Model] = None,
        metrics: Optional[List[Union[str, Callable]]] = None,
        parents: Optional[List["TensorMap"]] = None,
        sentinel: Optional[float] = None,
        validator: Optional[Union[Callable, List[Callable]]] = None,
        cacheable: Optional[bool] = False,
        activation: Optional[Union[str, Callable]] = None,
        days_window: int = 1825,
        path_prefix: Optional[str] = None,
        loss_weight: Optional[float] = 1.0,
        channel_map: Optional[Dict[str, int]] = None,
        storage_type: Optional[StorageType] = None,
        dependent_map: Optional["TensorMap"] = None,
        augmentations: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,
        normalization: Optional[Normalizer] = None,
        annotation_units: Optional[int] = 32,
        tensor_from_file: Optional[Callable] = None,
        time_series_limit: Optional[int] = None,
        time_series_order: Optional[TimeSeriesOrder] = TimeSeriesOrder.NEWEST,
        time_series_lookup: Optional[Dict[int, Tuple]] = None,
        discretization_bounds: Optional[List[float]] = None,
    ):
        """TensorMap constructor


        :param name: String name of the tensor mapping
        :param interpretation: Enum specifying semantic interpretation of the tensor: is it a label, a continuous value an embedding...
        :param loss: Loss function or str specifying pre-defined loss function
        :param shape: Tuple of integers specifying tensor shape
        :param model: The model that computes the embedding layer, only used by embedding tensor maps
        :param metrics: List of metric functions of strings
        :param parents: List of TensorMaps which must be attached to the model graph before this one
        :param sentinel: If set, this value should never naturally occur in this TensorMap, it will be used for masking loss function
        :param validator: boolean function or list of functions that validate numpy arrays (eg checks ranges or NaNs)
        :param cacheable: boolean true if tensors made by this TensorMap can be cached. Avoid this if there is randomness in tensor construction. Only compatible with legacy TensorGenerator.
        :param activation: String specifying activation function
        :param days_window: Number of days to consider for survival curve TensorMaps, the longest possible follow up.
        :param path_prefix: Path prefix of HD5 file groups where the data we are tensor mapping is located inside hd5 files
        :param loss_weight: Relative weight of the loss from this tensormap
        :param channel_map: Dictionary mapping strings indicating channel meaning to channel index integers
        :param storage_type: StorageType of tensor map
        :param dependent_map: TensorMap that depends on or is determined by this one
        :param augmentations: Tensor shape preserving transformations not applied at validation or test time
        :param normalization: Dictionary specifying normalization values
        :param annotation_units: Size of embedding dimension for unstructured input tensor maps.
        :param tensor_from_file: Function that returns numpy array from hd5 file for this TensorMap
        :param time_series_limit: If set, indicates dynamic shaping and sets the maximum number of tensors in a time series to use
        :param time_series_order: When selecting tensors in a time series, use newest, oldest, or randomly ordered tensors
        :param time_series_lookup: Dict of time intervals filtering which tensors are used in a time series
        :param discretization_bounds: List of floats that delineate the boundaries of the bins that will be used
                                          for producing categorical values from continuous values
        """
        self.name = name
        self.interpretation = interpretation

        self.loss = loss
        self.model = model
        self.shape = shape
        self.path_prefix = path_prefix
        self.metrics = metrics
        self.parents = parents
        self.sentinel = sentinel
        self.validator = validator
        self.cacheable = cacheable
        self.activation = activation
        self.days_window = days_window
        self.loss_weight = loss_weight
        self.channel_map = channel_map
        self.storage_type = storage_type
        self.augmentations = augmentations
        self.normalization = normalization
        self.dependent_map = dependent_map
        self.annotation_units = annotation_units
        self.tensor_from_file = tensor_from_file
        self.time_series_limit = time_series_limit
        self.time_series_order = time_series_order
        self.time_series_lookup = time_series_lookup
        self.discretization_bounds = discretization_bounds

        # Infer loss from interpretation
        if self.loss is None and self.is_categorical():
            self.loss = "categorical_crossentropy"
        elif self.loss is None and self.is_continuous() and self.sentinel is not None:
            self.loss = sentinel_logcosh_loss(self.sentinel)
        elif self.loss is None and self.is_continuous():
            self.loss = "mse"
        elif self.loss is None and self.is_survival_curve():
            self.loss = survival_likelihood_loss(self.static_shape[0] // 2)
        elif self.loss is None and self.is_time_to_event():
            self.loss = cox_hazard_loss
        elif self.loss is None and self.is_language():
            self.loss = "categorical_crossentropy"
        elif self.loss is None:
            self.loss = "mse"

        # Infer activation from interpretation
        if self.activation is None and (self.is_categorical() or self.is_language()):
            self.activation = "softmax"
        elif self.activation is None and self.is_continuous():
            self.activation = "linear"
        elif self.activation is None and (
            self.is_survival_curve() or self.is_time_to_event()
        ):
            self.activation = "sigmoid"

        # Infer shape from channel map or interpretation
        if self.shape is None:
            self.shape = (2,) if self.is_time_to_event() else (len(channel_map),)
            # Setting time_series_limit indicates dynamic shaping which is always accompanied by 1st dim of None
            if self.time_series_limit is not None:
                self.shape = (None,) + self.shape

        if self.channel_map is None and self.is_time_to_event():
            self.channel_map = DEFAULT_TIME_TO_EVENT_CHANNELS

        if self.discretization_bounds is not None:
            self.input_shape = self.shape
            self.input_channel_map = self.channel_map
            self.shape = self.input_shape[:-1] + (len(self.discretization_bounds) + 1,)
            self.channel_map = {
                f"channel_{k}": k for k in range(len(self.discretization_bounds) + 1)
            }

        self.infer_metrics()

        if self.tensor_from_file is None:
            self.tensor_from_file = _default_tensor_from_file

        if self.validator is None:
            self.validator = lambda tm, x, hd5: None
        if not isinstance(self.validator, list):
            self.validator = [self.validator]

    def __repr__(self):
        return f"TensorMap({self.name}, {self.shape}, {self.interpretation})"

    def __eq__(self, other):
        if not isinstance(other, TensorMap):
            return NotImplemented
        else:
            self_attributes = self.__dict__.items()
            other_attributes = other.__dict__.items()

            for (self_field, self_value), (other_field, other_value) in zip(
                self_attributes, other_attributes,
            ):
                if self_field != other_field:
                    return False
                if not _is_equal_field(self_value, other_value):
                    logging.debug(
                        f"Comparing two '{self.name}' tensor maps: '{self_field}'"
                        f" values '{self_value}' and '{other_value}' are not equal.",
                    )
                    return False
            return True

    def output_name(self):
        return JOIN_CHAR.join(["output", self.name, str(self.interpretation)])

    def input_name(self):
        return JOIN_CHAR.join(["input", self.name, str(self.interpretation)])

    def is_categorical(self):
        """For most cases categorical and discretized TensorMaps should be handled in the same way.
        The two main differences are:
            1. Discretized TensorMaps are read from disk as continuous values before they are discretized
            2. Discretization is applied to discretized TensorMaps (obvious)
        """
        return (
            self.interpretation == Interpretation.CATEGORICAL
            or self.interpretation == Interpretation.DISCRETIZED
        )

    def is_continuous(self):
        return self.interpretation == Interpretation.CONTINUOUS

    def is_embedding(self):
        return self.interpretation == Interpretation.EMBEDDING

    def is_language(self):
        return self.interpretation == Interpretation.LANGUAGE

    def is_mesh(self):
        return self.interpretation == Interpretation.MESH

    def is_time_to_event(self):
        return self.interpretation == Interpretation.TIME_TO_EVENT

    def is_survival_curve(self):
        return self.interpretation == Interpretation.SURVIVAL_CURVE

    def is_discretized(self):
        return self.interpretation == Interpretation.DISCRETIZED

    def axes(self):
        return len(self.shape)

    def static_axes(self):
        return len(self.static_shape)

    def first_dynamic_index(self):
        try:
            return self.shape.index(None)
        except ValueError:
            return -1

    @property
    def static_shape(self):
        _shape = tuple()
        for dim in self.shape:
            if dim is not None:
                _shape += (dim,)
        return _shape

    def flat_size(self):
        size = 1
        for dim in self.static_shape:
            size *= dim
        return size

    def hd5_key_guess(self):
        if self.path_prefix is None:
            return f"/{self.name}/"
        else:
            return f"/{self.path_prefix}/{self.name}/"

    def hd5_first_dataset_in_group(self, hd5, key_prefix):
        if key_prefix not in hd5:
            raise ValueError(f"Could not find key:{key_prefix} in hd5.")
        data = hd5[key_prefix]
        if isinstance(data, h5py.Dataset):
            return data
        deeper_key_prefix = f"{key_prefix}{min(hd5[key_prefix])}/"
        return self.hd5_first_dataset_in_group(hd5, deeper_key_prefix)

    def normalize(self, np_tensor):
        if self.normalization is None:
            return np_tensor
        return self.normalization.normalize(np_tensor)

    def discretize(self, np_tensor):
        if not self.is_discretized():
            return np_tensor
        return to_categorical(
            np.digitize(np_tensor, bins=self.discretization_bounds),
            num_classes=len(self.discretization_bounds) + 1,
        )

    def postprocess_tensor(self, np_tensor, augment: bool, hd5: h5py.File):
        for validator in self.validator:
            validator(self, np_tensor, hd5)
        np_tensor = self.apply_augmentations(np_tensor, augment)
        np_tensor = self.normalize(np_tensor)
        return self.discretize(np_tensor)

    def rescale(self, np_tensor):
        if self.normalization is None:
            return np_tensor
        return self.normalization.un_normalize(np_tensor)

    def apply_augmentations(self, tensor: np.ndarray, augment: bool) -> np.ndarray:
        if augment and self.augmentations is not None:
            for augmentation in self.augmentations:
                tensor = augmentation(tensor)
        return tensor

    def infer_metrics(self):
        if self.metrics is None and self.is_categorical():
            self.metrics = ["categorical_accuracy"]
            if self.static_axes() == 1:
                self.metrics += per_class_precision(self.channel_map)
                self.metrics += per_class_recall(self.channel_map)
            elif self.static_axes() == 2:
                self.metrics += per_class_precision_3d(self.channel_map)
                self.metrics += per_class_recall_3d(self.channel_map)
            elif self.static_axes() == 3:
                self.metrics += per_class_precision_4d(self.channel_map)
                self.metrics += per_class_recall_4d(self.channel_map)
            elif self.static_axes() == 4:
                self.metrics += per_class_precision_5d(self.channel_map)
                self.metrics += per_class_recall_5d(self.channel_map)
        elif self.metrics is None and self.is_continuous() and self.shape[-1] == 1:
            self.metrics = [pearson]
        elif self.metrics is None:
            self.metrics = []


def _translate(val, cur_min, cur_max, new_min, new_max):
    val -= cur_min
    val /= cur_max - cur_min
    val *= new_max - new_min
    val += new_min
    return val


def _is_equal_field(field1: Any, field2: Any) -> bool:
    """We consider two fields equal if
        a. they are not functions and they are equal, or
        b. one or both are functions and their names match
    If the fields are lists, we check for the above equality for corresponding
    elements from the list.
    """
    if isinstance(field1, list) and isinstance(field2, list):
        if len(field1) != len(field2):
            return False
        elif len(field1) == 0:
            return True

        fields1 = map(_get_name_if_function, field1)
        fields2 = map(_get_name_if_function, field2)

        return all([f1 == f2] for f1, f2 in zip(sorted(fields1), sorted(fields2)))
    else:
        return _get_name_if_function(field1) == _get_name_if_function(field2)


def _get_name_if_function(field: Any) -> Any:
    """We assume 'field' is a function if it's 'callable()'"""
    if callable(field):
        return field.__name__
    else:
        return field


def _default_continuous_tensor_from_file(tm, hd5, input_shape, input_channel_map):
    """input_shape and input_channel_map are supplied as arguments rather than accessed as attributes of tm
    so that this function can be applied to TensorMaps that are to be discretized for which tm.input_shape and
    tm.input_channel_map reflect the state of the TensorMap post-discretization
    """
    missing = True
    continuous_data = np.zeros(input_shape, dtype=np.float32)
    if tm.hd5_key_guess() in hd5:
        missing = False
        data = tm.hd5_first_dataset_in_group(hd5, tm.hd5_key_guess())
        if tm.static_axes() > 1 or tm.shape[0] > 1:
            continuous_data = np.array(data)
        elif hasattr(data, "__shape__"):
            continuous_data[0] = data[0]
        else:
            continuous_data[0] = data[()]
    if missing and input_channel_map is not None and tm.path_prefix in hd5:
        for k in input_channel_map:
            if k in hd5[tm.path_prefix]:
                missing = False
                continuous_data[input_channel_map[k]] = hd5[tm.path_prefix][k][0]
    if missing and tm.sentinel is None:
        raise ValueError(
            f"No value found for {tm.name}, channel map is {input_channel_map}"
            f" hd5_key_guess:{tm.hd5_key_guess()}",
        )
    elif missing:
        continuous_data[:] = tm.sentinel
    return continuous_data


def _default_tensor_from_file(tm, hd5, dependents={}):
    """Reconstruct a tensor from an hd5 file

    Arguments
        tm: The TensorMap that describes the type of tensor to make
        hd5: The file where the tensor was saved
        dependents: A dict that maps dependent TensorMaps to numpy arrays
            if self has a dependent TensorMap it will be constructed and added here

    Returns
        A numpy array whose dimension and type is dictated by tm
    """
    if tm.is_categorical() and not tm.is_discretized():
        index = 0
        missing = True
        categorical_data = np.zeros(tm.shape, dtype=np.float32)
        if tm.hd5_key_guess() in hd5:
            data = tm.hd5_first_dataset_in_group(hd5, tm.hd5_key_guess())
            if (
                tm.storage_type == StorageType.CATEGORICAL_INDEX
                or tm.storage_type == StorageType.CATEGORICAL_FLAG
            ):
                index = int(data[0])
                categorical_data[index] = 1.0
            else:
                categorical_data = np.array(data)
            missing = False
        elif tm.storage_type == StorageType.CATEGORICAL_FLAG:
            categorical_data[index] = 1.0
            missing = False
        elif tm.path_prefix in hd5 and tm.channel_map is not None:
            for k in tm.channel_map:
                if k in hd5[tm.path_prefix]:
                    categorical_data[tm.channel_map[k]] = 1.0
                    missing = False
        if missing:
            raise ValueError(
                f"No HD5 data found at prefix {tm.path_prefix} found for tensor map:"
                f" {tm.name}.",
            )
        return categorical_data
    elif tm.is_continuous():
        return _default_continuous_tensor_from_file(tm, hd5, tm.shape, tm.channel_map)
    elif tm.is_discretized():
        return _default_continuous_tensor_from_file(
            tm, hd5, tm.input_shape, tm.input_channel_map,
        )
    elif tm.is_embedding():
        input_dict = {}
        for input_parent_tm in tm.parents:
            input_dict[input_parent_tm.input_name()] = np.expand_dims(
                input_parent_tm.tensor_from_file(input_parent_tm, hd5), axis=0,
            )
        return tm.model.predict(input_dict)
    elif tm.is_language():
        tensor = np.zeros(tm.shape, dtype=np.float32)
        if ECG_READ_TEXT in tm.name:
            caption = decompress_data(
                data_compressed=hd5[tm.name][()], dtype=hd5[tm.name].attrs["dtype"],
            )
        else:
            caption = str(
                tm.hd5_first_dataset_in_group(hd5, tm.hd5_key_guess())[()],
            ).strip()
        char_idx = np.random.randint(tm.shape[0], len(caption) + 1)
        if char_idx == len(caption):
            next_char = STOP_CHAR
        else:
            next_char = caption[char_idx]
        if tm.dependent_map is not None:
            dependents[tm.dependent_map] = np.zeros(
                tm.dependent_map.shape, dtype=np.float32,
            )
            dependents[tm.dependent_map][tm.dependent_map.channel_map[next_char]] = 1.0
            for i, k in enumerate(range(char_idx - tm.shape[0], char_idx)):
                if caption[k] not in tm.dependent_map.channel_map:
                    logging.warning(
                        f"Could not find character {caption[k]} in channel map:"
                        f" {tm.dependent_map.channel_map}",
                    )
                    continue
                tensor[i, tm.dependent_map.channel_map[caption[k]]] = 1.0
        return tensor
    else:
        raise ValueError(
            f"No default tensor_from_file for TensorMap {tm.name} with interpretation:"
            f" {tm.interpretation}",
        )


def decompress_data(data_compressed: np.array, dtype: str) -> np.array:
    """Decompresses a compressed byte array. If the primitive type of the data
    to decompress is a string, calls decode using the zstd codec. If the
    primitive type of the data to decompress is not a string (e.g. int or
    float), the buffer is interpreted using the passed dtype."""
    codec = numcodecs.zstd.Zstd()
    data_decompressed = codec.decode(data_compressed)
    if dtype == "str":
        data = data_decompressed.decode()
    else:
        data = np.frombuffer(data_decompressed, dtype)
    return data


def outcome_channels(outcome: str):
    return {f"no_{outcome}": 0, f"{outcome}": 1}


def id_from_filename(fpath: str) -> int:
    return int(os.path.basename(fpath).split(".")[0])


def update_tmaps(tmap_name: str, tmaps: Dict[str, TensorMap]) -> Dict[str, TensorMap]:
    """
    Given name of desired TMap, and dict of all TMaps generated thus far, look if the
    desired TMap is in tmaps. If yes, return that TMap. If no, build more TMaps,
    update tmaps dict, and try to find the desired TMap again.

    :param tmap_name: name of the desired TMap that we want in the master dict
    :param tmaps: dict of all TMaps we've built so far
    """
    # If desired tmap in tmaps, return tmaps
    if tmap_name in tmaps:
        return tmaps

    # fmt: off

    # Base tmaps: ECG
    from ml4cvd.tensor_maps_ecg import tmaps as tmaps_ecg  # isort:skip
    tmaps.update(tmaps_ecg)
    if tmap_name in tmaps:
        return tmaps

    # Base tmaps: STS
    from ml4cvd.tensor_maps_sts import tmaps as tmaps_sts  # isort:skip
    tmaps.update(tmaps_sts)
    if tmap_name in tmaps:
        return tmaps

    # Base tmaps: ECG labels
    from ml4cvd.tensor_maps_ecg_labels import tmaps as tmaps_ecg_labels  # isort:skip
    tmaps.update(tmaps_ecg_labels)
    if tmap_name in tmaps:
        return tmaps

    # Base tmaps: ECG voltage
    from ml4cvd.tensor_map_updaters import update_tmaps_ecg_voltage  # isort:skip
    tmaps = update_tmaps_ecg_voltage(tmap_name=tmap_name, tmaps=tmaps)
    if tmap_name in tmaps:
        return tmaps

    # Modify: weighted loss
    from ml4cvd.tensor_map_updaters import update_tmaps_weighted_loss  # isort:skip
    tmaps = update_tmaps_weighted_loss(tmap_name=tmap_name, tmaps=tmaps)
    if tmap_name in tmaps:
        return tmaps

    # Modify: STS window (e.g. preop)
    from ml4cvd.tensor_map_updaters import update_tmaps_sts_window  # isort:skip
    tmaps = update_tmaps_sts_window(tmap_name=tmap_name, tmaps=tmaps)
    if tmap_name in tmaps:
        return tmaps

    # Modify: time series
    from ml4cvd.tensor_map_updaters import update_tmaps_time_series  # isort:skip
    tmaps = update_tmaps_time_series(tmap_name=tmap_name, tmaps=tmaps)
    if tmap_name in tmaps:
        return tmaps

    # fmt: on

    raise ValueError(
        f"{tmap_name} cannot be found in tmaps despite building every TMap we can",
    )
