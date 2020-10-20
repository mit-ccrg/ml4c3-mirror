# Imports: standard library
import os
import logging
from enum import Enum, auto
from typing import Any, Dict, List, Tuple, Union, Callable, Optional

# Imports: third party
import h5py
import numpy as np
from tensorflow.keras import Model

# Imports: first party
from ml4c3.metrics import (
    pearson,
    cox_hazard_loss,
    per_class_recall,
    per_class_precision,
    per_class_recall_3d,
    per_class_recall_4d,
    per_class_recall_5d,
    per_class_precision_3d,
    per_class_precision_4d,
    per_class_precision_5d,
    survival_likelihood_loss,
)
from ml4c3.normalizer import Normalizer

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
    LANGUAGE = auto()
    TIME_TO_EVENT = auto()
    SURVIVAL_CURVE = auto()

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

    def __init__(
        self,
        name: str,
        loss: Optional[Union[str, Callable]] = None,
        shape: Optional[Tuple[int, ...]] = None,
        model: Optional[Model] = None,
        metrics: Optional[List[Union[str, Callable]]] = None,
        activation: Optional[Union[str, Callable]] = None,
        days_window: int = 1825,
        path_prefix: Optional[str] = None,
        loss_weight: Optional[float] = 1.0,
        channel_map: Optional[Dict[str, int]] = None,
        augmenters: Optional[Union[Callable, List[Callable]]] = None,
        validators: Optional[Union[Callable, List[Callable]]] = None,
        normalizers: Optional[Union[Normalizer, List[Normalizer]]] = None,
        interpretation: Optional[Interpretation] = Interpretation.CONTINUOUS,
        annotation_units: Optional[int] = None,
        tensor_from_file: Optional[Callable] = None,
        time_series_limit: Optional[int] = None,
        time_series_order: Optional[TimeSeriesOrder] = TimeSeriesOrder.NEWEST,
        time_series_lookup: Optional[Dict[int, List[Tuple[str, str]]]] = None,
        linked_tensors: bool = False,
    ):
        """
        TensorMap constructor

        :param name: String name of the tensor mapping
        :param interpretation: Enum specifying semantic interpretation of the tensor: is it a label, a continuous value an embedding...
        :param loss: Loss function or str specifying pre-defined loss function
        :param shape: Tuple of integers specifying tensor shape
        :param model: The model that computes the embedding layer, only used by embedding tensor maps
        :param metrics: List of metric functions of strings
        :param activation: String specifying activation function
        :param path_prefix: Path prefix of HD5 file groups where the data we are tensor mapping is located inside hd5 files
        :param loss_weight: Relative weight of the loss from this tensormap
        :param channel_map: Dictionary mapping strings indicating channel meaning to channel index integers
        :param augmenters: Callable or List of Callables to transform the tensor while preserving shape
        :param validators: Callable or List of Callables that raise errors if tensor is invalid
        :param normalizers: Normalizer of List of Normalizers that normalize the tensor
        :param annotation_units: Size of embedding dimension for unstructured input tensor maps.
        :param tensor_from_file: Function that returns numpy array from hd5 file for this TensorMap
        :param time_series_limit: If set, indicates dynamic shaping and sets the maximum number of tensors in a time series to use
        :param time_series_order: When selecting tensors in a time series, use newest, oldest, or randomly ordered tensors
        :param time_series_lookup: Dict of list of time intervals filtering which tensors are used in a time series
        :param linked_tensors: Bool indicating if tensors in time series returned by this tensor map should be linked
        """
        self.name = name
        self.loss = loss
        self.model = model
        self.shape = shape
        self.metrics = metrics
        self.activation = activation
        self.days_window = days_window
        self.loss_weight = loss_weight
        self.path_prefix = path_prefix
        self.channel_map = channel_map
        self.augmenters = augmenters
        self.validators = validators
        self.normalizers = normalizers
        self.interpretation = interpretation
        self.annotation_units = annotation_units
        self.tensor_from_file = tensor_from_file
        self.time_series_limit = time_series_limit
        self.time_series_order = time_series_order
        self.time_series_lookup = time_series_lookup
        self.linked_tensors = linked_tensors

        if self.tensor_from_file is None:
            raise ValueError(f"{self.name}: tensor_from_file cannot be None")

        # Infer channel map from interpretation
        if self.channel_map is None and self.is_time_to_event:
            self.channel_map = DEFAULT_TIME_TO_EVENT_CHANNELS

        # Infer shape from channel map or interpretation
        if self.shape is None:
            if self.channel_map is not None:
                if self.is_time_to_event:
                    self.shape = (2,)
                else:
                    self.shape = (len(self.channel_map),)
            else:
                raise ValueError(f"{self.name}: cannot infer shape")

        # Infer loss from interpretation and shape
        if self.loss is None:
            if self.is_categorical or self.is_language:
                self.loss = "categorical_crossentropy"
            elif self.is_continuous:
                self.loss = "mse"
            elif self.is_survival_curve:
                self.loss = survival_likelihood_loss(self.shape[0] // 2)
            elif self.is_time_to_event:
                self.loss = cox_hazard_loss
            else:
                raise ValueError(f"{self.name}: cannot infer loss")

        # Infer activation from interpretation
        if self.activation is None:
            if self.is_categorical or self.is_language:
                self.activation = "softmax"
            elif self.is_continuous:
                self.activation = "linear"
            elif self.is_survival_curve or self.is_time_to_event:
                self.activation = "sigmoid"
            else:
                raise ValueError(f"{self.name}: cannot infer activation")

        # Infer metrics from interpretation
        if self.metrics is None:
            if self.is_categorical:
                self.metrics = ["categorical_accuracy"]
                if self.axes == 1:
                    self.metrics += per_class_precision(self.channel_map)
                    self.metrics += per_class_recall(self.channel_map)
                elif self.axes == 2:
                    self.metrics += per_class_precision_3d(self.channel_map)
                    self.metrics += per_class_recall_3d(self.channel_map)
                elif self.axes == 3:
                    self.metrics += per_class_precision_4d(self.channel_map)
                    self.metrics += per_class_recall_4d(self.channel_map)
                elif self.axes == 4:
                    self.metrics += per_class_precision_5d(self.channel_map)
                    self.metrics += per_class_recall_5d(self.channel_map)
            elif self.is_continuous and self.shape[-1] == 1:
                self.metrics = [pearson]
            else:
                self.metrics = []

        # Infer embedded dimensionality from shape
        if self.annotation_units is None:
            self.annotation_units = self.size

        # Wrap augmenter, validator, and normalizer in lists
        if self.augmenters is not None and not isinstance(self.augmenters, list):
            self.augmenters = [self.augmenters]
        if self.validators is not None and not isinstance(self.validators, list):
            self.validators = [self.validators]
        if self.normalizers is not None and not isinstance(self.normalizers, list):
            self.normalizers = [self.normalizers]

    def __hash__(self):
        return hash((self.name, self.shape, self.interpretation))

    def __repr__(self):
        return f"TensorMap({self.name}, {self.shape}, {self.interpretation})"

    def __eq__(self, other):
        if not isinstance(other, TensorMap):
            return NotImplemented
        else:
            self_attributes = self.__dict__.items()
            other_attributes = other.__dict__.items()

            for (self_field, self_value), (other_field, other_value) in zip(
                self_attributes,
                other_attributes,
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

    @property
    def output_name(self):
        return f"output_{self.name}_{self.interpretation}"

    @property
    def input_name(self):
        return f"input_{self.name}_{self.interpretation}"

    @property
    def is_categorical(self):
        return self.interpretation == Interpretation.CATEGORICAL

    @property
    def is_continuous(self):
        return self.interpretation == Interpretation.CONTINUOUS

    @property
    def is_language(self):
        return self.interpretation == Interpretation.LANGUAGE

    @property
    def is_time_to_event(self):
        return self.interpretation == Interpretation.TIME_TO_EVENT

    @property
    def is_survival_curve(self):
        return self.interpretation == Interpretation.SURVIVAL_CURVE

    @property
    def axes(self):
        return len(self.shape)

    @property
    def size(self) -> int:
        size = 1
        for dim in self.shape:
            size *= dim
        return size

    def postprocess_tensor(
        self,
        tensor: np.ndarray,
        hd5: h5py.File,
        augment: bool,
    ) -> np.ndarray:
        self.validate(tensor, hd5)
        tensor = self.augment(tensor) if augment else tensor
        tensor = self.normalize(tensor)
        return tensor

    def validate(self, tensor: np.ndarray, hd5: h5py.File):
        if self.validators is not None:
            for validator in self.validators:
                validator(self, tensor, hd5)

    def augment(self, tensor: np.ndarray) -> np.ndarray:
        if self.augmenters is not None:
            for augmenter in self.augmenters:
                tensor = augmenter(tensor)
        return tensor

    def normalize(self, tensor: np.ndarray) -> np.ndarray:
        if self.normalizers is not None:
            for normalizer in self.normalizers:
                tensor = normalizer.normalize(tensor)
        return tensor

    def rescale(self, tensor: np.ndarray) -> np.ndarray:
        if self.normalizers is not None:
            for normalizer in self.normalizers:
                tensor = normalizer.un_normalize(tensor)
        return tensor


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
        field = field.__name__

    if isinstance(field, str):
        return field.replace("-", "").replace("_", "")
    else:
        return field


def outcome_channels(outcome: str):
    return {f"no_{outcome}": 0, f"{outcome}": 1}


def find_negative_label_and_channel(
    labels: Dict[str, int],
    negative_label_prefix: str = "no_",
) -> Tuple[str, int]:
    """
    Given a dict of labels (e.g. tm.channel_map) and their channel indices,
    return the negative label and its channel; defaults to smallest channel index
    """
    labels = sorted(labels.items(), key=lambda cm: cm[1])
    negative_label, negative_label_index = labels[0]
    for label, index in labels:
        if label.startswith(negative_label_prefix):
            negative_label = label
            negative_label_index = index
    return negative_label, negative_label_index


def id_from_filename(fpath: str) -> int:
    return int(os.path.basename(fpath).split(".")[0])


def binary_channel_map(tm: TensorMap) -> bool:
    """Return true if"
    a) tensor map has a channel map,
    b) the channel map has two elements
    c) the string "no_" is in the channel map"""
    return (
        (tm.channel_map is not None)
        and (len(tm.channel_map) == 2)
        and (np.any(["no_" in cm for cm in tm.channel_map]))
    )


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

    # ICU tmaps
    try:
        from ml4icu.tensormap import TMAPS as tmaps_icu # isort: skip
        tm = tmaps_icu[tmap_name]
        tmaps[tmap_name] = tm
        return tmaps
    except ModuleNotFoundError:
        logging.debug("ml4icu is not installed")
    except KeyError:
        logging.debug(f"{tmap_name} not found in icu tmaps")

    # Base tmaps: ECG
    from ml4c3.tensormap.tensor_maps_ecg import tmaps as tmaps_ecg  # isort:skip
    tmaps.update(tmaps_ecg)
    if tmap_name in tmaps:
        return tmaps

    # Base tmaps: STS
    from ml4c3.tensormap.tensor_maps_sts import tmaps as tmaps_sts  # isort:skip
    tmaps.update(tmaps_sts)
    if tmap_name in tmaps:
        return tmaps

    # Base tmaps: ECG labels
    from ml4c3.tensormap.tensor_maps_ecg_labels import tmaps as tmaps_ecg_labels  # isort:skip
    tmaps.update(tmaps_ecg_labels)
    if tmap_name in tmaps:
        return tmaps

    # Base tmaps: ECG voltage
    from ml4c3.tensormap.tensor_map_updaters import update_tmaps_ecg_voltage  # isort:skip
    tmaps = update_tmaps_ecg_voltage(tmap_name=tmap_name, tmaps=tmaps)
    if tmap_name in tmaps:
        return tmaps

    # Modify: weighted loss
    from ml4c3.tensormap.tensor_map_updaters import update_tmaps_weighted_loss  # isort:skip
    tmaps = update_tmaps_weighted_loss(tmap_name=tmap_name, tmaps=tmaps)
    if tmap_name in tmaps:
        return tmaps

    # Modify: STS window (e.g. preop)
    from ml4c3.tensormap.tensor_map_updaters import update_tmaps_sts_window  # isort:skip
    tmaps = update_tmaps_sts_window(tmap_name=tmap_name, tmaps=tmaps)
    if tmap_name in tmaps:
        return tmaps

    # Modify: time series
    from ml4c3.tensormap.tensor_map_updaters import update_tmaps_time_series  # isort:skip
    tmaps = update_tmaps_time_series(tmap_name=tmap_name, tmaps=tmaps)
    if tmap_name in tmaps:
        return tmaps

    # Modify: load predictions
    from ml4c3.tensormap.tensor_map_updaters import update_tmaps_model_predictions  # isort:skip
    tmaps = update_tmaps_model_predictions(tmap_name=tmap_name, tmaps=tmaps)
    if tmap_name in tmaps:
        return tmaps

    # fmt: on

    raise ValueError(
        f"{tmap_name} cannot be found in tmaps despite building every TMap we can",
    )
