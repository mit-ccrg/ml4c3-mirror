# Imports: standard library
import os
import logging
from enum import Enum, auto
from typing import Any, Dict, List, Tuple, Union, Callable, Optional
from datetime import datetime

# Imports: third party
import h5py
import numpy as np
import pandas as pd
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
from ml4c3.definitions.globals import TIMEZONE

Axis = Union[int, None]

DEFAULT_TIME_TO_EVENT_CHANNELS = {"event": 0, "follow_up_days": 1}

# pylint: disable=line-too-long, import-outside-toplevel, too-many-return-statements


class Interpretation(Enum):
    """
    Interpretations give TensorMaps semantics encoded by the numpy array the
    TensorMap yields.

    Interpretations tell us the kind of thing stored but nothing about its size.
    For example, a binary label and 2D pixel mask for segmentation should both
    have interpretation CATEGORICAL.
    CONTINUOUS Interpretations are the default and make sense for scalar values
    like height and weight
    as well as multidimensional arrays of raw pixel or voxel values.
    Providing explicit interpretations in TensorMap constructors is encouraged.
    Interpretations are used to set reasonable defaults values when explicit
    arguments are not provided.
    """

    CONTINUOUS = auto()
    CATEGORICAL = auto()
    LANGUAGE = auto()
    EVENT = auto()
    TIMESERIES = auto()
    TIME_TO_EVENT = auto()
    SURVIVAL_CURVE = auto()

    def __str__(self):
        """
        class Interpretation.FLOAT_ARRAY becomes float_array
        """
        return str.lower(super().__str__().split(".")[1])


class TimeSeriesOrder(Enum):
    """
    TimeSeriesOrder defines which data in a time series are selected.
    """

    NEWEST = "NEWEST"
    OLDEST = "OLDEST"
    RANDOM = "RANDOM"


class TensorMap:
    """
    Tensor maps encode the semantics, shapes and types of tensors available

    The mapping can be to numpy nd arrays, categorical labels, or continuous values.
    The tensor shapes can be inferred for categorical TensorMaps which provide a
    channel mapping dictionary. The channel map is a dict mapping a description
    string to an index into a numpy array i.e the tensor. For categorical data the
    resulting tensor is a one hot vector with a 1 at the channel index and zeros
    elsewhere. In general, new data sources require new TensorMaps and new tensor
    writers. Input and output names are treated differently to allow self mappings,
    for example auto-encoders
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
        time_series_filter: Optional[Callable] = None,
        linked_tensors: bool = False,
    ):
        """
        TensorMap constructor

        :param name: String name of the tensor mapping
        :param interpretation: Enum specifying semantic interpretation of the tensor:
                               is it a label, a continuous value an embedding...
        :param loss: Loss function or str specifying pre-defined loss function
        :param shape: Tuple of integers specifying tensor shape
        :param model: The model that computes the embedding layer, only used by
                      embedding tensor maps
        :param metrics: List of metric functions of strings
        :param activation: String specifying activation function
        :param path_prefix: Path prefix of HD5 file groups where the data we are
                            tensor mapping is located inside hd5 files
        :param loss_weight: Relative weight of the loss from this tensormap
        :param channel_map: Dictionary mapping strings indicating channel meaning
                            to channel index integers
        :param augmenters: Callable or List of Callables to transform the tensor
                           while preserving shape
        :param validators: Callable or List of Callables that raise errors if
                           tensor is invalid
        :param normalizers: Normalizer of List of Normalizers that normalize the tensor
        :param annotation_units: Size of embedding dimension for unstructured input
                                 tensor maps.
        :param tensor_from_file: Function that returns numpy array from hd5 file
                                 for this TensorMap
        :param time_series_limit: If set, indicates dynamic shaping and sets the
                                  maximum number of tensors in a time series to use
        :param time_series_filter: Function that returns the distinct time events to use
        :param linked_tensors: Bool indicating if tensors in time series returned by
                               this tensor map should be linked.
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
        self.time_series_filter = time_series_filter
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
            elif self.is_continuous or self.is_timeseries or self.is_event:
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
            elif self.is_continuous or self.is_timeseries or self.is_event:
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

        if self.time_series_filter is None:
            self.time_series_filter = lambda hd5: sorted(list(hd5[self.path_prefix]))

    def __hash__(self):
        return hash((self.name, self.shape, self.interpretation))

    def __repr__(self):
        return f"TensorMap({self.name}, {self.shape}, {self.interpretation})"

    def __eq__(self, other):
        if not isinstance(other, TensorMap):
            return NotImplemented
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
    def is_event(self):
        return self.interpretation == Interpretation.EVENT

    @property
    def is_timeseries(self):
        return self.interpretation == Interpretation.TIMESERIES

    @property
    def axes(self):
        return len(self.shape)

    @property
    def size(self) -> int:
        size = 1
        if self.shape is None:
            raise ValueError(
                "No shape found. It is not possible to obtain the corresponding size",
            )
        for dim in self.shape:
            if dim is not None:
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
            for validator in self.validators:  # type: ignore
                validator(self, tensor, hd5)

    def augment(self, tensor: np.ndarray) -> np.ndarray:
        if self.augmenters is not None:
            for augmenter in self.augmenters:  # type: ignore
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


def get_visits(tm: TensorMap, hd5: h5py.File, **kwargs) -> List[str]:
    visits = None
    if "visits" in kwargs:
        visits = kwargs["visits"]
        if isinstance(visits, str):
            visits = [visits]
        elif not isinstance(visits, list):
            raise TypeError(f"{kwargs['visits']} is not a List[str] or [str].")
    if visits is None:
        if tm.path_prefix:
            visits = list(hd5[tm.path_prefix.split("*")[0]])
        else:
            raise TypeError("Unable to get_visits with the inputs given.")
    return visits


def get_local_timestamps(time_array: np.ndarray) -> np.ndarray:
    # Check if time_array falls in a time change (if timestamp is not nan)
    if not (pd.isnull(time_array[0]) or pd.isnull(time_array[-1])):
        init_dt = datetime.utcfromtimestamp(time_array[0])
        end_dt = datetime.utcfromtimestamp(time_array[-1])
        offset_init = TIMEZONE.utcoffset(  # type: ignore
            init_dt,
            is_dst=True,
        ).total_seconds()
        offset_end = TIMEZONE.utcoffset(  # type: ignore
            end_dt,
            is_dst=True,
        ).total_seconds()
        offsets = np.array([offset_init, offset_end])
    else:
        offsets = np.array([np.nan, np.nan])

    # Convert unix to local and readable timestamps
    if offsets[0] == offsets[1]:
        local_timearray = pd.to_datetime(time_array + offsets[0], unit="s")
        local_timearray = np.array(local_timearray, dtype="datetime64[us]")
    else:
        local_timearray = np.empty(np.size(time_array), dtype="datetime64[us]")
        for idx, time_value in enumerate(time_array):
            if not pd.isnull(time_value):
                time_dt = datetime.fromtimestamp(time_value, TIMEZONE)
                local_timearray[idx] = pd.to_datetime(
                    time_dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
                )
            else:
                local_timearray[idx] = np.datetime64("NaT")

    return local_timearray


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
        if len(field1) == 0:
            return True

        fields1 = map(_get_name_if_function, field1)
        fields2 = map(_get_name_if_function, field2)

        return all([f1 == f2] for f1, f2 in zip(sorted(fields1), sorted(fields2)))
    return _get_name_if_function(field1) == _get_name_if_function(field2)


def _get_name_if_function(field: Any) -> Any:
    """We assume 'field' is a function if it's 'callable()'"""
    if callable(field):
        field = field.__name__

    if isinstance(field, str):
        return field.replace("-", "").replace("_", "")
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
    labels = sorted(labels.items(), key=lambda cm: cm[1])  # type: ignore
    negative_label, negative_label_index = labels[0]  # type: ignore
    for label, index in labels:  # type: ignore
        if label.startswith(negative_label_prefix):  # type: ignore
            negative_label = label  # type: ignore
            negative_label_index = index  # type: ignore
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


def is_dynamic_shape(tm: TensorMap, num_samples: int) -> Tuple[bool, Tuple[int, ...]]:
    if tm.time_series_limit is not None:
        return True, (num_samples,) + tm.shape
    return False, tm.shape


def make_hd5_path(tm: TensorMap, date_key: str, value_key: str) -> str:
    return f"{tm.path_prefix}/{date_key}/{value_key}"


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

    # ICU Alarms tmaps
    from ml4c3.tensormap.icu_alarms import get_tmap as tmaps_icu_alarms # isort: skip
    tm = tmaps_icu_alarms(tmap_name)
    if tm:
        tmaps[tmap_name] = tm
        return tmaps

    # ICU Around event tmaps
    from ml4c3.tensormap.icu_around_event import get_tmap as tmaps_icu_around_event # isort: skip
    tm = tmaps_icu_around_event(tmap_name)
    if tm:
        tmaps[tmap_name] = tm
        return tmaps

    # ICU Bedmaster tmaps
    from ml4c3.tensormap.icu_bedmaster_signals import get_tmap as tmaps_icu_bedmaster_signals # isort: skip
    tm = tmaps_icu_bedmaster_signals(tmap_name)
    if tm:
        tmaps[tmap_name] = tm
        return tmaps

    # ICU ECG features tmaps
    from ml4c3.tensormap.icu_ecg_features import get_tmap as tmaps_icu_ecg_features # isort: skip
    tm = tmaps_icu_ecg_features(tmap_name)
    if tm:
        tmaps[tmap_name] = tm
        return tmaps

    # ICU Events tmaps
    from ml4c3.tensormap.icu_events import get_tmap as tmaps_icu_events # isort: skip
    tm = tmaps_icu_events(tmap_name)
    if tm:
        tmaps[tmap_name] = tm
        return tmaps

    # ICU first visit tmaps
    from ml4c3.tensormap.icu_first_visit_with_signal import get_tmap as tmaps_icu_first_visit # isort: skip
    tm = tmaps_icu_first_visit(tmap_name)
    if tm:
        tmaps[tmap_name] = tm
        return tmaps

    # ICU List signals tmaps
    from ml4c3.tensormap.icu_list_signals import get_tmap as tmaps_icu_list_signals # isort: skip
    tm = tmaps_icu_list_signals(tmap_name)
    if tm:
        tmaps[tmap_name] = tm
        return tmaps

    # ICU Measurements tmaps
    from ml4c3.tensormap.icu_measurements import get_tmap as tmaps_icu_measurements # isort: skip
    tm = tmaps_icu_measurements(tmap_name)
    if tm:
        tmaps[tmap_name] = tm
        return tmaps

    # ICU Medications tmaps
    from ml4c3.tensormap.icu_medications import get_tmap as tmaps_icu_medications # isort: skip
    tm = tmaps_icu_medications(tmap_name)
    if tm:
        tmaps[tmap_name] = tm
        return tmaps

    # ICU Standardized tmaps
    from ml4c3.tensormap.icu_normalized import get_tmap as tmaps_icu_normalized # isort: skip
    tm = tmaps_icu_normalized(tmap_name)
    if tm:
        tmaps[tmap_name] = tm
        return tmaps

    # ICU Standardized tmaps
    from ml4c3.tensormap.icu_static_around_event import get_tmap as tmaps_icu_static_around_event # isort: skip
    tm = tmaps_icu_static_around_event(tmap_name)
    if tm:
        tmaps[tmap_name] = tm
        return tmaps

   # ICU Standardized tmaps
    from ml4c3.tensormap.icu_static import get_tmap as tmaps_icu_static # isort: skip
    tm = tmaps_icu_static(tmap_name)
    if tm:
        tmaps[tmap_name] = tm
        return tmaps

    # Base tmaps: ECG
    from ml4c3.tensormap.ecg import tmaps as tmaps_ecg  # isort:skip
    tmaps.update(tmaps_ecg)
    if tmap_name in tmaps:
        return tmaps

    # Base tmaps: STS
    from ml4c3.tensormap.sts import tmaps as tmaps_sts  # isort:skip
    tmaps.update(tmaps_sts)
    if tmap_name in tmaps:
        return tmaps

    # Base tmaps: ECG labels
    from ml4c3.tensormap.ecg_labels import tmaps as tmaps_ecg_labels  # isort:skip
    tmaps.update(tmaps_ecg_labels)
    if tmap_name in tmaps:
        return tmaps

    # Base tmaps: ECG voltage
    from ml4c3.tensormap.ecg import update_tmaps_ecg_voltage  # isort:skip
    tmaps = update_tmaps_ecg_voltage(tmap_name=tmap_name, tmaps=tmaps)
    if tmap_name in tmaps:
        return tmaps

    # Modify: weighted loss
    from ml4c3.tensormap.updaters import update_tmaps_weighted_loss  # isort:skip
    tmaps = update_tmaps_weighted_loss(tmap_name=tmap_name, tmaps=tmaps)
    if tmap_name in tmaps:
        return tmaps

    # Modify: STS window (e.g. preop)
    from ml4c3.tensormap.sts import update_tmaps_sts_window  # isort:skip
    tmaps = update_tmaps_sts_window(tmap_name=tmap_name, tmaps=tmaps)
    if tmap_name in tmaps:
        return tmaps

    # Modify: time series
    from ml4c3.tensormap.updaters import update_tmaps_time_series  # isort:skip
    tmaps = update_tmaps_time_series(tmap_name=tmap_name, tmaps=tmaps)
    if tmap_name in tmaps:
        return tmaps

    # Modify: load predictions
    from ml4c3.tensormap.updaters import update_tmaps_model_predictions  # isort:skip
    tmaps = update_tmaps_model_predictions(tmap_name=tmap_name, tmaps=tmaps)
    if tmap_name in tmaps:
        return tmaps

    # fmt: on

    raise ValueError(
        f"{tmap_name} cannot be found in tmaps despite building every TMap we can",
    )
