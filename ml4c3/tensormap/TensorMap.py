# Imports: standard library
import os
import logging
from enum import Enum, auto
from typing import Any, Dict, List, Tuple, Union, Callable, Optional
from datetime import datetime

# Imports: third party
import numpy as np
import pandas as pd

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
from definitions.globals import TIMEZONE

Axis = Union[int, None]
Dates = Union[List[str], pd.Series]

DEFAULT_TIME_TO_EVENT_CHANNELS = {"event": 0, "follow_up_days": 1}
ChannelMap = Dict[str, int]

# pylint: disable=line-too-long, import-outside-toplevel, too-many-return-statements


class PatientData:
    """
    Wrapper around a patient's data from multiple sources.
    """

    def __init__(self, patient_id):
        self.data = dict()
        self.id = patient_id

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise KeyError(f"Patient dictionary keys must be strings: {key}")
        if len(key.split("/")) != 1:
            raise KeyError(f"Patient dictionary keys must not be nested: {key}")
        self.data[key] = value

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise KeyError(f"Patient dictionary keys must be strings: {key}")
        key = key.strip("/")
        splits = key.split("/", 1)
        try:
            if len(splits) == 1:
                return self.data[key]
            return self.data[splits[0]][splits[1]]
        except KeyError:
            raise KeyError(f"No {key} key in PatientData object")

    def __contains__(self, key):
        if not isinstance(key, str):
            raise KeyError(f"Patient dictionary keys must be strings: {key}")
        key = key.strip("/")
        splits = key.split("/", 1)
        if len(splits) == 1:
            return splits[0] in self.data
        return splits[0] in self.data and splits[1] in self.data[splits[0]]


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
        metrics: Optional[List[Union[str, Callable]]] = None,
        activation: Optional[Union[str, Callable]] = None,
        path_prefix: Optional[str] = None,
        loss_weight: float = 1.0,
        channel_map: Optional[Dict[str, int]] = None,
        augmenters: Optional[Union[Callable, List[Callable]]] = None,
        validators: Optional[Union[Callable, List[Callable]]] = None,
        normalizers: Optional[Union[Normalizer, List[Normalizer]]] = None,
        interpretation: Interpretation = Interpretation.CONTINUOUS,
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
        self.loss_weight = loss_weight
        self.path_prefix = path_prefix
        self.interpretation = interpretation
        self.linked_tensors = linked_tensors
        self.time_series_limit = time_series_limit

        if tensor_from_file is None:
            raise ValueError(f"{self.name}: tensor_from_file cannot be None")
        self.tensor_from_file = tensor_from_file

        # Infer channel map from interpretation
        if channel_map is None and self.is_time_to_event:
            channel_map = DEFAULT_TIME_TO_EVENT_CHANNELS
        self.channel_map = channel_map

        # Infer shape from channel map or interpretation
        if shape is None:
            if self.channel_map is not None:
                if self.is_time_to_event:
                    shape = (2,)
                else:
                    shape = (len(self.channel_map),)
            else:
                raise ValueError(f"{self.name}: cannot infer shape")
        self.shape = shape

        # Infer loss from interpretation and shape
        if loss is None:
            if self.is_categorical or self.is_language:
                loss = "categorical_crossentropy"
            elif self.is_continuous or self.is_timeseries or self.is_event:
                loss = "mse"
            elif self.is_survival_curve:
                loss = survival_likelihood_loss(self.shape[0] // 2)
            elif self.is_time_to_event:
                loss = cox_hazard_loss
            else:
                raise ValueError(f"{self.name}: cannot infer loss")
        self.loss = loss

        # Infer activation from interpretation
        if activation is None:
            if self.is_categorical or self.is_language:
                activation = "softmax"
            elif self.is_continuous or self.is_timeseries or self.is_event:
                activation = "linear"
            elif self.is_survival_curve or self.is_time_to_event:
                activation = "sigmoid"
            else:
                raise ValueError(f"{self.name}: cannot infer activation")
        self.activation = activation

        # Infer metrics from interpretation
        if metrics is None:
            if self.is_categorical:
                metrics = ["categorical_accuracy"]
                if self.axes == 1:
                    metrics += per_class_precision(self.channel_map)
                    metrics += per_class_recall(self.channel_map)
                elif self.axes == 2:
                    metrics += per_class_precision_3d(self.channel_map)
                    metrics += per_class_recall_3d(self.channel_map)
                elif self.axes == 3:
                    metrics += per_class_precision_4d(self.channel_map)
                    metrics += per_class_recall_4d(self.channel_map)
                elif self.axes == 4:
                    metrics += per_class_precision_5d(self.channel_map)
                    metrics += per_class_recall_5d(self.channel_map)
            elif self.is_continuous and self.shape[-1] == 1:
                metrics = [pearson]
            else:
                metrics = []
        self.metrics = metrics

        # Infer embedded dimensionality from shape
        if annotation_units is None:
            annotation_units = self.size
        self.annotation_units = annotation_units

        # Wrap augmenter, validator, and normalizer in lists
        if augmenters is not None and not isinstance(augmenters, list):
            augmenters = [augmenters]
        if validators is not None and not isinstance(validators, list):
            validators = [validators]
        if normalizers is not None and not isinstance(normalizers, list):
            normalizers = [normalizers]
        self.augmenters = augmenters
        self.validators = validators
        self.normalizers = normalizers

        if time_series_filter is None:
            time_series_filter = make_default_time_series_filter(path_prefix)
        self.time_series_filter = time_series_filter

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
        data: PatientData,
        augment: bool,
    ) -> np.ndarray:
        self.validate(tensor, data)
        tensor = self.augment(tensor) if augment else tensor
        tensor = self.normalize(tensor)
        return tensor

    def validate(self, tensor: np.ndarray, data: PatientData):
        if self.validators is not None:
            for validator in self.validators:  # type: ignore
                validator(self, tensor, data)

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


def get_visits(tm: TensorMap, data: PatientData, **kwargs) -> List[str]:
    visits = None
    if "visits" in kwargs:
        visits = kwargs["visits"]
        if isinstance(visits, str):
            visits = [visits]
        elif not isinstance(visits, list):
            raise TypeError(f"{kwargs['visits']} is not a List[str] or [str].")
    if visits is None:
        if tm.path_prefix:
            visits = list(data[tm.path_prefix.split("*")[0]])
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


def make_default_time_series_filter(
    path_prefix: Optional[str] = None,
) -> Callable[[PatientData], Dates]:
    if path_prefix is None:
        path_prefix = ""

    def default_time_series_filter(data: PatientData) -> Dates:
        return sorted(list(data[path_prefix]))

    return default_time_series_filter


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


def is_dynamic_shape(
    tm: TensorMap,
    num_samples: Optional[int] = None,
) -> Union[bool, Tuple[bool, Tuple[int, ...]]]:
    is_dynamic = tm.time_series_limit is not None
    if num_samples is not None:
        shape = tm.shape or tuple()
        if is_dynamic:
            return is_dynamic, (num_samples,) + shape
        return is_dynamic, shape
    return is_dynamic


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

    # ICU Signal tmaps
    from ml4c3.tensormap.icu_signals import get_tmap as tmaps_icu_signals # isort: skip
    tm = tmaps_icu_signals(tmap_name)
    if tm:
        tmaps[tmap_name] = tm
        return tmaps

    # ICU Event department tmaps
    from ml4c3.tensormap.icu_event_department import get_tmap as tmaps_icu_event_department # isort: skip
    tm = tmaps_icu_event_department(tmap_name)
    if tm:
        tmaps[tmap_name] = tm
        return tmaps

    # ICU Around event tmaps
    from ml4c3.tensormap.icu_around_event import get_tmap as tmaps_icu_around_event # isort: skip
    tm = tmaps_icu_around_event(tmap_name)
    if tm:
        tmaps[tmap_name] = tm
        return tmaps

    # ICU Around event explore tmaps
    from ml4c3.tensormap.icu_around_event_explore import get_tmap as tmaps_icu_around_explore # isort: skip
    tm = tmaps_icu_around_explore(tmap_name)
    if tm:
        tmaps[tmap_name] = tm
        return tmaps

    # ICU ECG features tmaps
    from ml4c3.tensormap.icu_ecg_features import get_tmap as tmaps_icu_ecg_features # isort: skip
    tm = tmaps_icu_ecg_features(tmap_name)
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

    # ICU Standardized tmaps
    from ml4c3.tensormap.icu_normalized import get_tmap as tmaps_icu_normalized # isort: skip
    tm = tmaps_icu_normalized(tmap_name)
    if tm:
        tmaps[tmap_name] = tm
        return tmaps

    # ICU Static around event tmaps
    from ml4c3.tensormap.icu_static_around_event import get_tmap as tmaps_icu_static_around_event # isort: skip
    tm = tmaps_icu_static_around_event(tmap_name)
    if tm:
        tmaps[tmap_name] = tm
        return tmaps

    # Base tmaps: ECG
    from ml4c3.tensormap.ecg import tmaps as tmaps_ecg # isort:skip
    tmaps.update(tmaps_ecg)
    if tmap_name in tmaps:
        return tmaps

    # Base tmaps: STS
    from ml4c3.tensormap.sts import tmaps as tmaps_sts # isort:skip
    tmaps.update(tmaps_sts)
    if tmap_name in tmaps:
        return tmaps

    # Base tmaps: Echo
    from ml4c3.tensormap.echo import tmaps as tmaps_echo  # isort:skip
    tmaps.update(tmaps_echo)
    if tmap_name in tmaps:
        return tmaps

    # Base tmaps: ECG labels
    from ml4c3.tensormap.ecg_labels import tmaps as tmaps_ecg_labels # isort:skip
    tmaps.update(tmaps_ecg_labels)
    if tmap_name in tmaps:
        return tmaps

    # Base tmaps: ECG voltage
    from ml4c3.tensormap.ecg import update_tmaps_ecg_voltage # isort:skip
    tmaps = update_tmaps_ecg_voltage(tmap_name=tmap_name, tmaps=tmaps)
    if tmap_name in tmaps:
        return tmaps

    # Modify: Weighted loss
    from ml4c3.tensormap.updaters import update_tmaps_weighted_loss # isort:skip
    tmaps = update_tmaps_weighted_loss(tmap_name=tmap_name, tmaps=tmaps)
    if tmap_name in tmaps:
        return tmaps

    # Modify: Window (e.g. "pre_echo" or "pre_sts")
    from ml4c3.tensormap.updaters import update_tmaps_window # isort:skip
    tmaps = update_tmaps_window(tmap_name=tmap_name, tmaps=tmaps)
    if tmap_name in tmaps:
        return tmaps

    # Modify: Time series
    from ml4c3.tensormap.updaters import update_tmaps_time_series # isort:skip
    tmaps = update_tmaps_time_series(tmap_name=tmap_name, tmaps=tmaps)
    if tmap_name in tmaps:
        return tmaps

    # fmt: on

    raise ValueError(
        f"{tmap_name} cannot be found in tmaps despite building every TMap we can",
    )


def create_tmap(signal, source, field) -> Optional[TensorMap]:
    tm_name = f"{signal}_{field}"
    if source in ["vital", "waveform"]:
        # Imports: first party
        from ml4c3.tensormap.icu_signals import get_bedmaster_signal_tmap

        return get_bedmaster_signal_tmap(tm_name, signal, source)

    if source == "alarm":
        # Imports: first party
        from ml4c3.tensormap.icu_signals import create_alarm_tmap

        return create_alarm_tmap(tm_name, signal)

    if source in ["flowsheet", "labs"]:
        # Imports: first party
        from ml4c3.tensormap.icu_signals import create_measurement_tmap

        return create_measurement_tmap(tm_name, signal, source)
    if source == "med":
        # Imports: first party
        from ml4c3.tensormap.icu_signals import create_med_tmap

        return create_med_tmap(tm_name, signal)
    if source in ["events", "surgery", "transfusions", "procedures"]:
        # Imports: first party
        from ml4c3.tensormap.icu_signals import create_event_tmap

        return create_event_tmap(tm_name, signal, source)

    raise ValueError(
        f"Could not create tmap for field {field} of signal {signal} and source {source}",
    )
