# pylint: disable=attribute-defined-outside-init
# Imports: standard library
import os
import logging
from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Sequence
from itertools import chain

# Imports: third party
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import log_loss, hinge_loss
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import backend as K
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.layers import (
    ELU,
    Add,
    Dense,
    Input,
    Layer,
    PReLU,
    Conv1D,
    Conv2D,
    Conv3D,
    Dropout,
    Flatten,
    Reshape,
    LeakyReLU,
    Activation,
    InputLayer,
    Concatenate,
    MaxPooling1D,
    MaxPooling2D,
    MaxPooling3D,
    UpSampling1D,
    UpSampling2D,
    UpSampling3D,
    DepthwiseConv2D,
    SeparableConv1D,
    SeparableConv2D,
    ThresholdedReLU,
    AveragePooling1D,
    AveragePooling2D,
    AveragePooling3D,
    SpatialDropout1D,
    SpatialDropout2D,
    SpatialDropout3D,
    BatchNormalization,
    LayerNormalization,
    concatenate,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import (
    History,
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.regularizers import Regularizer, l1_l2

# Imports: first party
from ml4c3.plots import plot_metric_history, plot_condensed_architecture_diagram
from ml4c3.metrics import get_metric_dict
from ml4c3.datasets import (
    BATCH_INPUT_INDEX,
    BATCH_OUTPUT_INDEX,
    get_array_from_dict_of_arrays,
    get_dicts_of_arrays_from_dataset,
)
from ml4c3.optimizers import NON_KERAS_OPTIMIZERS, get_optimizer
from definitions.models import BottleneckType
from definitions.globals import MODEL_EXT
from ml4c3.tensormap.TensorMap import TensorMap

#######################################
##                                   ##
## Defines and Enums                 ##
##                                   ##
#######################################

tfd = tfp.distributions
Tensor = tf.Tensor
SKLEARN_MODELS = Union[
    LogisticRegression,
    LinearSVC,
    RandomForestClassifier,
    XGBClassifier,
]


#######################################
##                                   ##
## Model Training Functions          ##
##                                   ##
#######################################


def train_model_from_datasets(
    model: Union[Model, SKLEARN_MODELS],
    tensor_maps_in: List[TensorMap],
    tensor_maps_out: List[TensorMap],
    train_dataset: tf.data.Dataset,
    valid_dataset: Optional[tf.data.Dataset],
    epochs: int,
    patience: int,
    learning_rate_patience: int,
    learning_rate_reduction: float,
    output_folder: str,
    image_ext: str,
    return_history: bool = False,
    plot: bool = True,
) -> Union[Model, Tuple[Model, History]]:
    """
    Train a model from tensorflow.data.Datasets for validation and training data.

    Training data lives on disk and is dynamically loaded by the Datasets.
    Plots the metric history after training. Creates a directory to save weights,
    if necessary.

    :param model: The model to optimize
    :param tensor_maps_in: List of input TensorMaps
    :param tensor_maps_out: List of output TensorMaps
    :param train_dataset: Dataset that yields batches of training data
    :param valid_dataset: Dataset that yields batches of validation data.
           Scikit-learn models do not utilize the validation split.
    :param epochs: Maximum number of epochs to run regardless of Early Stopping
    :param patience: Number of epochs to wait before reducing learning rate
    :param learning_rate_patience: Number of epochs without validation loss improvement
           to wait before reducing learning rate
    :param learning_rate_reduction: Scale factor to reduce learning rate by
    :param output_folder: Directory where output file will be stored
    :param image_ext: File format of saved image
    :param return_history: Whether or not to return history from training
    :param plot: Whether or not to plot metrics from training
    :return: The optimized model which achieved the best validation loss or training
             loss if validation data was not provided
    """
    model_file = os.path.join(output_folder, "model_weights" + MODEL_EXT)
    if not os.path.exists(os.path.dirname(model_file)):
        os.makedirs(os.path.dirname(model_file))

    # If keras instead of sklearn model
    if isinstance(model, Model):
        history = model.fit(
            x=train_dataset,
            epochs=epochs,
            verbose=1,
            validation_data=valid_dataset,
            callbacks=_get_callbacks(
                model_file=model_file,
                patience=patience,
                learning_rate_patience=learning_rate_patience,
                learning_rate_reduction=learning_rate_reduction,
                monitor="loss" if valid_dataset is None else "val_loss",
            ),
        )
        logging.info(f"Model weights saved at: {model_file}")

        if plot:
            plot_metric_history(
                history=history,
                image_ext=image_ext,
                prefix=os.path.dirname(model_file),
            )

        # load the weights from model which achieved the best validation loss
        model.load_weights(model_file)
        if return_history:
            return model, history

    # If sklearn model
    elif isinstance(model, SKLEARN_MODELS.__args__):

        # Get dicts of arrays of data keyed by tmap name from the dataset
        data = get_dicts_of_arrays_from_dataset(dataset=train_dataset)
        input_data, output_data = data[BATCH_INPUT_INDEX], data[BATCH_OUTPUT_INDEX]

        # Get desired arrays from dicts of arrays
        X = get_array_from_dict_of_arrays(
            tensor_maps=tensor_maps_in,
            data=input_data,
            drop_redundant_columns=False,
        )
        y = get_array_from_dict_of_arrays(
            tensor_maps=tensor_maps_out,
            data=output_data,
            drop_redundant_columns=True,
        )

        # Fit sklearn model
        model.fit(X, y)
        logging.info(f"{model.name} trained on data array with shape {X.shape}")
    else:
        raise NotImplementedError(
            "Cannot get data for inference from data of type "
            f"{type(data).__name__}: {data}",
        )
    return model


def sklearn_model_loss_from_dataset(
    model: SKLEARN_MODELS,
    dataset: tf.data.Dataset,
    tensor_maps_in: List[TensorMap],
    tensor_maps_out: List[TensorMap],
) -> Tuple[float, str]:
    """
    Given a sklearn model and a dataset, computes the logistic loss.
    """
    if not isinstance(model, SKLEARN_MODELS.__args__):
        raise ValueError(f"Uknown sklearn model: {model}")
    # Get dicts of arrays of data keyed by tmap name from the dataset
    data = get_dicts_of_arrays_from_dataset(dataset=dataset)
    input_data, output_data = data[BATCH_INPUT_INDEX], data[BATCH_OUTPUT_INDEX]

    # Get desired arrays from dicts of arrays
    X = get_array_from_dict_of_arrays(
        tensor_maps=tensor_maps_in,
        data=input_data,
        drop_redundant_columns=False,
    )
    y = get_array_from_dict_of_arrays(
        tensor_maps=tensor_maps_out,
        data=output_data,
        drop_redundant_columns=True,
    )
    loss = model.loss_function(y, model.predict_proba(X))
    return loss, model.loss_name


#######################################
##                                   ##
## Model Factory Functions           ##
##                                   ##
#######################################


def make_model(args):
    """
    Create a model to train according the input arguments.
    """
    if args.recipe in ["train", "train_simclr", "infer", "build"]:
        if args.model_file is None:
            plot_condensed_architecture_diagram(args)
        model = make_multimodal_multitask_model(**args.__dict__)
    elif args.recipe == "train_keras_logreg":
        model = make_shallow_model(
            tensor_maps_in=args.tensor_maps_in,
            tensor_maps_out=args.tensor_maps_out,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            learning_rate_schedule=args.learning_rate_schedule,
            model_file=args.model_file,
            donor_layers=args.donor_layers,
            l1=args.l1,
            l2=args.l2,
        )
    else:
        hyperparameters = {}
        if args.recipe == "train_sklearn_logreg":
            if args.l1 == 0 and args.l2 == 0:
                c = 1e7
            else:
                c = 1 / (args.l1 + args.l2)
            hyperparameters["c"] = c
            hyperparameters["l1_ratio"] = c * args.l1
        elif args.mode == "train_sklearn_svm":
            hyperparameters["c"] = args.c
        elif args.recipe == "train_sklearn_randomforest":
            hyperparameters["n_estimators"] = args.n_estimators
            hyperparameters["max_depth"] = args.max_depth
            hyperparameters["min_samples_split"] = args.min_samples_split
            hyperparameters["min_samples_leaf"] = args.min_samples_leaf
        elif args.recipe == "train_sklearn_xgboost":
            hyperparameters["n_estimators"] = args.n_estimators
            hyperparameters["max_depth"] = args.max_depth
            hyperparameters["gamma"] = args.gamma
            hyperparameters["l1_ratio"] = args.l1
            hyperparameters["l2_ratio"] = args.l2
        else:
            raise ValueError("Unknown train mode: ", args.recipe)
        assert len(args.tensor_maps_out) == 1
        model_type = args.recipe.split("_")[-1]
        model = make_sklearn_model(
            model_type=model_type,
            hyperparameters=hyperparameters,
        )
    return model


def make_shallow_model(
    tensor_maps_in: List[TensorMap],
    tensor_maps_out: List[TensorMap],
    optimizer: str,
    learning_rate: float,
    learning_rate_schedule: str,
    donor_layers: str = None,
    model_file: str = None,
    l1: float = 0,
    l2: float = 0,
    **kwargs,
) -> Model:
    """Make a shallow model (e.g. linear or logistic regression)

    :param tensor_maps_in: List of input TensorMaps
    :param tensor_maps_out: List of output TensorMaps
    :param optimizer: which optimizer to use. See optimizers.py.
    :param learning_rate: Size of learning steps in SGD optimization
    :param learning_rate_schedule: learning rate schedule to train with, e.g. triangular
    :param donor_layers: Optional HD5 model file whose weights will be loaded into
                         this model when layer names match.
    :param model_file: Optional HD5 model file to load and return.
    :param l1: Optional float value to use for L1 regularization.
    :param l2: Optional float value to use for L2 regularization.
    :return: a compiled keras model
    """
    if model_file is not None:
        m = load_model(model_file, custom_objects=get_metric_dict(tensor_maps_out))
        m.summary()
        logging.info(f"Loaded model file from: {model_file}")
        return m

    losses = []
    outputs = []
    my_metrics = {}
    loss_weights = []
    regularizer = l1_l2(l1=l1, l2=l2)

    input_tensors = [Input(shape=tm.shape, name=tm.input_name) for tm in tensor_maps_in]
    it = concatenate(input_tensors) if len(input_tensors) > 1 else input_tensors[0]

    for ot in tensor_maps_out:
        losses.append(ot.loss)
        loss_weights.append(ot.loss_weight)
        my_metrics[ot.output_name] = ot.metrics
        outputs.append(
            Dense(
                units=len(ot.channel_map) if ot.channel_map else 1,
                activation=ot.activation,
                name=ot.output_name,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
            )(it),
        )

    opt = get_optimizer(
        name=optimizer,
        learning_rate=learning_rate,
        learning_rate_schedule=learning_rate_schedule,
        optimizer_kwargs=kwargs.get("optimizer_kwargs"),
    )

    model = Model(inputs=input_tensors, outputs=outputs)
    model.compile(
        optimizer=opt,
        loss=losses,
        loss_weights=loss_weights,
        metrics=my_metrics,
    )
    model.summary()
    if donor_layers is not None:
        model.load_weights(donor_layers, by_name=True)
        logging.info(f"Loaded model weights from: {donor_layers}")

    return model


def make_sklearn_model(
    model_type: str,
    hyperparameters: Dict[str, float],
) -> SKLEARN_MODELS:
    """
    Initialize and return a scikit-learn model.

    :param model_type: String defining type of scikit-learn model to initialize
    :param hyperparameters: Dict of hyperparameter names and values

    """
    if model_type == "logreg":
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=5000,
            C=hyperparameters["c"],
            l1_ratio=hyperparameters["l1_ratio"],
        )
        model.loss_function = log_loss
        model.loss_name = "log"
    elif model_type == "svm":
        model = MySVM(class_weight="balanced", C=hyperparameters["c"])
        model.loss_function = hinge_loss
        model.loss_name = "hinge"
    elif model_type == "randomforest":
        model = RandomForestClassifier(
            class_weight="balanced",
            n_estimators=hyperparameters["n_estimators"],
            max_depth=hyperparameters["max_depth"],
            min_samples_split=hyperparameters["min_samples_split"],
            min_samples_leaf=hyperparameters["min_samples_leaf"],
        )
        model.loss_function = _gini_loss
        model.loss_name = "gini"
    elif model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=hyperparameters["n_estimators"],
            max_depth=hyperparameters["max_depth"],
            gamma=hyperparameters["gamma"],
            reg_alpha=hyperparameters["l1_ratio"],
            reg_lambda=hyperparameters["l2_ratio"],
        )
        model.loss_function = log_loss
        model.loss_name = "log"
    else:
        raise ValueError(f"Invalid model name: {model_type}")
    model.name = model_type
    return model


def make_multimodal_multitask_model(
    tensor_maps_in: List[TensorMap],
    tensor_maps_out: List[TensorMap],
    conv_type: str = None,
    conv_blocks: List[int] = None,
    conv_block_size: int = None,
    conv_block_layer_order: List[str] = None,
    residual_blocks: List[int] = None,
    residual_block_size: int = None,
    residual_block_layer_order: List[str] = None,
    dense_blocks: List[int] = None,
    dense_block_size: int = None,
    dense_block_layer_order: List[str] = None,
    conv_x: List[int] = None,
    conv_y: List[int] = None,
    conv_z: List[int] = None,
    conv_padding: str = None,
    pool_type: str = None,
    pool_x: int = None,
    pool_y: int = None,
    pool_z: int = None,
    bottleneck_type: BottleneckType = None,
    dense_layers: List[int] = None,
    activation_layer: str = None,
    normalization_layer: str = None,
    dense_dropout: float = 0,
    spatial_dropout: float = 0,
    dense_layer_order: List[str] = None,
    l1: float = 0,
    l2: float = 0,
    optimizer: str = None,
    learning_rate: float = None,
    learning_rate_schedule: str = None,
    model_file: str = None,
    donor_layers: str = None,
    nest_model: List[List[str]] = None,
    remap_layer: Dict[str, str] = None,
    freeze_donor_layers: bool = None,
    **kwargs,
) -> Model:
    custom_dict = _get_custom_objects(tensor_maps_out)
    opt = get_optimizer(
        name=optimizer,
        learning_rate=learning_rate,
        learning_rate_schedule=learning_rate_schedule,
        optimizer_kwargs=kwargs.get("optimizer_kwargs"),
    )
    if model_file is not None:
        logging.info(f"Attempting to load model file from: {model_file}")
        m = load_model(model_file, custom_objects=custom_dict, compile=False)
        m.compile(optimizer=opt, loss=custom_dict["loss"])
        m.summary()
        logging.info(f"Loaded model file from: {model_file}")
        return m

    # list of filter dimensions should match the number of blocks
    conv_blocks = conv_blocks or []
    residual_blocks = residual_blocks or []
    dense_blocks = dense_blocks or []
    num_conv = len(conv_blocks)
    num_residual = len(residual_blocks)
    num_dense = len(dense_blocks)
    num_filters_needed = num_conv + num_residual + num_dense

    conv_x = _repeat_dimension(
        dim=conv_x,
        name="x",
        num_filters_needed=num_filters_needed,
    )
    conv_y = _repeat_dimension(
        dim=conv_y,
        name="y",
        num_filters_needed=num_filters_needed,
    )
    conv_z = _repeat_dimension(
        dim=conv_z,
        name="z",
        num_filters_needed=num_filters_needed,
    )

    nest_model = nest_model or []
    nested_models = [
        (
            load_model(model_file, custom_objects=custom_dict, compile=False),
            hidden_layer,
        )
        for model_file, hidden_layer in nest_model
    ]
    nested_model_inputs = {
        input_layer.name
        for nested_model, _ in nested_models
        for input_layer in nested_model.inputs
    }

    regularizer = l1_l2(l1=l1, l2=l2)

    encoders: Dict[TensorMap, Encoder] = {}
    for tm in tensor_maps_in:
        # Do not create encoder for input which is input to a nested model
        if any(tm.input_name in input_layer for input_layer in nested_model_inputs):
            continue
        if tm.axes > 1:
            encoders[tm] = ConvEncoder(
                dimension=tm.axes,
                conv_type=conv_type,
                conv_blocks=conv_blocks,
                conv_block_size=conv_block_size,
                conv_block_layer_order=conv_block_layer_order,
                residual_blocks=residual_blocks,
                residual_block_size=residual_block_size,
                residual_block_layer_order=residual_block_layer_order,
                dense_blocks=dense_blocks,
                dense_block_size=dense_block_size,
                dense_block_layer_order=dense_block_layer_order,
                conv_x=conv_x,
                conv_y=conv_y,
                conv_z=conv_z,
                conv_padding=conv_padding,
                regularizer=regularizer,
                activation_layer=activation_layer,
                normalization_layer=normalization_layer,
                dropout_rate=spatial_dropout,
                pool_type=pool_type,
                pool_x=pool_x,
                pool_y=pool_y,
                pool_z=pool_z,
            )
        else:
            encoders[tm] = FullyConnectedEncoder(
                width=tm.annotation_units,
                activation_layer=activation_layer,
                normalization_layer=normalization_layer,
                dropout_rate=dense_dropout,
                layer_order=dense_layer_order,
                regularizer=regularizer,
            )

    pre_decoder_shapes: Dict[TensorMap, Optional[Tuple[int, ...]]] = {}
    for tm in tensor_maps_out:
        if tm.axes == 1:
            pre_decoder_shapes[tm] = None
        else:
            pre_decoder_shapes[tm] = _calc_start_shape(
                num_upsamples=len(conv_blocks),
                output_shape=tm.shape,
                upsample_rates=[pool_x, pool_y, pool_z],
                channels=conv_blocks[-1],
            )

    if bottleneck_type in {
        BottleneckType.FlattenRestructure,
        BottleneckType.GlobalAveragePoolStructured,
    }:
        bottleneck = ConcatenateRestructure(
            bottleneck_type=bottleneck_type,
            widths=dense_layers,
            activation_layer=activation_layer,
            normalization_layer=normalization_layer,
            dropout_rate=dense_dropout,
            layer_order=dense_layer_order,
            regularizer=regularizer,
            pre_decoder_shapes=pre_decoder_shapes,
        )
    elif bottleneck_type == BottleneckType.Variational:
        bottleneck = VariationalBottleNeck(
            widths=dense_layers[:-1],
            latent_size=dense_layers[-1],
            activation_layer=activation_layer,
            normalization_layer=normalization_layer,
            dropout_rate=dense_dropout,
            layer_order=dense_layer_order,
            regularizer=regularizer,
            pre_decoder_shapes=pre_decoder_shapes,
        )
    else:
        raise NotImplementedError(f"Unknown BottleneckType {bottleneck_type}.")

    decoders: Dict[TensorMap, Decoder] = {}
    for tm in tensor_maps_out:
        if tm.axes > 1:
            decoders[tm] = ConvDecoder(
                tm=tm,
                conv_type=conv_type,
                conv_blocks=conv_blocks,
                conv_block_size=conv_block_size,
                conv_block_layer_order=conv_block_layer_order,
                conv_layer_type=conv_type,
                conv_x=conv_x,
                conv_y=conv_y,
                conv_z=conv_z,
                conv_padding=conv_padding,
                regularizer=regularizer,
                activation_layer=activation_layer,
                normalization_layer=normalization_layer,
                dropout_rate=spatial_dropout,
                upsample_x=pool_x,
                upsample_y=pool_y,
                upsample_z=pool_z,
            )
        else:
            decoders[tm] = FullyConnectedDecoder(tm=tm)

    m = _make_multimodal_multitask_model(
        encoders=encoders,
        bottleneck=bottleneck,
        decoders=decoders,
        nested_models=nested_models,
        freeze=freeze_donor_layers or False,
    )

    # Load layers for transfer learning
    if donor_layers is not None:
        loaded = 0
        freeze = freeze_donor_layers or False
        layer_map = remap_layer or dict()
        try:
            m_other = load_model(
                donor_layers,
                custom_objects=custom_dict,
                compile=False,
            )
            for other_layer in m_other.layers:
                try:
                    other_layer_name = other_layer.name
                    if other_layer_name in layer_map:
                        other_layer_name = layer_map[other_layer_name]
                    target_layer = m.get_layer(other_layer_name)
                    target_layer.set_weights(other_layer.get_weights())
                    loaded += 1
                    if freeze:
                        target_layer.trainable = False
                except (ValueError, KeyError):
                    logging.warning(
                        f"Error loading layer {other_layer.name} from model:"
                        f" {donor_layers}. Will still try to load other layers.",
                    )
        except ValueError as e:
            logging.info(
                f"Loaded model weights, but got ValueError in model loading: {str(e)}",
            )
        logging.info(
            f'Loaded {"and froze " if freeze else ""}{loaded} layers from'
            f" {donor_layers}.",
        )
    m.compile(
        optimizer=opt,
        loss=[tm.loss for tm in tensor_maps_out],
        metrics={tm.output_name: tm.metrics for tm in tensor_maps_out},
    )
    m.summary()
    return m


#######################################
##                                   ##
## Helper Model Classes              ##
##                                   ##
#######################################


class MySVM(LinearSVC):
    """
    Modified Linear Support Vector Classification Sklearn class.
    """

    def __init__(self, class_weight, C):
        super().__init__(
            penalty="l2",
            dual=False,
            class_weight=class_weight,
            C=C,
        )

    def fit(self, x, y):
        # Fit internal LinearSVC to obtain coefs
        self.LSVC = LinearSVC(
            penalty=self.penalty,
            dual=self.dual,
            class_weight=self.class_weight,
            C=self.C,
        )

        # Fit LinearSVC using all features
        self.LSVC.fit(x, y)

        # Wrap internal SVC in calibrated model to enable predict_proba
        self.calibratedmodel = CalibratedClassifierCV(base_estimator=self.LSVC, cv=5)

        # Train calibrated model on top features
        self.calibratedmodel.fit(x, y)
        return self

    def predict_proba(self, x):
        """
        Call predict_proba on internal linear calibrated SVC
        """
        return self.calibratedmodel.predict_proba(x)


# Blocks


class ConvolutionalBlock:
    def __init__(
        self,
        *,
        dimension: int,
        block_size: int,
        conv_type: str,
        filters: int,
        conv_x: int,
        conv_y: int,
        conv_z: int,
        conv_padding: str,
        regularizer: Regularizer,
        activation_layer: str,
        normalization_layer: str,
        dropout_rate: float,
        layer_order: List[str],
        pool_type: str,
        pool_x: int,
        pool_y: int,
        pool_z: int,
    ):
        conv_layer, kernel, regularizer_args = _conv_layer(
            dimension=dimension,
            conv_type=conv_type,
            conv_x=conv_x,
            conv_y=conv_y,
            conv_z=conv_z,
            regularizer=regularizer,
        )
        self.conv_layers = [
            conv_layer(
                filters=filters,
                kernel_size=kernel,
                padding=conv_padding,
                **regularizer_args,
            )
            for _ in range(block_size)
        ]
        self.activation_layers = [
            _activation_layer(activation_layer) for _ in range(block_size)
        ]
        self.normalization_layers = [
            _normalization_layer(normalization_layer) for _ in range(block_size)
        ]
        self.dropout_layers = [
            _dropout_layer(dimension, dropout_rate) for _ in range(block_size)
        ]
        self.layer_order = layer_order
        self.pool_layer = _pool_layer(
            dimension,
            pool_type,
            pool_x,
            pool_y,
            pool_z,
        )

    def __call__(self, x: Tensor) -> Tensor:
        for convolve, activate, normalize, dropout in zip(
            self.conv_layers,
            self.activation_layers,
            self.normalization_layers,
            self.dropout_layers,
        ):
            x = _order_layers(
                layer_order=self.layer_order,
                conv_or_dense_layer=convolve,
                activation_layer=activate,
                normalization_layer=normalize,
                dropout_layer=dropout,
            )(x)
        x = self.pool_layer(x)
        return x


class ResidualBlock(ConvolutionalBlock):
    def __init__(
        self,
        *,
        dimension: int,
        conv_type: str,
        filters: int,
        conv_padding: str,
        regularizer: Regularizer,
        **kwargs,
    ):
        super().__init__(
            dimension=dimension,
            conv_type=conv_type,
            filters=filters,
            conv_padding=conv_padding,
            regularizer=regularizer,
            **kwargs,
        )
        conv_layer, kernel, regularizer_args = _conv_layer(
            dimension=dimension,
            conv_type=conv_type,
            conv_x=1,
            conv_y=1,
            conv_z=1,
            regularizer=regularizer,
        )
        self.match_dimensions = conv_layer(
            filters=filters,
            kernel_size=kernel,
            padding=conv_padding,
            **regularizer_args,
        )

    def __call__(self, x: Tensor) -> Tensor:
        initial = self.match_dimensions(x)
        for convolve, activate, normalize, dropout in zip(
            self.conv_layers,
            self.activation_layers,
            self.normalization_layers,
            self.dropout_layers,
        ):
            x = _order_layers(
                layer_order=self.layer_order,
                conv_or_dense_layer=convolve,
                activation_layer=activate,
                normalization_layer=normalize,
                dropout_layer=dropout,
            )(x)

            # TODO add residual_block_bottleneck
        x = Add()([initial, x])
        x = self.pool_layer(x)
        return x


class DenseBlock(ConvolutionalBlock):
    def __call__(self, x: Tensor) -> Tensor:
        for i, (convolve, activate, normalize, dropout) in enumerate(
            zip(
                self.conv_layers,
                self.activation_layers,
                self.normalization_layers,
                self.dropout_layers,
            ),
        ):
            last_concat = x
            x = _order_layers(
                layer_order=self.layer_order,
                conv_or_dense_layer=convolve,
                activation_layer=activate,
                normalization_layer=normalize,
                dropout_layer=dropout,
            )(x)

            # TODO add dense_block_bottleneck

            # Last convolutional layer in our dense block is the transition layer
            # between blocks described the original paper. As such, prior layers are
            # not concatenated to the final transition convolutional layer.
            if i < len(self.conv_layers) - 1:
                x = Concatenate()([x, last_concat])
        x = self.pool_layer(x)
        return x


class FullyConnectedBlock:
    def __init__(
        self,
        *,
        widths: List[int],
        activation_layer: str,
        normalization_layer: str,
        dropout_rate: float,
        layer_order: List[str],
        regularizer: Regularizer,
    ):
        self.dense_layers = [
            Dense(
                units=width,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
            )
            for width in widths
        ]
        self.activation_layers = [_activation_layer(activation_layer) for _ in widths]
        self.normalization_layers = [
            _normalization_layer(normalization_layer) for _ in widths
        ]
        self.dropout_layers = [_dropout_layer(1, dropout_rate) for _ in widths]
        self.layer_order = layer_order

    def __call__(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        for dense, normalize, activate, dropout in zip(
            self.dense_layers,
            self.normalization_layers,
            self.activation_layers,
            self.dropout_layers,
        ):
            x = _order_layers(
                layer_order=self.layer_order,
                conv_or_dense_layer=dense,
                activation_layer=activate,
                normalization_layer=normalize,
                dropout_layer=dropout,
            )(x)
        return x


# Bottlenecks


class ConcatenateRestructure:
    """
    Flattens or GAPs then concatenates all inputs, applies a dense layer, then
    restructures to provided shapes
    """

    def __init__(
        self,
        bottleneck_type: BottleneckType,
        widths: List[int],
        activation_layer: str,
        normalization_layer: str,
        dropout_rate: float,
        layer_order: List[str],
        regularizer: Regularizer,
        pre_decoder_shapes: Dict[TensorMap, Optional[Tuple[int, ...]]],
    ):
        self.bottleneck_type = bottleneck_type
        self.fully_connected = FullyConnectedBlock(
            widths=widths,
            activation_layer=activation_layer,
            normalization_layer=normalization_layer,
            dropout_rate=dropout_rate,
            layer_order=layer_order,
            regularizer=regularizer,
        )
        self.restructures = {}
        self.no_restructures = []
        for tm, shape in pre_decoder_shapes.items():
            if shape is None:
                self.no_restructures.append(tm)
            else:
                self.restructures[tm] = FlatToStructure(
                    output_shape=shape,
                    activation_layer=activation_layer,
                    normalization_layer=normalization_layer,
                    layer_order=layer_order,
                    regularizer=regularizer,
                )

    def __call__(
        self,
        encoder_outputs: Dict[TensorMap, Tensor],
    ) -> Dict[TensorMap, Tensor]:
        if self.bottleneck_type == BottleneckType.FlattenRestructure:
            y = [Flatten()(x) for x in encoder_outputs.values()]
        elif self.bottleneck_type == BottleneckType.GlobalAveragePoolStructured:
            y = [
                Flatten()(x) for tm, x in encoder_outputs.items() if len(x.shape) == 2
            ]  # Flat tensors
            y += [
                _global_average_pool(x)
                for tm, x in encoder_outputs.items()
                if len(x.shape) > 2
            ]  # Structured tensors
        else:
            raise NotImplementedError(
                f"bottleneck_type {self.bottleneck_type} does not exist.",
            )
        if len(y) > 1:
            y = concatenate(y)
        else:
            y = y[0]
        y = self.fully_connected(y) if self.fully_connected else y
        outputs: Dict[TensorMap, Tensor] = {}
        return {
            **{tm: restructure(y) for tm, restructure in self.restructures.items()},
            **{tm: y for tm in self.no_restructures if tm not in outputs},
            **outputs,
        }


class VariationalBottleNeck:
    def __init__(
        self,
        widths: List[int],
        activation_layer: str,
        normalization_layer: str,
        dropout_rate: float,
        layer_order: List[str],
        regularizer: Regularizer,
        latent_size: int,
        pre_decoder_shapes: Dict[TensorMap, Optional[Tuple[int, ...]]],
    ):
        self.latent_size = latent_size
        self.regularizer = regularizer
        self.sampler: Callable = VariationalDiagNormal(latent_size)
        self.fully_connected = FullyConnectedBlock(
            widths=widths,
            activation_layer=activation_layer,
            normalization_layer=normalization_layer,
            dropout_rate=dropout_rate,
            layer_order=layer_order,
            regularizer=regularizer,
        )
        self.restructures = {}
        self.no_restructures = []
        for tm, shape in pre_decoder_shapes.items():
            if shape is None:
                self.no_restructures.append(tm)
            else:
                self.restructures[tm] = FlatToStructure(
                    output_shape=shape,
                    activation_layer=activation_layer,
                    normalization_layer=normalization_layer,
                    layer_order=layer_order,
                    regularizer=regularizer,
                )

    def __call__(
        self,
        encoder_outputs: Dict[TensorMap, Tensor],
    ) -> Dict[TensorMap, Tensor]:
        y = [Flatten()(x) for x in encoder_outputs.values()]
        if len(y) > 1:
            y = concatenate(y)
        else:
            y = y[0]
        y = self.fully_connected(y) if self.fully_connected else y
        mu = Dense(self.latent_size, name="embed")(y)
        log_sigma = Dense(
            units=self.latent_size,
            name="log_sigma",
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer,
        )(y)
        y = self.sampler(mu, log_sigma)
        return {
            **{tm: restructure(y) for tm, restructure in self.restructures.items()},
            **{tm: y for tm in self.no_restructures},
        }


class FlatToStructure:
    """Takes a flat input, applies a dense layer, then restructures to output_shape"""

    def __init__(
        self,
        output_shape: Tuple[int, ...],
        activation_layer: str,
        normalization_layer: str,
        layer_order: List[str],
        regularizer: Regularizer,
    ):
        self.input_shapes = output_shape
        self.dense = Dense(
            units=int(np.prod(output_shape)),
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
        )
        self.activation_layer = _activation_layer(activation_layer)
        self.normalization_layer = _normalization_layer(normalization_layer)
        self.layer_order = layer_order
        self.reshape = Reshape(output_shape)

    def __call__(self, x: Tensor) -> Tensor:
        x = _order_layers(
            layer_order=self.layer_order,
            conv_or_dense_layer=self.dense,
            activation_layer=self.activation_layer,
            normalization_layer=self.normalization_layer,
            dropout_layer=lambda a: a,
        )(x)
        x = self.reshape(x)
        return x


class VariationalDiagNormal(Layer):
    def __init__(self, latent_size: int, **kwargs):
        self.latent_size = latent_size
        super(VariationalDiagNormal, self).__init__(**kwargs)
        self.prior = tfd.MultivariateNormalDiag(
            loc=tf.zeros([latent_size]),
            scale_identity_multiplier=1.0,
        )

    def call(self, mu: Tensor, log_sigma: Tensor, **kwargs):
        """mu and sigma must be shape (None, latent_size)"""
        approx_posterior = tfd.MultivariateNormalDiag(
            loc=mu,
            scale_diag=tf.math.exp(log_sigma),
        )
        kl = tf.reduce_mean(tfd.kl_divergence(approx_posterior, self.prior))
        self.add_loss(kl)
        self.add_metric(kl, aggregation="mean", name="KL_divergence")
        return approx_posterior.sample()

    def get_config(self):
        return {"latent_size": self.latent_size}


# Encoders


class ConvEncoder:
    def __init__(
        self,
        *,
        dimension: int,
        conv_type: str,
        conv_blocks: List[int],
        conv_block_size: int,
        conv_block_layer_order: List[str],
        residual_blocks: List[int],
        residual_block_size: int,
        residual_block_layer_order: List[str],
        dense_blocks: List[int],
        dense_block_size: int,
        dense_block_layer_order: List[str],
        conv_x: List[int],
        conv_y: List[int],
        conv_z: List[int],
        conv_padding: str,
        regularizer: Regularizer,
        activation_layer: str,
        normalization_layer: str,
        dropout_rate: float,
        pool_type: str,
        pool_x: int,
        pool_y: int,
        pool_z: int,
    ):
        self.conv_blocks = [
            ConvolutionalBlock(
                dimension=dimension,
                block_size=conv_block_size,
                conv_type=conv_type,
                filters=filters,
                conv_x=filter_x,
                conv_y=filter_y,
                conv_z=filter_z,
                conv_padding=conv_padding,
                regularizer=regularizer,
                activation_layer=activation_layer,
                normalization_layer=normalization_layer,
                dropout_rate=dropout_rate,
                layer_order=conv_block_layer_order,
                pool_type=pool_type,
                pool_x=pool_x,
                pool_y=pool_y,
                pool_z=pool_z,
            )
            for filters, filter_x, filter_y, filter_z in zip(
                conv_blocks,
                conv_x[: len(conv_blocks)],
                conv_y[: len(conv_blocks)],
                conv_z[: len(conv_blocks)],
            )
        ]
        self.res_blocks = [
            ResidualBlock(
                dimension=dimension,
                block_size=residual_block_size,
                conv_type=conv_type,
                filters=filters,
                conv_x=filter_x,
                conv_y=filter_y,
                conv_z=filter_z,
                conv_padding=conv_padding,
                regularizer=regularizer,
                activation_layer=activation_layer,
                normalization_layer=normalization_layer,
                dropout_rate=dropout_rate,
                layer_order=residual_block_layer_order,
                pool_type=pool_type,
                pool_x=pool_x,
                pool_y=pool_y,
                pool_z=pool_z,
            )
            for filters, filter_x, filter_y, filter_z in zip(
                residual_blocks,
                conv_x[len(conv_blocks) : len(conv_blocks) + len(residual_blocks)],
                conv_y[len(conv_blocks) : len(conv_blocks) + len(residual_blocks)],
                conv_z[len(conv_blocks) : len(conv_blocks) + len(residual_blocks)],
            )
        ]
        self.dense_blocks = [
            DenseBlock(
                dimension=dimension,
                block_size=dense_block_size,
                conv_type=conv_type,
                filters=filters,
                conv_x=filter_x,
                conv_y=filter_y,
                conv_z=filter_z,
                conv_padding=conv_padding,
                regularizer=regularizer,
                activation_layer=activation_layer,
                normalization_layer=normalization_layer,
                dropout_rate=dropout_rate,
                layer_order=dense_block_layer_order,
                pool_type=pool_type,
                pool_x=pool_x,
                pool_y=pool_y,
                pool_z=pool_z,
            )
            for filters, filter_x, filter_y, filter_z in zip(
                dense_blocks,
                conv_x[len(conv_blocks) + len(residual_blocks) :],
                conv_y[len(conv_blocks) + len(residual_blocks) :],
                conv_z[len(conv_blocks) + len(residual_blocks) :],
            )
        ]

    def __call__(self, x: Tensor) -> Tensor:
        for block in self.conv_blocks + self.res_blocks + self.dense_blocks:
            x = block(x)
        return x


class FullyConnectedEncoder:
    def __init__(
        self,
        *,
        width: int,
        activation_layer: str,
        normalization_layer: str,
        dropout_rate: float,
        layer_order: List[str],
        regularizer: Regularizer,
    ):
        self.dense = lambda x: x
        self.activation_layer = lambda x: x
        self.normalization_layer = lambda x: x
        self.dropout_layer = lambda x: x
        self.layer_order = layer_order
        if width > 0:
            self.dense = Dense(
                units=width,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
            )
            self.activation_layer = _activation_layer(activation_layer)
            self.normalization_layer = _normalization_layer(normalization_layer)
            self.dropout_layer = _dropout_layer(1, dropout_rate)

    def __call__(self, x: Tensor) -> Tensor:
        x = _order_layers(
            layer_order=self.layer_order,
            conv_or_dense_layer=self.dense,
            activation_layer=self.activation_layer,
            normalization_layer=self.normalization_layer,
            dropout_layer=self.dropout_layer,
        )(x)
        return x


# Decoders


class ConvDecoder:
    def __init__(
        self,
        *,
        tm: TensorMap,
        conv_type: str,
        conv_blocks: List[int],
        conv_block_size: int,
        conv_block_layer_order: List[str],
        conv_layer_type: str,
        conv_x: List[int],
        conv_y: List[int],
        conv_z: List[int],
        conv_padding: str,
        regularizer: Regularizer,
        activation_layer: str,
        normalization_layer: str,
        dropout_rate: float,
        upsample_x: int,
        upsample_y: int,
        upsample_z: int,
    ):
        dimension = tm.axes
        self.conv_blocks = [
            ConvolutionalBlock(
                dimension=dimension,
                block_size=conv_block_size,
                conv_type=conv_type,
                filters=filters,
                conv_x=filter_x,
                conv_y=filter_y,
                conv_z=filter_z,
                conv_padding=conv_padding,
                regularizer=regularizer,
                activation_layer=activation_layer,
                normalization_layer=normalization_layer,
                dropout_rate=dropout_rate,
                layer_order=conv_block_layer_order,
                pool_type="upsample",
                pool_x=upsample_x,
                pool_y=upsample_y,
                pool_z=upsample_z,
            )
            for filters, filter_x, filter_y, filter_z in zip(
                conv_blocks,
                conv_x[: len(conv_blocks)],
                conv_y[: len(conv_blocks)],
                conv_z[: len(conv_blocks)],
            )
        ]

        conv_layer, kernel, regularizer_args = _conv_layer(
            dimension=dimension,
            conv_type=conv_layer_type,
            conv_x=1,
            conv_y=1,
            conv_z=1,
            regularizer=regularizer,
        )
        self.output_conv = conv_layer(
            filters=tm.shape[-1],
            kernel_size=kernel,
            activation=tm.activation,
            name=tm.output_name,
            **regularizer_args,
        )

    def __call__(self, x: Tensor) -> Tensor:
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = self.output_conv(x)
        return x


class FullyConnectedDecoder:
    def __init__(
        self,
        tm: TensorMap,
    ):
        self.dense = Dense(
            units=tm.shape[0],
            name=tm.output_name,
            activation=tm.activation,
        )

    def __call__(
        self,
        x: Tensor,
    ) -> Tensor:
        return self.dense(x)


Encoder = Union[ConvEncoder, FullyConnectedEncoder]
Bottleneck = Union[ConcatenateRestructure, VariationalBottleNeck]
Decoder = Union[ConvDecoder, FullyConnectedDecoder]


#######################################
##                                   ##
## Helper Model Functions            ##
##                                   ##
#######################################


def _gini_loss(y_true: np.ndarray, y_est: np.ndarray):
    y_est = np.array([np.where(x == max(x))[0][0] for x in y_est])
    gini = 0
    for label_est in np.unique(y_est):
        indices = np.where(y_est == label_est)[0]
        p = 0
        for label_true in np.unique(y_true):
            p = len(np.where(y_true[indices] == label_true)[0]) / len(indices)
            gini += p * (1 - p) * len(indices)
    return gini


def _get_custom_objects(tensor_maps_out: List[TensorMap]) -> Dict[str, Any]:
    custom_objects = {
        obj.__name__: obj
        for obj in chain(
            NON_KERAS_OPTIMIZERS.values(),
            ACTIVATION_FUNCTIONS.values(),
            NORMALIZATION_CLASSES.values(),
            [VariationalDiagNormal],
        )
    }
    return {**custom_objects, **get_metric_dict(tensor_maps_out)}


def _global_average_pool(x: Tensor) -> Tensor:
    return K.mean(x, axis=tuple(range(1, len(x.shape) - 1)))


def _calc_start_shape(
    num_upsamples: int,
    output_shape: Tuple[int, ...],
    upsample_rates: Sequence[int],
    channels: int,
) -> Tuple[int, ...]:
    """
    Given the number of blocks in the decoder and the upsample rates, return
    required input shape to get to output shape
    """
    upsample_rates = list(upsample_rates) + [1] * len(output_shape)
    return tuple(
        (
            shape // rate ** num_upsamples
            for shape, rate in zip(output_shape[:-1], upsample_rates)
        )
    ) + (channels,)


def _repeat_dimension(dim: List[int], name: str, num_filters_needed: int) -> List[int]:
    if dim is None:
        dim = [-1]
    if len(dim) != num_filters_needed:
        logging.warning(
            f"Number of {name} dimensions for convolutional kernel sizes ({len(dim)})"
            " do not match number of convolutional layers/blocks"
            f" ({num_filters_needed}), matching values to fit {num_filters_needed}"
            " convolutional layers/blocks.",
        )
        repeat = num_filters_needed // len(dim) + 1
        dim = (dim * repeat)[:num_filters_needed]
    return dim


def _make_multimodal_multitask_model(
    encoders: Dict[TensorMap, Encoder],
    bottleneck: Bottleneck,
    decoders: Dict[TensorMap, Decoder],
    nested_models: List[Tuple[Model, str]],
    freeze: bool,
) -> Model:
    inputs: List[Input] = []
    encoder_outputs: Dict[TensorMap, Tensor] = {}
    for tm, encoder in encoders.items():
        x = Input(shape=tm.shape, name=tm.input_name)
        inputs.append(x)
        x = encoder(x)
        encoder_outputs[tm] = x

    for i, (model, hidden_layer) in enumerate(nested_models):
        for layer in model.layers:
            layer.trainable = not freeze
            if isinstance(layer, InputLayer):
                continue
            layer._name = f"{layer.name}_n{i}"
        model._name = str(i)
        inputs.extend(model.inputs)
        hidden_layer = f"{hidden_layer}_n{i}"
        encoder_outputs[i] = model.get_layer(hidden_layer).output

    bottleneck_outputs = bottleneck(encoder_outputs)

    decoder_outputs = {}
    for tm, decoder in decoders.items():
        decoder_outputs[tm] = decoder(bottleneck_outputs[tm])

    return Model(inputs=inputs, outputs=list(decoder_outputs.values()))


def _get_callbacks(
    model_file: str,
    patience: int,
    learning_rate_patience: int,
    learning_rate_reduction: float,
    monitor: str = "val_loss",
) -> List[Callback]:
    callbacks = [
        ModelCheckpoint(
            filepath=model_file,
            monitor=monitor,
            verbose=1,
            save_best_only=True,
        ),
        EarlyStopping(monitor=monitor, patience=patience, verbose=1),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=learning_rate_reduction,
            patience=learning_rate_patience,
            verbose=1,
        ),
    ]
    return callbacks


#######################################
##                                   ##
## Helper Layer Functions            ##
##                                   ##
#######################################


ACTIVATION_CLASSES = {
    "leaky": LeakyReLU(),
    "prelu": PReLU(),
    "elu": ELU(),
    "thresh_relu": ThresholdedReLU,
}
ACTIVATION_FUNCTIONS = {
    "swish": tf.nn.swish,
    "gelu": tfa.activations.gelu,
    "lisht": tfa.activations.lisht,
    "mish": tfa.activations.mish,
}
NORMALIZATION_CLASSES = {
    "batch_norm": BatchNormalization,
    "layer_norm": LayerNormalization,
    "instance_norm": tfa.layers.InstanceNormalization,
    "poincare_norm": tfa.layers.PoincareNormalize,
}


def _conv_layer(
    dimension: int,
    conv_type: str,
    conv_x: int,
    conv_y: int,
    conv_z: int,
    regularizer: Regularizer,
) -> Tuple[Layer, Tuple[int, ...], Dict[str, Regularizer]]:
    regularizer_keywords = ["bias_regularizer"]
    if dimension == 4 and conv_type == "conv":
        conv_layer = Conv3D
        kernel = (conv_x, conv_y, conv_z)
        regularizer_keywords.append("kernel_regularizer")
    elif dimension == 3 and conv_type == "conv":
        conv_layer = Conv2D
        kernel = (conv_x, conv_y)
        regularizer_keywords.append("kernel_regularizer")
    elif dimension == 2 and conv_type == "conv":
        conv_layer = Conv1D
        kernel = (conv_x,)
        regularizer_keywords.append("kernel_regularizer")
    elif dimension == 3 and conv_type == "separable":
        conv_layer = SeparableConv2D
        kernel = (conv_x, conv_y)
        regularizer_keywords.append("depthwise_regularizer")
        regularizer_keywords.append("pointwise_regularizer")
    elif dimension == 2 and conv_type == "separable":
        conv_layer = SeparableConv1D
        kernel = (conv_x,)
        regularizer_keywords.append("depthwise_regularizer")
        regularizer_keywords.append("pointwise_regularizer")
    elif dimension == 3 and conv_type == "depth":
        conv_layer = DepthwiseConv2D
        kernel = (conv_x, conv_y)
        regularizer_keywords.append("depthwise_regularizer")
    else:
        raise ValueError(f"{conv_type} is an unknown convolution layer type")

    regularizer_args = {kw: regularizer for kw in regularizer_keywords}
    return conv_layer, kernel, regularizer_args


def _activation_layer(activation: str) -> Activation:
    if not activation:
        return lambda x: x
    # fmt: off
    return (
        ACTIVATION_CLASSES.get(activation, None) or
        Activation(ACTIVATION_FUNCTIONS.get(activation, None) or activation)
    )
    # fmt: on


def _normalization_layer(normalization: str) -> Layer:
    if not normalization:
        return lambda x: x
    return NORMALIZATION_CLASSES[normalization]()


def _dropout_layer(dimension: int, dropout_rate: float) -> Layer:
    if dropout_rate > 0:
        if dimension == 4:
            return SpatialDropout3D(dropout_rate)
        elif dimension == 3:
            return SpatialDropout2D(dropout_rate)
        elif dimension == 2:
            return SpatialDropout1D(dropout_rate)
        else:
            return Dropout(dropout_rate)
    return lambda x: x


def _order_layers(
    layer_order: List[str],
    conv_or_dense_layer: Layer,
    activation_layer: Layer,
    normalization_layer: Layer,
    dropout_layer: Layer,
) -> Layer:
    def ordered_layers(x):
        for order in layer_order:
            if order == "dense" or order == "convolution":
                x = conv_or_dense_layer(x)
            elif order == "activation":
                x = activation_layer(x)
            elif order == "normalization":
                x = normalization_layer(x)
            elif order == "dropout":
                x = dropout_layer(x)
            else:
                pass
        return x

    return ordered_layers


def _pool_layer(
    dimension: int,
    pool_type: str,
    pool_x: int,
    pool_y: int,
    pool_z: int,
) -> Layer:
    if dimension == 4 and pool_type == "max":
        return MaxPooling3D(pool_size=(pool_x, pool_y, pool_z))
    elif dimension == 3 and pool_type == "max":
        return MaxPooling2D(pool_size=(pool_x, pool_y))
    elif dimension == 2 and pool_type == "max":
        return MaxPooling1D(pool_size=pool_x)
    elif dimension == 4 and pool_type == "average":
        return AveragePooling3D(pool_size=(pool_x, pool_y, pool_z))
    elif dimension == 3 and pool_type == "average":
        return AveragePooling2D(pool_size=(pool_x, pool_y))
    elif dimension == 2 and pool_type == "average":
        return AveragePooling1D(pool_size=pool_x)
    elif dimension == 4 and pool_type == "upsample":
        return UpSampling3D(size=(pool_x, pool_y, pool_z))
    elif dimension == 3 and pool_type == "upsample":
        return UpSampling2D(size=(pool_x, pool_y))
    elif dimension == 2 and pool_type == "upsample":
        return UpSampling1D(size=pool_x)
    return lambda x: x
