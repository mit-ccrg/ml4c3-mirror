# pylint: disable=attribute-defined-outside-init
# Imports: standard library
import os
import logging
from enum import Enum, auto
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
from tensorflow.keras.utils import model_to_dot
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
    RepeatVector,
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

# Imports: first party
from ml4c3.plots import plot_metric_history, plot_architecture_diagram
from ml4c3.metrics import get_metric_dict
from ml4c3.datasets import (
    BATCH_INPUT_INDEX,
    BATCH_OUTPUT_INDEX,
    get_array_from_dict_of_arrays,
    get_dicts_of_arrays_from_dataset,
)
from ml4c3.optimizers import NON_KERAS_OPTIMIZERS, get_optimizer
from definitions.globals import MODEL_EXT
from ml4c3.tensormap.TensorMap import TensorMap

CHANNEL_AXIS = -1  # Set to 1 for Theano backend
LANGUAGE_MODEL_SUFFIX = "_next_character"

SKLEARN_MODELS = Union[
    LogisticRegression,
    LinearSVC,
    RandomForestClassifier,
    XGBClassifier,
]

tfd = tfp.distributions


class BottleneckType(Enum):
    # All decoder outputs are flattened to put into embedding
    FlattenRestructure = auto()

    # Structured (not flat) decoder outputs are global average pooled
    GlobalAveragePoolStructured = auto()

    # All decoder outputs are flattened then variationally sampled to put into embedding
    Variational = auto()


def make_model(args):
    """
    Create a model to train according the input arguments.
    """
    if args.recipe in ["train", "train_simclr", "infer", "build"]:
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
        model.loss_function = gini_loss
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


def make_waveform_model_unet(
    tensor_maps_in: List[TensorMap],
    tensor_maps_out: List[TensorMap],
    learning_rate: float,
    model_file: str = None,
    donor_layers: str = None,
) -> Model:
    """Make a waveform predicting model

    Input and output tensor maps are set from the command line.
    Model summary printed to output

    :param tensor_maps_in: List of input TensorMaps, only 1 input TensorMap is
                           currently supported, otherwise there are layer name
                           collisions.
    :param tensor_maps_out: List of output TensorMaps
    :param learning_rate: Size of learning steps in SGD optimization
    :param model_file: Optional HD5 model file to load and return.
    :param donor_layers: Optional HD5 model file whose weights will be loaded into
                         this model when layer names match.
    :return: a compiled keras model
    """
    if model_file is not None:
        m = load_model(model_file, custom_objects=get_metric_dict(tensor_maps_out))
        m.summary()
        logging.info(f"Loaded model file from: {model_file}")
        return m

    neurons = 24
    input_tensor = residual = Input(
        shape=tensor_maps_in[0].shape,
        name=tensor_maps_in[0].input_name,
    )
    x = c600 = Conv1D(
        filters=neurons,
        kernel_size=11,
        activation="relu",
        padding="same",
    )(input_tensor)
    x = Conv1D(filters=neurons, kernel_size=51, activation="relu", padding="same")(x)
    x = MaxPooling1D(2)(x)
    x = c300 = Conv1D(
        filters=neurons,
        kernel_size=111,
        activation="relu",
        padding="same",
    )(x)
    x = Conv1D(filters=neurons, kernel_size=201, activation="relu", padding="same")(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(filters=neurons, kernel_size=301, activation="relu", padding="same")(x)
    x = Conv1D(filters=neurons, kernel_size=301, activation="relu", padding="same")(x)
    x = UpSampling1D(2)(x)
    x = concatenate([x, c300])
    x = Conv1D(filters=neurons, kernel_size=201, activation="relu", padding="same")(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(filters=neurons, kernel_size=111, activation="relu", padding="same")(x)
    x = concatenate([x, c600])
    x = Conv1D(filters=neurons, kernel_size=51, activation="relu", padding="same")(x)
    x = concatenate([x, residual])
    conv_label = Conv1D(
        filters=tensor_maps_out[0].shape[CHANNEL_AXIS],
        kernel_size=1,
        activation="linear",
    )(x)
    output_y = Activation(
        tensor_maps_out[0].activation,
        name=tensor_maps_out[0].output_name,
    )(conv_label)
    m = Model(inputs=[input_tensor], outputs=[output_y])
    m.summary()
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0)
    m.compile(
        optimizer=opt,
        loss=tensor_maps_out[0].loss,
        metrics=tensor_maps_out[0].metrics,
    )

    if donor_layers is not None:
        m.load_weights(donor_layers, by_name=True)
        logging.info(f"Loaded model weights from:{donor_layers}")

    return m


def _order_layers(
    layer_order: List[str],
    activate: Layer = None,
    normalize: Layer = None,
    dropout: Layer = None,
) -> Layer:
    identity = lambda x: x
    activate = activate or identity
    normalize = normalize or identity
    dropout = dropout or identity

    def ordered_layers(x):
        for order in layer_order:
            if order == "activation":
                x = activate(x)
            elif order == "normalization":
                x = normalize(x)
            elif order == "dropout":
                x = dropout(x)
            else:
                pass
        return x

    return ordered_layers


Tensor = tf.Tensor
Encoder = Callable[[Tensor], Tuple[Tensor, List[Tensor]]]
Decoder = Callable[
    [Tensor, Dict[TensorMap, List[Tensor]], Dict[TensorMap, Tensor]],
    Tensor,
]
BottleNeck = Callable[[Dict[TensorMap, Tensor]], Dict[TensorMap, Tensor]]


class ResidualBlock:
    def __init__(
        self,
        *,
        dimension: int,
        filters_per_conv: List[int],
        conv_layer_type: str,
        conv_x: List[int],
        conv_y: List[int],
        conv_z: List[int],
        activation: str,
        normalization: str,
        dropout_type: str,
        dropout_rate: float,
        layer_order: List[str],
        dilate: bool,
        regularizer: tf.keras.regularizers = None,
    ):
        block_size = len(filters_per_conv)
        assert len(conv_x) == len(conv_y) == len(conv_z) == block_size
        conv_layer, kernels = _build_conv_layer(
            dimension=dimension,
            conv_layer_type=conv_layer_type,
            conv_x=conv_x,
            conv_y=conv_y,
            conv_z=conv_z,
        )
        self.conv_layers = []

        # Regularization differs by convolutional layer class
        for i, (filters, kernel) in enumerate(zip(filters_per_conv, kernels)):
            if conv_layer == Conv3D or conv_layer == Conv2D or conv_layer == Conv1D:
                c = conv_layer(
                    filters=filters,
                    kernel_size=kernel,
                    padding="same",
                    dilation_rate=2 ** i if dilate else 1,
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer,
                )
            # Separable 1 and 2D conv layers have depth and pointwise kernel matrices
            elif conv_layer == SeparableConv2D or conv_layer == SeparableConv1D:
                c = conv_layer(
                    filters=filters,
                    kernel_size=kernel,
                    padding="same",
                    dilation_rate=2 ** i if dilate else 1,
                    depthwise_regularizer=regularizer,
                    pointwise_regularizer=regularizer,
                    bias_regularizer=regularizer,
                )
            # Depthwise 2D conv layer lacks pointwise kernel matrix to regularize
            elif conv_layer == DepthwiseConv2D:
                c = conv_layer(
                    kernel_size=kernel,
                    padding="same",
                    dilation_rate=2 ** i if dilate else 1,
                    depthwise_regularizer=regularizer,
                    bias_regularizer=regularizer,
                )
            else:
                raise ValueError(
                    f"{conv_layer} is not a valid convolutional layer class",
                )
            self.conv_layers.append(c)

        self.activations = [_activation_layer(activation) for _ in range(block_size)]
        self.normalizations = [
            _normalization_layer(normalization) for _ in range(block_size)
        ]
        self.dropout_layers = [
            _dropout_layer(dimension, dropout_type, dropout_rate)
            for _ in range(block_size)
        ]
        residual_conv_layer, _ = _build_conv_layer(
            dimension=dimension,
            conv_layer_type=conv_layer_type,
            conv_x=conv_x,
            conv_y=conv_y,
            conv_z=conv_z,
        )
        self.residual_convs = [
            residual_conv_layer(
                filters=filters_per_conv[0],
                kernel_size=_one_by_n_kernel(dimension),
            )
            for _ in range(block_size - 1)
        ]
        self.layer_order = layer_order
        logging.info(
            "Residual Block Convolutional Layers (num_filters, kernel_size):"
            f" {list(zip(filters_per_conv, kernels))}",
        )

    def __call__(self, x: Tensor) -> Tensor:
        previous = x
        for convolve, activate, normalize, dropout, one_by_n_convolve in zip(
            self.conv_layers,
            self.activations,
            self.normalizations,
            self.dropout_layers,
            [None] + self.residual_convs,
        ):
            x = _order_layers(self.layer_order, activate, normalize, dropout)(
                convolve(x),
            )
            if one_by_n_convolve is not None:  # Do not residual add the input
                x = Add()([one_by_n_convolve(x), previous])
            previous = x
        return x


class DenseConvolutionalBlock:
    def __init__(
        self,
        *,
        dimension: int,
        block_size: int,
        conv_layer_type: str,
        filters: int,
        conv_x: List[int],
        conv_y: List[int],
        conv_z: List[int],
        activation: str,
        normalization: str,
        dropout_type: str,
        dropout_rate: float,
        layer_order: List[str],
        regularizer: tf.keras.regularizers = None,
    ):
        conv_layer, kernels = _build_conv_layer(
            dimension=dimension,
            conv_layer_type=conv_layer_type,
            conv_x=conv_x,
            conv_y=conv_y,
            conv_z=conv_z,
        )

        self.conv_layers = []
        for kernel in kernels:
            if conv_layer == Conv3D or conv_layer == Conv2D or conv_layer == Conv1D:
                c = conv_layer(
                    filters=filters,
                    kernel_size=kernel,
                    padding="same",
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer,
                )
            # Separable 1 and 2D conv layers have depth and pointwise kernel matrices
            elif conv_layer == SeparableConv2D or conv_layer == SeparableConv1D:
                c = conv_layer(
                    filters=filters,
                    kernel_size=kernel,
                    padding="same",
                    depthwise_regularizer=regularizer,
                    pointwise_regularizer=regularizer,
                    bias_regularizer=regularizer,
                )
            # Depthwise 2D conv layer lacks pointwise kernel matrix to regularize
            elif conv_layer == DepthwiseConv2D:
                c = conv_layer(
                    kernel_size=kernel,
                    padding="same",
                    depthwise_regularizer=regularizer,
                    bias_regularizer=regularizer,
                )
            else:
                raise ValueError(
                    f"{conv_layer} is not a valid convolutional layer type",
                )
            self.conv_layers.append(c)

        self.activations = [_activation_layer(activation) for _ in range(block_size)]
        self.normalizations = [
            _normalization_layer(normalization) for _ in range(block_size)
        ]
        self.dropout_layers = [
            _dropout_layer(dimension, dropout_type, dropout_rate)
            for _ in range(block_size)
        ]
        self.layer_order = layer_order
        logging.info(
            "Dense Block Convolutional Layers (num_filters, kernel_size):"
            f" {list(zip([filters]*len(kernels), kernels))}",
        )

    def __call__(self, x: Tensor) -> Tensor:
        dense_connections = [x]
        for i, (convolve, activate, normalize, dropout_layer) in enumerate(
            zip(
                self.conv_layers,
                self.activations,
                self.normalizations,
                self.dropout_layers,
            ),
        ):
            x = _order_layers(self.layer_order, activate, normalize, dropout_layer)(
                convolve(x),
            )
            # Output of block does not get concatenated to
            if i < len(self.conv_layers) - 1:
                dense_connections.append(x)
                x = Concatenate()(dense_connections[:])
        return x


class FullyConnectedBlock:
    """
    Creates a fully connected block with dense, activation, regularization, and
    normalization layers
    """

    def __init__(
        self,
        *,
        widths: List[int],
        activation: str,
        normalization: str,
        dropout_type: str,
        dropout_rate: float,
        layer_order: List[str],
        name: str = None,
        is_encoder: bool = False,
        regularizer: tf.keras.regularizers = None,
    ):
        """
        :param widths: number of neurons in each dense layer
        :param activation: string name of activation function
        :param normalization: optional string name of normalization function
        :param dropout_type: optional string name of dropout function
        :param dropout_rate: if dropout is applied, this is the rate
        :param layer_order: list of strings specifying the activation, normalization,
                            and dropout layers after the dense layer
        :param is_encoder: boolean indicator if fully connected block is an input
                           block
        :param name: name of last dense layer in fully connected block, otherwise
                     all dense layers are named ordinally e.g. dense_3 for the
                     third dense layer in the model
        """
        final_dense = (
            Dense(
                units=widths[-1],
                name=name,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
            )
            if name
            else Dense(
                units=widths[-1],
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
            )
        )
        self.denses = [
            Dense(
                units=width,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
            )
            for width in widths[:-1]
        ] + [final_dense]
        self.activations = [_activation_layer(activation) for _ in widths]
        self.dropout_layers = [
            _dropout_layer(1, dropout_type, dropout_rate) for _ in widths
        ]
        self.norms = [_normalization_layer(normalization) for _ in widths]
        self.is_encoder = is_encoder
        self.layer_order = layer_order

    def __call__(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        for dense, normalize, activate, dropout in zip(
            self.denses,
            self.norms,
            self.activations,
            self.dropout_layers,
        ):
            x = _order_layers(
                layer_order=self.layer_order,
                activate=activate,
                normalize=normalize,
                dropout=dropout,
            )(dense(x))
        if self.is_encoder:
            return x, []
        return x


def adaptive_normalize_from_tensor(tensor: Tensor, target: Tensor) -> Tensor:
    """
    Uses Dense layers to convert `tensor` to a mean and standard deviation to
    normalize `target`
    """
    return adaptive_normalization(
        mu=Dense(target.shape[-1])(tensor),
        sigma=Dense(target.shape[-1])(tensor),
        target=target,
    )


def adaptive_normalization(mu: Tensor, sigma: Tensor, target: Tensor) -> Tensor:
    target = tfa.layers.InstanceNormalization()(target)
    normalizer_shape = (1,) * (len(target.shape) - 2) + (target.shape[-1],)
    mu = Reshape(normalizer_shape)(mu)
    sigma = Reshape(normalizer_shape)(sigma)
    target *= sigma
    target += mu
    return target


def global_average_pool(x: Tensor) -> Tensor:
    return K.mean(x, axis=tuple(range(1, len(x.shape) - 1)))


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


class VariationalBottleNeck:
    def __init__(
        self,
        activation: str,
        normalization: str,
        fully_connected_widths: List[int],
        latent_size: int,
        dropout_type: str,
        dropout_rate: float,
        layer_order: List[str],
        pre_decoder_shapes: Dict[TensorMap, Optional[Tuple[int, ...]]],
        regularizer: tf.keras.regularizers = None,
    ):
        self.fully_connected = (
            FullyConnectedBlock(
                widths=fully_connected_widths,
                activation=activation,
                normalization=normalization,
                dropout_type=dropout_type,
                dropout_rate=dropout_rate,
                regularizer=regularizer,
                layer_order=layer_order,
            )
            if fully_connected_widths
            else None
        )
        self.restructures = {
            tm: FlatToStructure(
                output_shape=shape,
                regularizer=regularizer,
                activation=activation,
                normalization=normalization,
                layer_order=layer_order,
            )
            for tm, shape in pre_decoder_shapes.items()
            if shape is not None
        }
        self.latent_size = latent_size
        self.sampler: Callable = VariationalDiagNormal(latent_size)
        self.no_restructures = [
            tm for tm, shape in pre_decoder_shapes.items() if shape is None
        ]
        self.regularizer = regularizer

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


class ConcatenateRestructure:
    """
    Flattens or GAPs then concatenates all inputs, applies a dense layer, then
    restructures to provided shapes
    """

    def __init__(
        self,
        activation: str,
        bottleneck_type: BottleneckType,
        dropout_rate: float,
        dropout_type: str,
        layer_order: List[str],
        normalization: str,
        pre_decoder_shapes: Dict[TensorMap, Optional[Tuple[int, ...]]],
        widths: List[int],
        regularizer: tf.keras.regularizers = None,
    ):
        self.fully_connected = (
            FullyConnectedBlock(
                widths=widths,
                activation=activation,
                normalization=normalization,
                dropout_type=dropout_type,
                dropout_rate=dropout_rate,
                regularizer=regularizer,
                layer_order=layer_order,
            )
            if widths
            else None
        )
        self.restructures = {
            tm: FlatToStructure(
                output_shape=shape,
                activation=activation,
                normalization=normalization,
                layer_order=layer_order,
            )
            for tm, shape in pre_decoder_shapes.items()
            if shape is not None
        }
        self.no_restructures = [
            tm for tm, shape in pre_decoder_shapes.items() if shape is None
        ]
        self.bottleneck_type = bottleneck_type

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
                global_average_pool(x)
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


class FlatToStructure:
    """Takes a flat input, applies a dense layer, then restructures to output_shape"""

    def __init__(
        self,
        output_shape: Tuple[int, ...],
        activation: str,
        normalization: str,
        layer_order: List[str],
        regularizer: tf.keras.regularizers = None,
    ):
        self.input_shapes = output_shape
        self.dense = Dense(
            units=int(np.prod(output_shape)),
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
        )
        self.activation = _activation_layer(activation)
        self.reshape = Reshape(output_shape)
        self.norm = _normalization_layer(normalization)
        self.layer_order = layer_order

    def __call__(self, x: Tensor) -> Tensor:
        return self.reshape(
            _order_layers(self.layer_order, self.activation, self.norm)(self.dense(x)),
        )


class ConvEncoder:
    def __init__(
        self,
        *,
        activation: str,
        block_size: int,
        conv_layer_type: str,
        conv_x: List[int],
        conv_y: List[int],
        conv_z: List[int],
        dilate: bool,
        dimension: int,
        dropout_rate: float,
        dropout_type: str,
        filters_per_dense_block: List[int],
        layer_order: List[str],
        normalization: str,
        pool_after_final_dense_block: bool,
        pool_type: str,
        pool_x: int,
        pool_y: int,
        pool_z: int,
        res_filters: List[int],
        regularizer: tf.keras.regularizers = None,
    ):

        num_res = len(res_filters)
        res_x, res_y, res_z = conv_x[:num_res], conv_y[:num_res], conv_z[:num_res]
        self.res_block = ResidualBlock(
            dimension=dimension,
            filters_per_conv=res_filters,
            conv_layer_type=conv_layer_type,
            conv_x=res_x,
            conv_y=res_y,
            conv_z=res_z,
            activation=activation,
            normalization=normalization,
            dropout_type=dropout_type,
            dropout_rate=dropout_rate,
            regularizer=regularizer,
            layer_order=layer_order,
            dilate=dilate,
        )

        dense_x, dense_y, dense_z = conv_x[num_res:], conv_y[num_res:], conv_z[num_res:]
        self.dense_blocks = [
            DenseConvolutionalBlock(
                dimension=dimension,
                conv_layer_type=conv_layer_type,
                filters=filters,
                conv_x=[x] * block_size,
                conv_y=[y] * block_size,
                conv_z=[z] * block_size,
                block_size=block_size,
                activation=activation,
                normalization=normalization,
                dropout_type=dropout_type,
                dropout_rate=dropout_rate,
                regularizer=regularizer,
                layer_order=layer_order,
            )
            for filters, x, y, z in zip(
                filters_per_dense_block,
                dense_x,
                dense_y,
                dense_z,
            )
        ]

        self.pool_after_final_dense_block = pool_after_final_dense_block
        self.pool_before_first_dense_block = num_res > 0

        self.pools = _pool_layers_from_kind_and_dimension(
            dimension,
            pool_type,
            len(filters_per_dense_block) + 1,
            pool_x,
            pool_y,
            pool_z,
        )

    def __call__(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        intermediates = []
        x = self.res_block(x)
        intermediates.append(x)
        if self.pool_before_first_dense_block:
            x = self.pools[0](x)

        for i, (dense_block, pool) in enumerate(zip(self.dense_blocks, self.pools[1:])):
            x = dense_block(x)
            intermediates.append(x)

            # Add pooling layer for every dense block
            if self.pool_after_final_dense_block:
                x = pool(x)

            # Do not pool after final dense block
            else:
                x = pool(x) if i < len(self.dense_blocks) - 1 else x

        return x, intermediates


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


class DenseDecoder:
    def __init__(
        self,
        tensor_map_out: TensorMap,
    ):
        self.dense = Dense(
            units=tensor_map_out.shape[0],
            name=tensor_map_out.output_name,
            activation=tensor_map_out.activation,
        )

    def __call__(
        self,
        x: Tensor,
        _,
        decoder_outputs: Dict[TensorMap, Tensor],
    ) -> Tensor:
        return self.dense(x)


class ConvDecoder:
    def __init__(
        self,
        *,
        activation: str,
        block_size: int,
        conv_layer_type: str,
        conv_x: List[int],
        conv_y: List[int],
        conv_z: List[int],
        dropout_rate: float,
        dropout_type: str,
        filters_per_dense_block: List[int],
        layer_order: List[str],
        normalization: str,
        tensor_map_out: TensorMap,
        upsample_x: int,
        upsample_y: int,
        upsample_z: int,
        regularizer: tf.keras.regularizers = None,
    ):
        dimension = tensor_map_out.axes
        self.dense_blocks = [
            DenseConvolutionalBlock(
                dimension=tensor_map_out.axes,
                conv_layer_type=conv_layer_type,
                filters=filters,
                conv_x=[x] * block_size,
                conv_y=[y] * block_size,
                conv_z=[z] * block_size,
                block_size=block_size,
                activation=activation,
                normalization=normalization,
                dropout_type=dropout_type,
                dropout_rate=dropout_rate,
                regularizer=regularizer,
                layer_order=layer_order,
            )
            for filters, x, y, z in zip(filters_per_dense_block, conv_x, conv_y, conv_z)
        ]
        conv_layer, _ = _build_conv_layer(
            dimension=dimension,
            conv_layer_type=conv_layer_type,
            conv_x=conv_x,
            conv_y=conv_y,
            conv_z=conv_z,
        )

        if conv_layer == Conv3D or conv_layer == Conv2D or conv_layer == Conv1D:
            self.conv_label = conv_layer(
                filters=tensor_map_out.shape[-1],
                kernel_size=_one_by_n_kernel(dimension),
                activation=tensor_map_out.activation,
                name=tensor_map_out.output_name,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
            )
        # Separable 1 and 2D conv layers have depth and pointwise kernel matrices
        elif conv_layer == SeparableConv2D or conv_layer == SeparableConv1D:
            self.conv_label = conv_layer(
                filters=tensor_map_out.shape[-1],
                kernel_size=_one_by_n_kernel(dimension),
                activation=tensor_map_out.activation,
                name=tensor_map_out.output_name,
                depthwise_regularizer=regularizer,
                pointwise_regularizer=regularizer,
                bias_regularizer=regularizer,
            )
        # Depthwise 2D conv layer lacks pointwise kernel matrix to regularize
        elif conv_layer == DepthwiseConv2D:
            self.conv_label = conv_layer(
                kernel_size=_one_by_n_kernel(dimension),
                activation=tensor_map_out.activation,
                name=tensor_map_out.output_name,
                depthwise_regularizer=regularizer,
                bias_regularizer=regularizer,
            )
        else:
            raise ValueError(f"{conv_layer} is not a valid convolutional layer type")

        self.upsamples = [
            _upsampler(dimension, upsample_x, upsample_y, upsample_z)
            for _ in range(len(filters_per_dense_block) + 1)
        ]

    def __call__(
        self,
        x: Tensor,
        intermediates: Dict[TensorMap, List[Tensor]],
        _,
    ) -> Tensor:
        for _, (dense_block, upsample) in enumerate(
            zip(self.dense_blocks, self.upsamples),
        ):
            x = dense_block(x)
            x = upsample(x)
        return self.conv_label(x)


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


def _repeat_dimension(dim: List[int], name: str, num_filters_needed: int) -> List[int]:
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


def make_multimodal_multitask_model(
    activation: str,
    bottleneck_type: BottleneckType,
    learning_rate: float,
    optimizer: str,
    tensor_maps_in: List[TensorMap],
    tensor_maps_out: List[TensorMap],
    block_size: int = None,
    conv_dilate: bool = None,
    conv_layers: List[int] = None,
    conv_type: str = None,
    conv_x: List[int] = None,
    conv_y: List[int] = None,
    conv_z: List[int] = None,
    dense_blocks: List[int] = None,
    dense_dropout: float = 0,
    dense_layers: List[int] = None,
    directly_embed_and_repeat: int = None,
    donor_layers: str = None,
    freeze_donor_layers: bool = None,
    l1: float = 0,
    l2: float = 0,
    layer_normalization: str = None,
    layer_order: List[str] = None,
    learning_rate_schedule: str = None,
    model_file: str = None,
    nest_model: List[List[str]] = None,
    pool_after_final_dense_block: bool = None,
    pool_type: str = None,
    pool_x: int = None,
    pool_y: int = None,
    pool_z: int = None,
    remap_layer: Dict[str, str] = None,
    spatial_dropout: float = 0,
    **kwargs,
) -> Model:
    """
    Make multi-task, multi-modal feed forward neural network for all kinds of prediction

    This model factory can be used to make networks for classification, regression,
    and segmentation
    The tasks attempted are given by the output TensorMaps.
    The modalities and the first layers in the architecture are determined by the
    input TensorMaps.

    Hyperparameters are exposed to the command line.
    Model summary is printed to output.

    :param activation: Activation function as a string ('relu', 'linear', 'softmax')
    :param block_size: Number of layers within each Densenet module for densenet convolutional models
    :param bottleneck_type: How to merge the representations coming from the different input modalities
    :param conv_dilate: whether to use dilation in conv layers
    :param conv_layers: List of number of filters in each convolutional layer
    :param conv_regularize: Type of regularization for convolutions, e.g. dropout
    :param conv_type: Type of convolution to use, e.g. separable
    :param conv_x: Size of X dimension for 2D and 3D convolutional kernels
    :param conv_y: Size of Y dimension for 2D and 3D convolutional kernels
    :param conv_z: Size of Z dimension for 3D convolutional kernels
    :param dense_blocks: List of number of filters in densenet modules for densenet convolutional models
    :param dense_layer_dropout: Dropout rate of dense layers; must be in [0, 1].
    :param dense_layers: List of number of filters in each dense layer.
    :param directly_embed_and_repeat: If set, directly embed input tensors (without passing to a dense layer) into concatenation layer, and repeat each input N times, where N is this argument's value. To directly embed a feature without repetition, set to 1.
    :param donor_layers: HD5 model file whose weights will be loaded into this model when layer names match.
    :param dropout: Dropout rate in dense layers
    :param freeze_donor_layers: Whether to freeze layers from loaded from donor_layers
    :param layer_normalization: Type of normalization layer for fully connected and convolutional layers, e.g. batch norm
    :param layer_normalization: Type of normalization layer for fully connected and convolutional layers, e.g. batch norm
    :param layer_order: Order of activation, normalization, and regularization layers
    :param learning_rate: learning rate for optimizer
    :param learning_rate_schedule: learning rate schedule to train with, e.g. triangular
    :param l1: Optional float value to use for L1 regularization.
    :param l2: Optional float value to use for L2 regularization.
    :param model_file: HD5 model file to load and return.
    :param nest_model: Embed a nested model ending at the specified layer before the bottleneck layer of the current model. List of models to embed and layer name of embedded layer.
    :param optimizer: which optimizer to use. See optimizers.py.
    :param pool_after_final_dense_block: Add pooling layer after final dense block
    :param pool_type: Max or average pooling following convolutional blocks
    :param pool_x: Pooling in the X dimension for Convolutional models.
    :param pool_y: Pooling in the Y dimension for Convolutional models.
    :param pool_z: Pooling in the Z dimension for 3D Convolutional models.
    :param remap_layer: Dictionary remapping layers from donor_layers to layers in current model
    :param spatial_dropout: Rate of spatial dropout in convolutional layers.
    :param tensor_maps_in: List of input TensorMaps
    :param tensor_maps_out: List of output TensorMaps
    :return: a compiled keras model
    """
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

    dense_normalize = layer_normalization

    # list of filter dimensions should match the number of convolutional
    # layers = len(dense_blocks) + [ + len(conv_layers) if convolving input tensors]
    num_dense = len(dense_blocks)
    num_res = len(conv_layers) if any(tm.axes > 1 for tm in tensor_maps_in) else 0
    num_filters_needed = num_res + num_dense
    conv_x = _repeat_dimension(conv_x, "x", num_filters_needed)
    conv_y = _repeat_dimension(conv_y, "y", num_filters_needed)
    conv_z = _repeat_dimension(conv_z, "z", num_filters_needed)

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

    encoders: Dict[TensorMap] = {}
    for tm in tensor_maps_in:
        if any(tm.input_name in input_layer for input_layer in nested_model_inputs):
            continue
        if tm.axes > 1:
            encoders[tm] = ConvEncoder(
                filters_per_dense_block=dense_blocks,
                dimension=tm.axes,
                res_filters=conv_layers,
                conv_layer_type=conv_type,
                conv_x=conv_x,
                conv_y=conv_y,
                conv_z=conv_z,
                block_size=block_size,
                activation=activation,
                normalization=layer_normalization,
                dropout_type="spatial_dropout" if spatial_dropout > 0 else None,
                dropout_rate=spatial_dropout,
                regularizer=regularizer,
                layer_order=layer_order,
                dilate=conv_dilate,
                pool_after_final_dense_block=pool_after_final_dense_block,
                pool_type=pool_type,
                pool_x=pool_x,
                pool_y=pool_y,
                pool_z=pool_z,
            )
        else:
            if directly_embed_and_repeat is not None:
                encoders[tm] = lambda x: (
                    RepeatVector(directly_embed_and_repeat)(x),
                    [],
                )
            else:
                encoders[tm] = FullyConnectedBlock(
                    widths=[tm.annotation_units],
                    activation=activation,
                    normalization=dense_normalize,
                    dropout_type="dropout" if dense_dropout > 0 else None,
                    dropout_rate=dense_dropout,
                    regularizer=regularizer,
                    layer_order=layer_order,
                    is_encoder=True,
                )

    pre_decoder_shapes: Dict[TensorMap, Optional[Tuple[int, ...]]] = {}
    for tm in tensor_maps_out:
        if tm.axes == 1:
            pre_decoder_shapes[tm] = None
        else:
            pre_decoder_shapes[tm] = _calc_start_shape(
                num_upsamples=len(dense_blocks),
                output_shape=tm.shape,
                upsample_rates=[pool_x, pool_y, pool_z],
                channels=dense_blocks[-1],
            )

    if bottleneck_type in {
        BottleneckType.FlattenRestructure,
        BottleneckType.GlobalAveragePoolStructured,
    }:
        bottleneck = ConcatenateRestructure(
            widths=dense_layers,
            activation=activation,
            dropout_type="dropout" if dense_dropout > 0 else None,
            dropout_rate=dense_dropout,
            regularizer=regularizer,
            normalization=dense_normalize,
            layer_order=layer_order,
            pre_decoder_shapes=pre_decoder_shapes,
            bottleneck_type=bottleneck_type,
        )
    elif bottleneck_type == BottleneckType.Variational:
        bottleneck = VariationalBottleNeck(
            fully_connected_widths=dense_layers[:-1],
            latent_size=dense_layers[-1],
            activation=activation,
            dropout_type="dropout" if dense_dropout > 0 else None,
            dropout_rate=dense_dropout,
            regularizer=regularizer,
            normalization=dense_normalize,
            layer_order=layer_order,
            pre_decoder_shapes=pre_decoder_shapes,
        )
    else:
        raise NotImplementedError(f"Unknown BottleneckType {bottleneck_type}.")

    conv_x, conv_y, conv_z = conv_x[num_res:], conv_y[num_res:], conv_z[num_res:]
    decoders: Dict[TensorMap, Layer] = {}
    for tm in tensor_maps_out:
        if tm.axes > 1:
            decoders[tm] = ConvDecoder(
                tensor_map_out=tm,
                filters_per_dense_block=dense_blocks,
                conv_layer_type=conv_type,
                conv_x=conv_x,
                conv_y=conv_y,
                conv_z=conv_z,
                block_size=1,
                activation=activation,
                normalization=layer_normalization,
                dropout_type="spatial_dropout" if spatial_dropout > 0 else None,
                dropout_rate=spatial_dropout,
                regularizer=regularizer,
                layer_order=layer_order,
                upsample_x=pool_x,
                upsample_y=pool_y,
                upsample_z=pool_z,
            )
        else:
            decoders[tm] = DenseDecoder(tensor_map_out=tm)

    m = _make_multimodal_multitask_model(
        encoders=encoders,
        nested_models=nested_models,
        bottleneck=bottleneck,
        decoders=decoders,
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


def _make_multimodal_multitask_model(
    encoders: Dict[TensorMap, Encoder],
    nested_models: List[Tuple[Model, str]],
    bottleneck: BottleNeck,
    decoders: Dict[TensorMap, Decoder],
    freeze: bool,
) -> Model:
    inputs: List[Input] = []
    encoder_outputs: Dict[
        TensorMap,
        Tuple[Tensor, List[Tensor]],
    ] = {}  # TensorMap -> embed, encoder_intermediates
    encoder_intermediates = {}
    for tm, encoder in encoders.items():
        x = Input(shape=tm.shape, name=tm.input_name)
        inputs.append(x)
        y, intermediates = encoder(x)
        encoder_outputs[tm] = y
        encoder_intermediates[tm] = intermediates

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

    bottle_neck_outputs = bottleneck(encoder_outputs)

    decoder_outputs = {}
    for tm, decoder in decoders.items():
        decoder_outputs[tm] = decoder(
            bottle_neck_outputs[tm],
            encoder_intermediates,
            decoder_outputs,
        )

    return Model(inputs=inputs, outputs=list(decoder_outputs.values()))


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
        if plot:
            image_path = os.path.join(output_folder, "architecture_graph" + image_ext)
            plot_architecture_diagram(
                dot=model_to_dot(model, show_shapes=True, expand_nested=True),
                image_path=image_path,
            )
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


def _one_by_n_kernel(dimension):
    return tuple([1] * (dimension - 1))


def _build_conv_layer(
    dimension: int,
    conv_layer_type: str,
    conv_x: List[int],
    conv_y: List[int],
    conv_z: List[int],
) -> Tuple[Layer, List[Tuple[int, ...]]]:
    if dimension == 4 and conv_layer_type == "conv":
        conv_layer = Conv3D
        kernel = zip(conv_x, conv_y, conv_z)
    elif dimension == 3 and conv_layer_type == "conv":
        conv_layer = Conv2D
        kernel = zip(conv_x, conv_y)
    elif dimension == 2 and conv_layer_type == "conv":
        conv_layer = Conv1D
        kernel = zip(conv_x)
    elif dimension == 3 and conv_layer_type == "separable":
        conv_layer = SeparableConv2D
        kernel = zip(conv_x, conv_y)
    elif dimension == 2 and conv_layer_type == "separable":
        conv_layer = SeparableConv1D
        kernel = zip(conv_x)
    elif dimension == 3 and conv_layer_type == "depth":
        conv_layer = DepthwiseConv2D
        kernel = zip(conv_x, conv_y)
    else:
        raise ValueError(f"{conv_layer_type} is an unknown convolution layer type")
    return conv_layer, list(kernel)


def _pool_layers_from_kind_and_dimension(
    dimension,
    pool_type,
    pool_number,
    pool_x,
    pool_y,
    pool_z,
):
    if dimension == 4 and pool_type == "max":
        return [
            MaxPooling3D(pool_size=(pool_x, pool_y, pool_z)) for _ in range(pool_number)
        ]
    if dimension == 3 and pool_type == "max":
        return [MaxPooling2D(pool_size=(pool_x, pool_y)) for _ in range(pool_number)]
    if dimension == 2 and pool_type == "max":
        return [MaxPooling1D(pool_size=pool_x) for _ in range(pool_number)]
    if dimension == 4 and pool_type == "average":
        return [
            AveragePooling3D(pool_size=(pool_x, pool_y, pool_z))
            for _ in range(pool_number)
        ]
    if dimension == 3 and pool_type == "average":
        return [
            AveragePooling2D(pool_size=(pool_x, pool_y)) for _ in range(pool_number)
        ]
    if dimension == 2 and pool_type == "average":
        return [AveragePooling1D(pool_size=pool_x) for _ in range(pool_number)]
    raise ValueError(f"Unknown pooling type: {pool_type} for dimension: {dimension}")


def _upsampler(dimension, pool_x, pool_y, pool_z):
    if dimension == 4:
        return UpSampling3D(size=(pool_x, pool_y, pool_z))
    if dimension == 3:
        return UpSampling2D(size=(pool_x, pool_y))
    if dimension == 2:
        return UpSampling1D(size=pool_x)


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


def _activation_layer(activation: str) -> Activation:
    return ACTIVATION_CLASSES.get(activation, None) or Activation(
        ACTIVATION_FUNCTIONS.get(activation, None) or activation,
    )


def _normalization_layer(norm: str) -> Layer:
    if not norm:
        return lambda x: x
    return NORMALIZATION_CLASSES[norm]()


def _dropout_layer(dimension: int, dropout_type: str, dropout_rate: float):
    if dimension == 4 and dropout_type == "spatial_dropout":
        return SpatialDropout3D(dropout_rate)
    if dimension == 3 and dropout_type == "spatial_dropout":
        return SpatialDropout2D(dropout_rate)
    if dimension == 2 and dropout_type == "spatial_dropout":
        return SpatialDropout1D(dropout_rate)
    if dropout_type == "dropout":
        return Dropout(dropout_rate)
    return lambda x: x


def saliency_map(
    input_tensor: np.ndarray,
    model: Model,
    output_layer_name: str,
    output_index: int,
) -> np.ndarray:
    """
    Compute saliency maps of the given model (presumably already trained) on a
    batch of inputs with respect to the desired output layer and index.

    For example, with a trinary classification layer called quality and classes
    good, medium, and bad output layer name would be "quality_output"
    and output_index would be 0 to get gradients w.r.t. good, 1 to get gradients
    w.r.t. medium, and 2 for gradients w.r.t. bad.

    :param input_tensor: A batch of input tensors
    :param model: A trained model expecting those inputs
    :param output_layer_name: The name of the output layer that the derivative will
                              be taken with respect to
    :param output_index: The index within the output layer that the derivative will
                         be taken with respect to

    :return: Array of the gradients same shape as input_tensor
    """
    get_gradients = _gradients_from_output(model, output_layer_name, output_index)
    _, gradients = get_gradients([input_tensor])
    return gradients


def _gradients_from_output(model, output_layer, output_index):
    K.set_learning_phase(1)
    input_tensor = model.input
    x = model.get_layer(output_layer).output[:, output_index]
    grads = K.gradients(x, input_tensor)[0]
    grads /= (
        K.sqrt(K.mean(K.square(grads))) + 1e-6
    )  # normalization trick: we normalize the gradient
    iterate = K.function([input_tensor], [x, grads])
    return iterate


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


def gini_loss(y_true: np.ndarray, y_est: np.ndarray):
    y_est = np.array([np.where(x == max(x))[0][0] for x in y_est])
    gini = 0
    for label_est in np.unique(y_est):
        indices = np.where(y_est == label_est)[0]
        p = 0
        for label_true in np.unique(y_true):
            p = len(np.where(y_true[indices] == label_true)[0]) / len(indices)
            gini += p * (1 - p) * len(indices)
    return gini
