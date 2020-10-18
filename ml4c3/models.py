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
from tensorflow.keras import backend as K
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
from ml4c3.optimizers import NON_KERAS_OPTIMIZERS, get_optimizer
from ml4c3.definitions.globals import MODEL_EXT
from ml4c3.tensormap.TensorMap import TensorMap

CHANNEL_AXIS = -1  # Set to 1 for Theano backend
LANGUAGE_MODEL_SUFFIX = "_next_character"

tfd = tfp.distributions


class BottleneckType(Enum):
    FlattenRestructure = (
        auto()
    )  # All decoder outputs are flattened to put into embedding
    GlobalAveragePoolStructured = (
        auto()
    )  # Structured (not flat) decoder outputs are global average pooled
    Variational = (
        auto()
    )  # All decoder outputs are flattened then variationally sampled to put into embedding


def make_shallow_model(
    tensor_maps_in: List[TensorMap],
    tensor_maps_out: List[TensorMap],
    optimizer: str,
    learning_rate: float,
    learning_rate_schedule: str,
    l1: float,
    l2: float,
    model_file: str = None,
    donor_layers: str = None,
    **kwargs,
) -> Model:
    """Make a shallow model (e.g. linear or logistic regression)

    :param tensor_maps_in: List of input TensorMaps
    :param tensor_maps_out: List of output TensorMaps
    :param optimizer: which optimizer to use. See optimizers.py.
    :param learning_rate: Size of learning steps in SGD optimization
    :param learning_rate_schedule: learning rate schedule to train with, e.g. triangular
    :param l1: Optional float value to use for L1 regularization.
    :param l2: Optional float value to use for L2 regularization.
    :param model_file: Optional HD5 model file to load and return.
    :param donor_layers: Optional HD5 model file whose weights will be loaded into this model when layer names match.
    :return: a compiled keras model
    """
    if model_file is not None:
        m = load_model(model_file, custom_objects=get_metric_dict(tensor_maps_out))
        m.summary()
        logging.info("Loaded model file from: {}".format(model_file))
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
                units=len(ot.channel_map),
                activation=ot.activation,
                name=ot.output_name,
                kernel_regularizer=regularizer,
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
        optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=my_metrics,
    )
    model.summary()
    if donor_layers is not None:
        model.load_weights(donor_layers, by_name=True)
        logging.info("Loaded model weights from:{}".format(donor_layers))

    return model


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

    :param tensor_maps_in: List of input TensorMaps, only 1 input TensorMap is currently supported,
                            otherwise there are layer name collisions.
    :param tensor_maps_out: List of output TensorMaps
    :param learning_rate: Size of learning steps in SGD optimization
    :param model_file: Optional HD5 model file to load and return.
    :param donor_layers: Optional HD5 model file whose weights will be loaded into this model when layer names match.
    :return: a compiled keras model
    """
    if model_file is not None:
        m = load_model(model_file, custom_objects=get_metric_dict(tensor_maps_out))
        m.summary()
        logging.info("Loaded model file from: {}".format(model_file))
        return m

    neurons = 24
    input_tensor = residual = Input(
        shape=tensor_maps_in[0].shape, name=tensor_maps_in[0].input_name,
    )
    x = c600 = Conv1D(
        filters=neurons, kernel_size=11, activation="relu", padding="same",
    )(input_tensor)
    x = Conv1D(filters=neurons, kernel_size=51, activation="relu", padding="same")(x)
    x = MaxPooling1D(2)(x)
    x = c300 = Conv1D(
        filters=neurons, kernel_size=111, activation="relu", padding="same",
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
        tensor_maps_out[0].activation, name=tensor_maps_out[0].output_name,
    )(conv_label)
    m = Model(inputs=[input_tensor], outputs=[output_y])
    m.summary()
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0)
    m.compile(
        optimizer=opt, loss=tensor_maps_out[0].loss, metrics=tensor_maps_out[0].metrics,
    )

    if donor_layers is not None:
        m.load_weights(donor_layers, by_name=True)
        logging.info("Loaded model weights from:{}".format(donor_layers))

    return m


def _order_layers(
    layer_order: List[str],
    activate: Layer = None,
    normalize: Layer = None,
    regularize: Layer = None,
) -> Layer:
    identity = lambda x: x
    activate = activate or identity
    normalize = normalize or identity
    regularize = regularize or identity

    def ordered_layers(x):
        for order in layer_order:
            if order == "activation":
                x = activate(x)
            elif order == "normalization":
                x = normalize(x)
            elif order == "regularization":
                x = regularize(x)
            else:
                pass
        return x

    return ordered_layers


Tensor = tf.Tensor
Encoder = Callable[[Tensor], Tuple[Tensor, List[Tensor]]]
Decoder = Callable[
    [Tensor, Dict[TensorMap, List[Tensor]], Dict[TensorMap, Tensor]], Tensor,
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
        regularization: str,
        regularization_rate: float,
        layer_order: List[str],
        dilate: bool,
    ):
        block_size = len(filters_per_conv)
        assert len(conv_x) == len(conv_y) == len(conv_z) == block_size
        conv_layer, kernels = _conv_layer_from_kind_and_dimension(
            dimension, conv_layer_type, conv_x, conv_y, conv_z,
        )
        self.conv_layers = [
            conv_layer(
                filters=num_filters,
                kernel_size=kernel,
                padding="same",
                dilation_rate=2 ** i if dilate else 1,
            )
            for i, (num_filters, kernel) in enumerate(zip(filters_per_conv, kernels))
        ]
        self.activations = [_activation_layer(activation) for _ in range(block_size)]
        self.normalizations = [
            _normalization_layer(normalization) for _ in range(block_size)
        ]
        self.regularizations = [
            _regularization_layer(dimension, regularization, regularization_rate)
            for _ in range(block_size)
        ]
        residual_conv_layer, _ = _conv_layer_from_kind_and_dimension(
            dimension, "conv", conv_x, conv_y, conv_z,
        )
        self.residual_convs = [
            residual_conv_layer(
                filters=filters_per_conv[0], kernel_size=_one_by_n_kernel(dimension),
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
        for convolve, activate, normalize, regularize, one_by_n_convolve in zip(
            self.conv_layers,
            self.activations,
            self.normalizations,
            self.regularizations,
            [None] + self.residual_convs,
        ):
            x = _order_layers(self.layer_order, activate, normalize, regularize)(
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
        regularization: str,
        regularization_rate: float,
        layer_order: List[str],
    ):
        conv_layer, kernels = _conv_layer_from_kind_and_dimension(
            dimension, conv_layer_type, conv_x, conv_y, conv_z,
        )
        self.conv_layers = [
            conv_layer(filters=filters, kernel_size=kernel, padding="same")
            for kernel in kernels
        ]
        self.activations = [_activation_layer(activation) for _ in range(block_size)]
        self.normalizations = [
            _normalization_layer(normalization) for _ in range(block_size)
        ]
        self.regularizations = [
            _regularization_layer(dimension, regularization, regularization_rate)
            for _ in range(block_size)
        ]
        self.layer_order = layer_order
        logging.info(
            "Dense Block Convolutional Layers (num_filters, kernel_size):"
            f" {list(zip([filters]*len(kernels), kernels))}",
        )

    def __call__(self, x: Tensor) -> Tensor:
        dense_connections = [x]
        for i, (convolve, activate, normalize, regularize) in enumerate(
            zip(
                self.conv_layers,
                self.activations,
                self.normalizations,
                self.regularizations,
            ),
        ):
            x = _order_layers(self.layer_order, activate, normalize, regularize)(
                convolve(x),
            )
            if (
                i < len(self.conv_layers) - 1
            ):  # output of block does not get concatenated to
                dense_connections.append(x)
                x = Concatenate()(
                    dense_connections[:],
                )  # [:] is necessary because of tf weirdness
        return x


class FullyConnectedBlock:
    def __init__(
        self,
        *,
        widths: List[int],
        activation: str,
        normalization: str,
        regularization: str,
        regularization_rate: float,
        layer_order: List[str],
        is_encoder: bool = False,
        name: str = None,
    ):
        """
        Creates a fully connected block with dense, activation, regularization, and normalization layers

        :param widths: number of neurons in each dense layer
        :param activation: string name of activation function
        :param normalization: optional string name of normalization function
        :param regularization: optional string name of regularization function
        :param regularization_rate: if regularization is applied, the rate at which each dense layer is regularized
        :param layer_order: list of strings specifying the activation, normalization, and regularization layers after the dense layer
        :param is_encoder: boolean indicator if fully connected block is an input block
        :param name: name of last dense layer in fully connected block, otherwise all dense layers are named ordinally e.g. dense_3 for the third dense layer in the model
        """
        final_dense = (
            Dense(units=widths[-1], name=name) if name else Dense(units=widths[-1])
        )
        self.denses = [Dense(units=width) for width in widths[:-1]] + [final_dense]
        self.activations = [_activation_layer(activation) for _ in widths]
        self.regularizations = [
            _regularization_layer(1, regularization, regularization_rate)
            for _ in widths
        ]
        self.norms = [_normalization_layer(normalization) for _ in widths]
        self.is_encoder = is_encoder
        self.layer_order = layer_order

    def __call__(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        for dense, normalize, activate, regularize in zip(
            self.denses, self.norms, self.activations, self.regularizations,
        ):
            x = _order_layers(self.layer_order, activate, normalize, regularize)(
                dense(x),
            )
        if self.is_encoder:
            return x, []
        return x


def adaptive_normalize_from_tensor(tensor: Tensor, target: Tensor) -> Tensor:
    """Uses Dense layers to convert `tensor` to a mean and standard deviation to normalize `target`"""
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
            loc=tf.zeros([latent_size]), scale_identity_multiplier=1.0,
        )

    def call(self, mu: Tensor, log_sigma: Tensor, **kwargs):
        """mu and sigma must be shape (None, latent_size)"""
        approx_posterior = tfd.MultivariateNormalDiag(
            loc=mu, scale_diag=tf.math.exp(log_sigma),
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
        regularization: str,
        regularization_rate: float,
        layer_order: List[str],
        pre_decoder_shapes: Dict[TensorMap, Optional[Tuple[int, ...]]],
    ):
        self.fully_connected = (
            FullyConnectedBlock(
                widths=fully_connected_widths,
                activation=activation,
                normalization=normalization,
                regularization=regularization,
                regularization_rate=regularization_rate,
                layer_order=layer_order,
            )
            if fully_connected_widths
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
        self.latent_size = latent_size
        self.sampler: Callable = VariationalDiagNormal(latent_size)
        self.no_restructures = [
            tm for tm, shape in pre_decoder_shapes.items() if shape is None
        ]

    def __call__(
        self, encoder_outputs: Dict[TensorMap, Tensor],
    ) -> Dict[TensorMap, Tensor]:
        y = [Flatten()(x) for x in encoder_outputs.values()]
        if len(y) > 1:
            y = concatenate(y)
        else:
            y = y[0]
        y = self.fully_connected(y) if self.fully_connected else y
        mu = Dense(self.latent_size, name="embed")(y)
        log_sigma = Dense(self.latent_size, name="log_sigma")(y)
        y = self.sampler(mu, log_sigma)
        return {
            **{tm: restructure(y) for tm, restructure in self.restructures.items()},
            **{tm: y for tm in self.no_restructures},
        }


class ConcatenateRestructure:
    """
    Flattens or GAPs then concatenates all inputs, applies a dense layer, then restructures to provided shapes
    """

    def __init__(
        self,
        pre_decoder_shapes: Dict[TensorMap, Optional[Tuple[int, ...]]],
        activation: str,
        normalization: str,
        widths: List[int],
        regularization: str,
        regularization_rate: float,
        layer_order: List[str],
        bottleneck_type: BottleneckType,
    ):
        self.fully_connected = (
            FullyConnectedBlock(
                widths=widths,
                activation=activation,
                normalization=normalization,
                regularization=regularization,
                regularization_rate=regularization_rate,
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
        self, encoder_outputs: Dict[TensorMap, Tensor],
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
    ):
        self.input_shapes = output_shape
        self.dense = Dense(units=int(np.prod(output_shape)))
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
        filters_per_dense_block: List[int],
        dimension: int,
        res_filters: List[int],
        conv_layer_type: str,
        conv_x: List[int],
        conv_y: List[int],
        conv_z: List[int],
        block_size: int,
        activation: str,
        normalization: str,
        regularization: str,
        regularization_rate: float,
        layer_order: List[str],
        dilate: bool,
        pool_after_final_dense_block: bool,
        pool_type: str,
        pool_x: int,
        pool_y: int,
        pool_z: int,
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
            regularization=regularization,
            regularization_rate=regularization_rate,
            dilate=dilate,
            layer_order=layer_order,
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
                regularization=regularization,
                regularization_rate=regularization_rate,
                layer_order=layer_order,
            )
            for filters, x, y, z in zip(
                filters_per_dense_block, dense_x, dense_y, dense_z,
            )
        ]

        self.pool_after_final_dense_block = pool_after_final_dense_block

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
    Given the number of blocks in the decoder and the upsample rates, return required input shape to get to output shape
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
        self, tensor_map_out: TensorMap,
    ):
        self.dense = Dense(
            units=tensor_map_out.shape[0],
            name=tensor_map_out.output_name,
            activation=tensor_map_out.activation,
        )

    def __call__(
        self, x: Tensor, _, decoder_outputs: Dict[TensorMap, Tensor],
    ) -> Tensor:
        return self.dense(x)


class ConvDecoder:
    def __init__(
        self,
        *,
        tensor_map_out: TensorMap,
        filters_per_dense_block: List[int],
        conv_layer_type: str,
        conv_x: List[int],
        conv_y: List[int],
        conv_z: List[int],
        block_size: int,
        activation: str,
        normalization: str,
        regularization: str,
        regularization_rate: float,
        layer_order: List[str],
        upsample_x: int,
        upsample_y: int,
        upsample_z: int,
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
                regularization=regularization,
                regularization_rate=regularization_rate,
                layer_order=layer_order,
            )
            for filters, x, y, z in zip(filters_per_dense_block, conv_x, conv_y, conv_z)
        ]
        conv_layer, _ = _conv_layer_from_kind_and_dimension(
            dimension, "conv", conv_x, conv_y, conv_z,
        )
        self.conv_label = conv_layer(
            tensor_map_out.shape[-1],
            _one_by_n_kernel(dimension),
            activation=tensor_map_out.activation,
            name=tensor_map_out.output_name,
        )
        self.upsamples = [
            _upsampler(dimension, upsample_x, upsample_y, upsample_z)
            for _ in range(len(filters_per_dense_block) + 1)
        ]

    def __call__(
        self, x: Tensor, intermediates: Dict[TensorMap, List[Tensor]], _,
    ) -> Tensor:
        for i, (dense_block, upsample) in enumerate(
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
    tensor_maps_in: List[TensorMap],
    tensor_maps_out: List[TensorMap],
    activation: str,
    learning_rate: float,
    bottleneck_type: BottleneckType,
    optimizer: str,
    dense_layers: List[int] = None,
    dropout: float = None,
    conv_layers: List[int] = None,
    dense_blocks: List[int] = None,
    block_size: int = None,
    conv_type: str = None,
    conv_normalize: str = None,
    conv_regularize: str = None,
    layer_order: List[str] = None,
    conv_x: List[int] = None,
    conv_y: List[int] = None,
    conv_z: List[int] = None,
    conv_dropout: float = None,
    conv_dilate: bool = None,
    pool_after_final_dense_block: bool = None,
    pool_type: str = None,
    pool_x: int = None,
    pool_y: int = None,
    pool_z: int = None,
    learning_rate_schedule: str = None,
    directly_embed_and_repeat: int = None,
    nest_model: List[List[str]] = None,
    model_file: str = None,
    donor_layers: str = None,
    remap_layer: Dict[str, str] = None,
    freeze_donor_layers: bool = None,
    **kwargs,
) -> Model:
    """Make multi-task, multi-modal feed forward neural network for all kinds of prediction

    This model factory can be used to make networks for classification, regression, and segmentation
    The tasks attempted are given by the output TensorMaps.
    The modalities and the first layers in the architecture are determined by the input TensorMaps.

    Hyperparameters are exposed to the command line.
    Model summary printed to output

    :param tensor_maps_in: List of input TensorMaps
    :param tensor_maps_out: List of output TensorMaps
    :param activation: Activation function as a string (e.g. 'relu', 'linear, or 'softmax)
    :param learning_rate: learning rate for optimizer
    :param bottleneck_type: How to merge the representations coming from the different input modalities
    :param dense_layers: List of number of filters in each dense layer.
    :param dropout: Dropout rate in dense layers
    :param conv_layers: List of number of filters in each convolutional layer
    :param dense_blocks: List of number of filters in densenet modules for densenet convolutional models
    :param block_size: Number of layers within each Densenet module for densenet convolutional models
    :param conv_type: Type of convolution to use, e.g. separable
    :param conv_normalize: Type of normalization layer for convolutions, e.g. batch norm
    :param conv_regularize: Type of regularization for convolutions, e.g. dropout
    :param layer_order: Order of activation, normalization, and regularization layers
    :param conv_x: Size of X dimension for 2D and 3D convolutional kernels
    :param conv_y: Size of Y dimension for 2D and 3D convolutional kernels
    :param conv_z: Size of Z dimension for 3D convolutional kernels
    :param conv_dropout: Dropout rate in convolutional layers
    :param conv_dilate: whether to use dilation in conv layers
    :param pool_after_final_dense_block: Add pooling layer after final dense block
    :param pool_type: Max or average pooling following convolutional blocks
    :param pool_x: Pooling in the X dimension for Convolutional models.
    :param pool_y: Pooling in the Y dimension for Convolutional models.
    :param pool_z: Pooling in the Z dimension for 3D Convolutional models.
    :param optimizer: which optimizer to use. See optimizers.py.
    :return: a compiled keras model
    :param learning_rate_schedule: learning rate schedule to train with, e.g. triangular
    :param model_file: HD5 model file to load and return.
    :param donor_layers: HD5 model file whose weights will be loaded into this model when layer names match.
    :param freeze_donor_layers: Whether to freeze layers from loaded from donor_layers
    :param remap_layer: Dictionary remapping layers from donor_layers to layers in current model
    :param directly_embed_and_repeat: If set, directly embed input tensors (without passing to a dense layer) into concatenation layer, and repeat each input N times, where N is this argument's value. To directly embed a feature without repetition, set to 1.
    :param nest_model: Embed a nested model ending at the specified layer before the bottleneck layer of the current model. List of models to embed and layer name of embedded layer.
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

    dense_normalize = conv_normalize
    dense_regularize = "dropout" if dropout else None
    dense_regularize_rate = dropout
    conv_regularize_rate = conv_dropout

    # list of filter dimensions should match the number of convolutional layers = len(dense_blocks) + [ + len(conv_layers) if convolving input tensors]
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

    encoders: Dict[TensorMap:Layer] = {}
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
                normalization=conv_normalize,
                regularization=conv_regularize,
                layer_order=layer_order,
                regularization_rate=conv_regularize_rate,
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
                    regularization=dense_regularize,
                    layer_order=layer_order,
                    regularization_rate=dense_regularize_rate,
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
            regularization=dense_regularize,
            regularization_rate=dense_regularize_rate,
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
            regularization=dense_regularize,
            regularization_rate=dense_regularize_rate,
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
                normalization=conv_normalize,
                regularization=conv_regularize,
                regularization_rate=conv_regularize_rate,
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

    # load layers for transfer learning
    if donor_layers is not None:
        loaded = 0
        freeze = freeze_donor_layers or False
        layer_map = remap_layer or dict()
        try:
            m_other = load_model(
                donor_layers, custom_objects=custom_dict, compile=False,
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
        TensorMap, Tuple[Tensor, List[Tensor]],
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
            bottle_neck_outputs[tm], encoder_intermediates, decoder_outputs,
        )

    return Model(inputs=inputs, outputs=list(decoder_outputs.values()))


def train_model_from_datasets(
    model: Model,
    train_dataset: tf.data.Dataset,
    valid_dataset: Optional[tf.data.Dataset],
    epochs: int,
    patience: int,
    learning_rate_patience: int,
    learning_rate_reduction: float,
    output_folder: str,
    run_id: str,
    image_ext: str,
    return_history: bool = False,
    plot: bool = True,
) -> Union[Model, Tuple[Model, History]]:
    """
    Train a model from tensorflow.data.Datasets for validation and training data.

    Training data lives on disk and is dynamically loaded by the Datasets.
    Plots the metric history after training. Creates a directory to save weights, if necessary.

    :param model: The model to optimize
    :param train_dataset: Dataset that yields batches of training data
    :param valid_dataset: Dataset that yields batches of validation data
    :param epochs: Maximum number of epochs to run regardless of Early Stopping
    :param patience: Number of epochs to wait before reducing learning rate
    :param learning_rate_patience: Number of epochs without validation loss improvement to wait before reducing learning rate
    :param learning_rate_reduction: Scale factor to reduce learning rate by
    :param output_folder: Directory where output file will be stored
    :param run_id: User-chosen string identifying this run
    :param image_ext: File format of saved image
    :param return_history: Whether or not to return history from training
    :param plot: Whether or not to plot metrics from training
    :return: The optimized model which achieved the best validation loss or training loss if validation data was not provided
    """
    model_file = os.path.join(output_folder, run_id, "model_weights" + MODEL_EXT)
    if not os.path.exists(os.path.dirname(model_file)):
        os.makedirs(os.path.dirname(model_file))

    if plot:
        image_path = os.path.join(
            output_folder, run_id, "architecture_graph" + image_ext,
        )
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
            training_steps=None,
            title=run_id,
            image_ext=image_ext,
            prefix=os.path.dirname(model_file),
        )

    # load the weights from model which achieved the best validation loss
    model.load_weights(model_file)
    if return_history:
        return model, history
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
            filepath=model_file, monitor=monitor, verbose=1, save_best_only=True,
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


def _conv_layer_from_kind_and_dimension(
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
        raise ValueError(
            f"Unknown convolution type: {conv_layer_type} for dimension: {dimension}",
        )
    return conv_layer, list(kernel)


def _pool_layers_from_kind_and_dimension(
    dimension, pool_type, pool_number, pool_x, pool_y, pool_z,
):
    if dimension == 4 and pool_type == "max":
        return [
            MaxPooling3D(pool_size=(pool_x, pool_y, pool_z)) for _ in range(pool_number)
        ]
    elif dimension == 3 and pool_type == "max":
        return [MaxPooling2D(pool_size=(pool_x, pool_y)) for _ in range(pool_number)]
    elif dimension == 2 and pool_type == "max":
        return [MaxPooling1D(pool_size=pool_x) for _ in range(pool_number)]
    elif dimension == 4 and pool_type == "average":
        return [
            AveragePooling3D(pool_size=(pool_x, pool_y, pool_z))
            for _ in range(pool_number)
        ]
    elif dimension == 3 and pool_type == "average":
        return [
            AveragePooling2D(pool_size=(pool_x, pool_y)) for _ in range(pool_number)
        ]
    elif dimension == 2 and pool_type == "average":
        return [AveragePooling1D(pool_size=pool_x) for _ in range(pool_number)]
    else:
        raise ValueError(
            f"Unknown pooling type: {pool_type} for dimension: {dimension}",
        )


def _upsampler(dimension, pool_x, pool_y, pool_z):
    if dimension == 4:
        return UpSampling3D(size=(pool_x, pool_y, pool_z))
    elif dimension == 3:
        return UpSampling2D(size=(pool_x, pool_y))
    elif dimension == 2:
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


def _regularization_layer(dimension: int, regularization_type: str, rate: float):
    if dimension == 4 and regularization_type == "spatial_dropout":
        return SpatialDropout3D(rate)
    elif dimension == 3 and regularization_type == "spatial_dropout":
        return SpatialDropout2D(rate)
    elif dimension == 2 and regularization_type == "spatial_dropout":
        return SpatialDropout1D(rate)
    elif regularization_type == "dropout":
        return Dropout(rate)
    else:
        return lambda x: x


def saliency_map(
    input_tensor: np.ndarray, model: Model, output_layer_name: str, output_index: int,
) -> np.ndarray:
    """Compute saliency maps of the given model (presumably already trained) on a batch of inputs with respect to the desired output layer and index.

    For example, with a trinary classification layer called quality and classes good, medium, and bad output layer name would be "quality_output"
    and output_index would be 0 to get gradients w.r.t. good, 1 to get gradients w.r.t. medium, and 2 for gradients w.r.t. bad.

    :param input_tensor: A batch of input tensors
    :param model: A trained model expecting those inputs
    :param output_layer_name: The name of the output layer that the derivative will be taken with respect to
    :param output_index: The index within the output layer that the derivative will be taken with respect to

    :return: Array of the gradients same shape as input_tensor
    """
    get_gradients = _gradients_from_output(model, output_layer_name, output_index)
    activation, gradients = get_gradients([input_tensor])
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