# Imports: standard library
from typing import Dict, List, Callable

# Imports: third party
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    Dense,
    Layer,
    Conv1D,
    Flatten,
    Multiply,
    Activation,
    BatchNormalization,
    GlobalAveragePooling1D,
)
from tensorflow.keras.models import Model


class GroupConv(Layer):
    """1D group convolution"""

    def __init__(self, filters: int, groups: int, kernel_size: int, stride: int):
        super(GroupConv, self).__init__()
        self.filters = filters
        self.splits = self._filters_per_conv(filters, groups)
        self.groups = groups
        self.convs = [
            Conv1D(
                filters=split,
                kernel_size=kernel_size,
                strides=stride,
                padding="same",
                kernel_initializer="he_normal",
                use_bias=False,
            )
            for split in self.splits
        ]

    @staticmethod
    def _filters_per_conv(total_filters: int, groups: int) -> List[int]:
        """Inspired by numpy array_split source"""
        filters_per_section, extras = divmod(total_filters, groups)
        return extras * [filters_per_section + 1] + (groups - extras) * [
            filters_per_section,
        ]

    def call(self, inputs, **kwargs):
        split_sizes = self._filters_per_conv(inputs.shape[-1], self.groups)
        split_inputs = tf.split(inputs, split_sizes, axis=-1)
        x_outputs = [c(x) for x, c in zip(split_inputs, self.convs)]
        x = tf.concat(x_outputs, axis=-1)
        return x


class Stem(Layer):
    def __init__(self, width: int, kernel_size: int, activation: Activation):
        super(Stem, self).__init__()
        self.conv = Conv1D(
            filters=width,
            strides=2,
            padding="same",
            kernel_size=kernel_size,
            kernel_initializer="he_normal",
            use_bias=False,
        )
        self.bn = BatchNormalization()
        self.activation = activation

    def call(self, inputs, **kwargs):
        return self.activation(self.bn(self.conv(inputs)))


class SqueezeExcite(Layer):
    def __init__(self, ratio: int = 16):
        super(SqueezeExcite, self).__init__()
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.dense_1 = Dense(
            channels // self.ratio,
            activation="relu",
            kernel_initializer="he_normal",
            use_bias=False,
        )
        self.dense_2 = Dense(
            channels,
            activation="sigmoid",
            kernel_initializer="he_normal",
            use_bias=False,
        )

    def call(self, inputs, **kwargs):
        x = GlobalAveragePooling1D()(inputs)
        x = Flatten()(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return Multiply()([inputs, x])


class Block(Layer):
    """Block from RegNetYBody"""

    def __init__(
        self,
        width: int,
        groups: int,
        stride: int,
        kernel_size: int,
        activation: Callable,
    ):
        super(Block, self).__init__()
        if stride > 1:
            self.shortcut_conv = Conv1D(
                filters=width,
                strides=stride,
                padding="same",
                kernel_size=kernel_size,
                kernel_initializer="he_normal",
                use_bias=False,
            )
            self.shortcut_bn = BatchNormalization()
        else:
            self.shortcut_conv = None

        self.conv_1 = Conv1D(
            filters=width,
            strides=1,
            padding="same",
            kernel_size=kernel_size,
            kernel_initializer="he_normal",
            use_bias=False,
        )
        self.bn_1 = BatchNormalization()
        self.group_conv = GroupConv(
            filters=width,
            groups=groups,
            stride=stride,
            kernel_size=kernel_size,
        )
        self.bn_2 = BatchNormalization()
        self.conv_3 = Conv1D(
            filters=width,
            strides=1,
            padding="same",
            kernel_size=kernel_size,
            kernel_initializer="he_normal",
            use_bias=False,
        )
        self.bn_3 = BatchNormalization()
        self.activation = activation
        self.se = SqueezeExcite()
        self.add = Add()

    def call(self, inputs, **kwargs):
        shortcut_x = tf.identity(inputs)
        if self.shortcut_conv is not None:
            shortcut_x = self.shortcut_conv(shortcut_x)
            shortcut_x = self.shortcut_bn(shortcut_x)
            shortcut_x = self.activation(shortcut_x)

        x = self.conv_1(inputs)
        x = self.activation(x)
        x = self.bn_1(x)

        x = self.group_conv(x)
        x = self.activation(x)
        x = self.bn_2(x)

        x = self.conv_3(x)
        x = self.activation(x)
        x = self.bn_3(x)

        x = self.se(x)
        x = self.add([x, shortcut_x])
        return x


class Stage(Layer):
    def __init__(
        self,
        width: int,
        depth: int,
        group_size: int,
        kernel_size: int,
        activation: Callable,
    ):
        super(Stage, self).__init__()
        self.width = width
        self.group_size = group_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.depth = depth

    def build(self, input_shape):
        channels = input_shape[-1]
        self.blocks = [
            Block(
                width=self.width,
                groups=channels // self.group_size,
                stride=2,
                kernel_size=self.kernel_size,
                activation=self.activation,
            ),
        ]
        for _ in range(self.depth - 1):
            self.blocks.append(
                Block(
                    width=self.width,
                    groups=channels // self.group_size,
                    stride=1,
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                ),
            )

    def call(self, inputs, **kwargs):
        x = inputs
        for block in self.blocks:
            x = block(x)
        return x


class RegNetYBody(Layer):
    def __init__(
        self,
        kernel_size: int,
        group_size: int,
        depth: int,
        initial_width: int,
        width_growth_rate: int,
        width_quantization: int,
    ):
        super(RegNetYBody, self).__init__()
        self.stem = Stem(
            initial_width,
            kernel_size * 3,
            tf.nn.swish,
        )
        widths = self.find_widths(
            depth,
            initial_width,
            width_growth_rate,
            width_quantization,
        )
        widths, depths = np.unique(widths, return_counts=True)
        self.stages = [
            Stage(
                width=width,
                depth=depth,
                group_size=group_size,
                kernel_size=kernel_size,
                activation=tf.nn.swish,
            )
            for width, depth in zip(widths, depths)
        ]

    @staticmethod
    def find_widths(
        depth: int,
        initial_width: int,
        width_growth_rate: float,
        width_quantization: float,
    ):
        idx = np.arange(0, depth)
        unquantized = initial_width + width_growth_rate * idx
        s = np.log(unquantized / initial_width) / np.log(width_quantization)
        return (initial_width * width_quantization ** np.round(s)).astype(int)

    def call(self, inputs, **kwargs):
        x = self.stem(inputs)
        for stage in self.stages:
            x = stage(x)
        return x


class ML4C3Regnet(Model):
    """Compatible with ML4C3 TensorGenerator"""

    def __init__(
        self,
        kernel_size: int,
        group_size: int,
        depth: int,
        initial_width: int,
        width_growth_rate: int,
        width_quantization: int,
        input_name: str,
        output_name_to_shape: Dict[str, int],
    ):
        super(ML4C3Regnet, self).__init__()
        self.body = RegNetYBody(
            kernel_size,
            group_size,
            depth,
            initial_width,
            width_growth_rate,
            width_quantization,
        )
        self.input_name = input_name
        self.denses = {
            name: Dense(shape, name=name)
            for name, shape in output_name_to_shape.items()
        }

    def call(self, inputs, training=None, mask=None):
        x = inputs[self.input_name]
        x = self.body(x)
        embed = GlobalAveragePooling1D(name="embed")(x)
        return {name: dense(embed) for name, dense in self.denses.items()}


def _test_regnet():
    """a little sanity check por moi"""
    output_name_to_shape = {"age": 1, "sex": 2}
    reg = ML4C3Regnet(
        depth=10,
        initial_width=32,
        width_growth_rate=3,
        width_quantization=2,
        group_size=3,
        kernel_size=3,
        input_name="ecg",
        output_name_to_shape=output_name_to_shape,
    )
    batch_size = 3
    x = {"ecg": np.random.randn(batch_size, 100, 3).astype(np.float32)}
    out = reg(x)
    for name, y in out.items():
        assert out[name].shape == (batch_size, output_name_to_shape[name])
