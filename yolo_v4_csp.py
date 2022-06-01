"""Build a scaled YOLOv4-CSP detector in TensorFlow 2.8
基于 2021 年 2 月发布的第二版 Scaled-YOLOv4： https://arxiv.org/abs/2011.08036
"""

import sys
from collections.abc import Iterable

import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

# 设置如下全局变量，用大写字母。

CLASSES = 80  # 如果使用 COCO 2017 数据集，则需要探测 80 个类别。

# 为了获得更快的速度，使用小的特征图。原始模型的 p5 特征图为 19x19，模型输入图片为 608x608
FEATURE_MAP_P5 = np.array((19, 19))  # 19, 19
FEATURE_MAP_P4 = FEATURE_MAP_P5 * 2  # 38, 38
FEATURE_MAP_P3 = FEATURE_MAP_P4 * 2  # 76, 76

# 格式为 height, width。当两者大小不同时尤其要注意。是 FEATURE_MAP_P3 的 8 倍。
MODEL_IMAGE_SIZE = FEATURE_MAP_P3 * 8  # 608, 608

# 如果使用不同大小的 FEATURE_MAP，应该相应调整预设框的大小。
resize_scale = 19 / FEATURE_MAP_P5[0]

# 根据 YOLO V3 论文的 2.3 节，设置 ANCHOR_BOXES 。除以比例后取整数部分。
ANCHOR_BOXES_P5 = [(116 // resize_scale, 90 // resize_scale),
                   (156 // resize_scale, 198 // resize_scale),
                   (373 // resize_scale, 326 // resize_scale)]
ANCHOR_BOXES_P4 = [(30 // resize_scale, 61 // resize_scale),
                   (62 // resize_scale, 45 // resize_scale),
                   (59 // resize_scale, 119 // resize_scale)]
ANCHOR_BOXES_P3 = [(10 // resize_scale, 13 // resize_scale),
                   (16 // resize_scale, 30 // resize_scale),
                   (33 // resize_scale, 23 // resize_scale)]

EPSILON = 1e-10


def check_inf_nan(inputs, name, max_value=50000):
    """检查输入中是否存在 inf 和 NaN 值，并进行提示。

    Arguments:
        inputs: 一个数据类型的张量，可以是任意形状。
        name: 一个字符串，是输入张量的名字。
        max_value: 一个整数，如果当前输入张量的最大值，大于 max_value ，则打印输出
            当前输入张量的最大值。尤其注意数值在超过 50,000 之后，是有可能无法使用混合精
            度计算的，因为 float16 格式下，数值达到 65520 时就会产生 inf 值。
    """

    if type(inputs) != tuple:
        # 排除整数类型和浮点数类型，只使用张量和数组。
        if not isinstance(inputs, (int, float)):
            input_is_keras_tensor = keras.backend.is_keras_tensor(inputs)
            # 如果输入是符号张量 symbolic tensor，则不检查该张量。
            if not input_is_keras_tensor:
                input_inf = tf.math.is_inf(inputs)
                if tf.math.reduce_any(input_inf):
                    # 在图模式下运行时，print 不起作用，只能用 tf.print
                    tf.print(f'\nInf! Found in {name}, its shape: ',
                             input_inf.shape)

                input_nan = tf.math.is_nan(inputs)
                if tf.math.reduce_any(input_nan):
                    tf.print(f'\nNaN! Found in {name}, its shape: ',
                             input_nan.shape)

                current_max = tf.math.reduce_max(inputs)
                if current_max > max_value:
                    max_value = current_max
                    tf.print(f'\nIn {name}, its shape: ', inputs.shape)
                    tf.print(f'max_value: ', max_value)

    else:
        # 模型的输出值，将进入这个分支。
        for i, each_input in enumerate(inputs):
            input_is_keras_tensor = keras.backend.is_keras_tensor(each_input)
            # 如果输入是符号张量 symbolic tensor，则不检查该张量。
            if not input_is_keras_tensor:
                subname = f'{name}_{i}'
                input_inf = tf.math.is_inf(each_input)
                if tf.math.reduce_any(input_inf):
                    tf.print(f'\nInf! Found found in {subname}, its shape: ',
                             input_inf.shape)

                input_nan = tf.math.is_nan(each_input)
                if tf.math.reduce_any(input_nan):
                    tf.print(f'\nNaN! Found in {subname}, its shape: ',
                             input_nan.shape)

                current_max = tf.math.reduce_max(each_input)
                if current_max > max_value:
                    max_value = current_max
                    tf.print(f'\nIn {subname}, its shape: ', each_input.shape)
                    tf.print(f'max_value change to: ', max_value)


class MishActivation(keras.layers.Layer):
    """mish 激活函数。为了便于迁移到其它平台 serialization ，使用子类方法 subclassing，
    不使用层 layers.Lambda。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # noinspection PyMethodMayBeStatic
    def call(self, inputs):
        """Conv, BN, _activation 模块，在 backbone 和 PANet 中将被多次调用。

        Arguments：
            inputs: 一个张量。数据类型为 float32 或 float16。
        Returns:
            x: 一个张量，经过 mish 激活函数的处理，形状和输入张量相同。
        """

        x = tfa.activations.mish(inputs)

        return x

    def get_config(self):
        config = super().get_config()
        return config


# 根据需要对模型的权重 weights 使用 Constraint。注意该 class 在 TF 2.4 可以运行，
# 但在 TF 2.8 中，对同一个模型，容易产生 NaN 权重，原因不明。
class LimitWeights(tf.keras.constraints.Constraint):

    def __call__(self, w):
        clipped_weight = tf.clip_by_value(
            w, clip_value_min=-1.,
            clip_value_max=1.)

        return clipped_weight


# noinspection PyUnusedLocal
def conv_bn_mish(inputs, filters, kernel_size, strides=1, use_bias=False,
                 padding='same', separableconv=False, rate_regularizer=0,
                 training=None, convolution_only=False, conv_name=None,
                 max_weight_norm=2.0):
    """Conv, BN, mish 激活函数模块，在 backbone 和 PANet 中将被多次调用。YOLO-v4-CSP 
    中不再使用 leaky_relu 激活函数。

    Arguments：
        inputs: 一个张量。数据类型为 float32。
        filters： 一个整数，是卷积模块的过滤器数量。
        kernel_size： 一个整数，是卷积核的大小。
        strides： 一个整数，是卷积的步进值。
        use_bias： 一个布尔值，只有在为 True 时卷积层才使用 bias 权重。
        padding： 一个字符串，是卷积的 padding 方式。
        separableconv： 一个布尔值，设置是否使用 Separableconv2D。
        rate_regularizer： 一个浮点数，是卷积的 L2 正则化率。
        training: 一个布尔值，表示处于训练模式或是推理 inference 模式。
        convolution_only： 一个布尔值，为 True 时表示该模块只使用卷积，为 False 则表示
            该模块包括卷积，BN，mish 激活函数。
        conv_name： 一个字符串，设置该 conv_bn_mish 模块的名字。
        max_weight_norm： 一个浮点数，用于设置权重的最大范数 MaxNorm。
    Returns:
        x: 一个张量，经过卷积，Batch Normalization 和 mish 激活函数的处理，形状和输入张
            量相同。
    """

    # 限制权重的范数 norm，使用 MaxNorm 类。避免过大的权重导致的 NaN 问题。
    # 对卷积，可以使用 axis=[0, 1, 2] 设定方式。如果使用 axis=[0, 1]，则是更严格的方式，
    # 即限制 3x3 的卷积核范数不超过 2.
    conv_weight_limits = keras.constraints.MaxNorm(
        max_value=max_weight_norm, axis=[0, 1, 2])
    # 卷积的 bias，batch normalization 的 γ，β，都是向量，可以使用 axis=0 设定方式。
    vector_weight_limits = keras.constraints.MaxNorm(
        max_value=max_weight_norm, axis=0)

    # 限制权重范数后，不使用 regularizers 也可以。因为它们的作用都是使得权重变小。
    # regularizer_l2 = keras.regularizers.L2(rate_regularizer)

    if separableconv:
        # 不要遗漏对 bias 的限制。因为在模型的最终输出部分，只有一个卷积层，并且该层会使用
        # bias，如果忘记了设置，将可能出现 bias 的权重极大，达到 inf 的状态，然后导致
        # 出现 NaN。
        x = keras.layers.SeparableConv2D(
            filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding, use_bias=use_bias,
            depthwise_initializer=keras.initializers.HeNormal(),
            pointwise_initializer=keras.initializers.HeNormal(),
            # depthwise_regularizer=regularizer_l2,
            # pointwise_regularizer=regularizer_l2,
            # bias_regularizer=regularizer_l2,
            depthwise_constraint=conv_weight_limits,
            pointwise_constraint=conv_weight_limits,
            bias_constraint=vector_weight_limits,
            name=conv_name)(inputs)

    else:
        # 不要遗漏对 bias 的限制。因为在模型的最终输出部分，只有一个卷积层，并且该层会使用
        # bias，如果忘记了设置，将可能出现 bias 的权重极大，达到 inf 的状态，然后导致
        # 出现 NaN。
        x = keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding, use_bias=use_bias,
            kernel_initializer=keras.initializers.HeNormal(),
            # kernel_regularizer=regularizer_l2,
            # bias_regularizer=regularizer_l2,

            kernel_constraint=conv_weight_limits,  # conv_weight_limits
            bias_constraint=vector_weight_limits,  # vector_weight_limits

            name=conv_name)(inputs)

    # convolution_only 为 True，则不需要进行 BatchNormalization 和 MishActivation。
    if not convolution_only:

        # 注意 BatchNormalization 也有 γ 和 β 两个参数 weight，在对参数进行
        # regularization 和 constraint 时，也要加上这两个参数。否则它们可以变得很大，
        # 可能导致出现 NaN。
        x = keras.layers.BatchNormalization(
            # beta_regularizer=regularizer_l2,
            # gamma_regularizer=regularizer_l2,
            beta_constraint=vector_weight_limits,
            gamma_constraint=vector_weight_limits,
        )(x, training=training)

        # 下面使用 noinspection PyCallingNonCallable，是因为 TF 的版本问题，导致
        # Pycharm 无法识别 keras.layers.Layer，会出现调用报错，手动关闭此报错即可。
        # noinspection PyCallingNonCallable
        x = MishActivation()(x)

    return x


def darknet53_residual(inputs, filters, training=None):
    """CSPDarknet53 的第一个 block，使用普通的 darknet53_residual，即没有经过 CSP
    分支操作。

    Arguments:
        inputs: 一个 3D 张量。形状为 (height, width, 3)。数据类型
        为 float32 (混合精度模式下为 float16 类型）。使用时 Keras 将会自动
        插入一个第 0 维度，作为批量维度。
        filters: 一个整数，表示当前 _csp_block 输出张量的过滤器数量。
        training: 一个布尔值，表示处于训练模式或是推理 inference 模式。
    Returns:
        x: 一个 3D 张量，形状为 (height / 2, width / 2, filters)。数据类型
            为 float32 (混合精度模式下为 float16 类型）。
    """

    x = conv_bn_mish(inputs=inputs, filters=filters, kernel_size=3,
                     strides=2, training=training)

    residual = x

    x = conv_bn_mish(inputs=x, filters=(filters // 2),
                     kernel_size=1, training=training)
    x = conv_bn_mish(inputs=x, filters=filters,
                     kernel_size=3, training=training)
    x = keras.layers.Add()([x, residual])

    return x


def csp_block(inputs, filters, residual_quantity, training=None):
    """CSPDarknet53 的基本组成单元，用于在保证模型准确度的前提下，降低计算量。

    大体实现方式是，把输入张量的特征图缩小到一半，然后分成两个分支，每个分支只使用一半数量的
    过滤器，并且只在主支进行卷积和 residual block 等计算，最后再把两个分支拼接起来。

    Arguments:
        inputs: 一个 3D 张量。形状为 (height, width, 3)。数据类型
        为 float32 (混合精度模式下为 float16 类型）。使用时 Keras 将会自动
        插入一个第 0 维度，作为批量维度。
        filters: 一个整数，表示当前 _csp_block 输出张量的过滤器数量。
        residual_quantity: 一个整数，表示在主支中 residual block 的数量。
        training: 一个布尔值，表示处于训练模式或是推理 inference 模式。
    Returns:
        x: 一个 3D 张量，形状为 (height / 2, width / 2, filters)。数据类型
            为 float32 (混合精度模式下为 float16 类型）。
    """

    # 每个 csp block 的第一个卷积，使用 strides=2 进行下采样。注意第一个卷积的卷积核大小
    # 应该是 3！ 论文中配图 figure 4 有误，配图 figure 4 左下角是 CSP block，它第一个
    # 卷积核大小写的是 1，其实应该是 3.
    x = conv_bn_mish(inputs=inputs, filters=filters, kernel_size=3,
                     strides=2, training=training)

    # split_branch 作为 CSP 的一个分支，最后将和主支进行 concatenation。
    split_branch = conv_bn_mish(inputs=x, filters=(filters // 2), 
                                kernel_size=1, training=training)

    x = conv_bn_mish(inputs=x, filters=(filters // 2), 
                     kernel_size=1, training=training)
    
    for _ in range(residual_quantity):
        residual = x
        x = conv_bn_mish(inputs=x, filters=(filters // 2), 
                         kernel_size=1, training=training)
        x = conv_bn_mish(inputs=x, filters=(filters // 2), 
                         kernel_size=3, training=training)
        x = keras.layers.Add()([x, residual])

    x = conv_bn_mish(inputs=x, filters=(filters // 2), 
                     kernel_size=1, training=training)

    x = keras.layers.Concatenate()([x, split_branch])

    x = conv_bn_mish(inputs=x, filters=filters, 
                     kernel_size=1, training=training)

    return x


def csp_darknet53(inputs, training=None):
    """CSPDarknet53 的主体架构。

    Arguments:
        inputs: 一个 3D 张量，表示一批图片。形状为 (608, 608, 3)。数据类型
            为 float32 (混合精度模式下为 float16 类型）。使用时 Keras 将会自动插入
            一个第 0 维度，作为批量维度。
        training: 一个布尔值，表示处于训练模式或是推理 inference 模式。
    Returns:
        p5_backbone: 一个 3D 张量，特征图最小。如果输入图片大小为 (608, 608, 3)，
            则该特征图大小为 (19, 19, 1024)。
        p4_backbone: 一个 3D 张量，特征图中等。如果输入图片大小为 (608, 608, 3)，
            则该特征图大小为 (38, 38, 512)。
        p3_backbone: 一个 3D 张量，特征图最大。如果输入图片大小为 (608, 608, 3)，
            则该特征图大小为 (76, 76, 256)。
    """

    p5_backbone, p4_backbone, p3_backbone = None, None, None

    # 图片输入模型之后，遇到的第一个卷积模块，使用普通卷积，不使用 separableconv。
    x = conv_bn_mish(inputs=inputs, filters=32, kernel_size=3,
                     separableconv=False, training=training)

    # 第一个 block 是普通的 darknet53_residual，residual 数量为 1.
    x = darknet53_residual(inputs=x, filters=64, training=training)

    # 后面 4 个 block 才是 csp residual
    parameters_csp_block_2 = 128, 2
    parameters_csp_block_3 = 256, 8
    parameters_csp_block_4 = 512, 8
    parameters_csp_block_5 = 1024, 4

    parameters_csp_blocks = [parameters_csp_block_2, parameters_csp_block_3,
                             parameters_csp_block_4, parameters_csp_block_5]

    for parameters_csp_block in parameters_csp_blocks:

        filters = parameters_csp_block[0]
        residual_number = parameters_csp_block[1]

        x = csp_block(inputs=x, filters=filters,
                      residual_quantity=residual_number, training=training)
        # backbone 应该输出 3 个分支， p5, p4, p3。
        if x.shape[-1] == 1024:
            p5_backbone = x
        elif x.shape[-1] == 512:
            p4_backbone = x
        elif x.shape[-1] == 256:
            p3_backbone = x

    return p5_backbone, p4_backbone, p3_backbone


def spp(inputs):
    """SPP 模块.

    将输入分成 4 个分支，其中 3 个分支分别使用 5, 9, 13 的 pool_size 进行池化，然后
    和输入再拼接起来，进行返回。该模块输入和输出的形状完全相同。
    """

    # 注意必须设置 strides=1，否则默认 strides=pool_size，会进行下采样。
    maxpooling_5 = keras.layers.MaxPooling2D(
        pool_size=5, strides=1, padding='same')(inputs)

    maxpooling_9 = keras.layers.MaxPooling2D(
        pool_size=9, strides=1, padding='same')(inputs)

    maxpooling_13 = keras.layers.MaxPooling2D(
        pool_size=13, strides=1, padding='same')(inputs)

    x = keras.layers.Concatenate(axis=-1)(
        [inputs, maxpooling_5, maxpooling_9, maxpooling_13])

    return x


def reversed_csp(inputs, target_filters, reversed_csp_quantity,
                 reversed_spp_csp=False, training=None):
    """Reversed CSP 模块。在 PANet 的上采样和下采样分支中要被多次用到。

    Reversed CSP 的大体过程和 CSP 模块基本一致。区别在于两点：
    1. CSP 的主支卷积核大小为 1, 3, residual, 1，而 Reversed CSP 将顺序反过来，
    变为 3, 1, residual, 3。
    2. CSP 不改变输入张量的特征通道数量，而 Reversed CSP 将输入的特征通道数量降低
    一半，以此大幅降低计算量。

    Arguments:
        inputs: 一个 3D 张量。形状为 (height, width, filters)。数据类型
            为 float32 (混合精度模式下为 float16 类型）。使用时 Keras 将会自动
            插入一个第 0 维度，作为批量维度。
        target_filters: 一个整数，表示该模块最终输出的过滤器数量。
        reversed_csp_quantity: 一个整数，表示在主支中 reversed_csp 的数量。
        reversed_spp_csp: 一个布尔值。该值为 True 时，插入最大池化模块 SPP 。此时
            这个 Reversed CSP 模块就变成了 Reversed-CSP-SPP 模块。
            只有 backbone 输出的最小特征层 p5 会用到这个 Reversed-CSP-SPP 模块。
            (如果是 YOLO-v4 large，最小特征层则可能是 p6, p7)
        training: 一个布尔值，表示处于训练模式或是推理 inference 模式。

    Returns:
        x: 一个 3D 张量，形状为 (height, width, filters / 2)。数据类型
            为 float32 （混合精度模式下为 float16 类型）。
    """

    # 如果是 reversed_spp_csp，第一个卷积块的特征通道数和输入是一样的，等于
    # target_filters 的 2 倍。实际上无须操作，等于直接使用 CSP-BLOCK 的输出。
    if reversed_spp_csp:
        x = inputs

    # 而如果是 reversed_csp，第一个卷积块就会把特征通道数量降一半。
    else:
        x = conv_bn_mish(inputs=inputs, filters=target_filters,
                         kernel_size=1, training=training)

    # split_branch 作为 CSP 的一个分支，最后将和主支进行 concatenation。
    split_branch = conv_bn_mish(inputs=x, filters=target_filters,
                                kernel_size=1, training=training)

    for i in range(reversed_csp_quantity):

        # 主支的第一个卷积，使用 1x1 卷积把特征通道数量降为一半。
        x = conv_bn_mish(inputs=x, filters=target_filters,
                         kernel_size=1, training=training)
        # reversed CSP 的 reversed，应该是指没有 residual 模块，只是单纯的循环。
        x = conv_bn_mish(inputs=x, filters=target_filters,
                         kernel_size=3, training=training)

        # 最小特征层 p5，要使用 SPP 模块，得到 Reversed-CSP-SPP 模块。并且 spp 应该只
        # 在第 0 个循环时使用一次。
        if reversed_spp_csp and (i == 0):

            # 对于 reversed_spp_csp，在 spp 之后增加了一个 1x1 的卷积
            x = conv_bn_mish(inputs=x, filters=target_filters,
                             kernel_size=1, training=training)
            x = spp(x)

    x = keras.layers.Concatenate(axis=-1)([x, split_branch])

    x = conv_bn_mish(inputs=x, filters=target_filters,
                     kernel_size=1, training=training)

    return x


def upsampling_branch(upsampling_input, lateral_input,
                      target_filters, reversed_csp_quantity, training=None):
    """PANet 的上采样分支。对 2个输入进行拼接，然后执行 reversed CSP 操作。

    Arguments:
        upsampling_input: 一个 3D 张量。形状为(height / 2, width / 2, filters)。
            数据类型为 float32 (混合精度模式下为 float16 类型）。使用时 Keras 将
            会自动插入一个第 0 维度，作为批量维度。
        lateral_input: 一个 3D 张量。形状为 (height, width, filters)。
            数据类型为 float32 (混合精度模式下为 float16 类型）。使用时 Keras 将
            会自动插入一个第 0 维度，作为批量维度。
        target_filters: 一个整数，表示在该模块最终输出的过滤器数量。
        reversed_csp_quantity: 一个整数，表示在主支中 residual block 的数量。
        training: 一个布尔值，表示处于训练模式或是推理 inference 模式。

    Returns:
        upsampling_output: 一个 3D 张量。形状为 (height, width, filters / 2)。
            数据类型为 float32 （混合精度模式下为 float16 类型）。
    """

    # 先用 1x1 卷积，将 2 个输入调整特征通道数量，然后再拼接。
    lateral_input = conv_bn_mish(inputs=lateral_input, filters=target_filters,
                                 kernel_size=1, training=training)

    upsampling_input = conv_bn_mish(
        inputs=upsampling_input, filters=target_filters,
        kernel_size=1, training=training)

    upsampling_input = keras.layers.UpSampling2D(size=2)(upsampling_input)

    concatenated = keras.layers.Concatenate(axis=-1)(
        [lateral_input, upsampling_input])

    # 进行 Reversed CSP 操作。
    upsampling_output = reversed_csp(
        inputs=concatenated, target_filters=target_filters,
        reversed_csp_quantity=reversed_csp_quantity, training=training)

    return upsampling_output


def downsampling_branch(downsampling_input, lateral_input,
                        target_filters, reversed_csp_quantity, training=None):
    """PANet 的下下采样分支。对 2 个输入进行拼接，然后执行 reversed CSP 操作。

    Arguments:
        downsampling_input: 一个 3D 张量。形状为(height * 2, width * 2,
            filters / 2)。
            数据类型为 float32 (混合精度模式下为 float16 类型）。使用时 Keras 将
            会自动插入一个第 0 维度，作为批量维度。
        lateral_input: 一个 3D 张量。形状为 (height, width, filters)。
            数据类型为 float32 (混合精度模式下为 float16 类型）。使用时 Keras 将
            会自动插入一个第 0 维度，作为批量维度。
        target_filters: 一个整数，表示在该模块最终输出的过滤器数量。
        reversed_csp_quantity: 一个整数，表示在主支中 residual block 的数量。
        training: 一个布尔值，表示处于训练模式或是推理 inference 模式。

    Returns:
        downsampling_output: 一个 3D 张量。形状为 (height, width, filters)。
            数据类型为 float32 （混合精度模式下为 float16 类型）。
    """

    # 先用 3x3 卷积，将 downsampling_input 调整特征通道数量，然后再拼接。
    downsampling_input = conv_bn_mish(
        inputs=downsampling_input, filters=target_filters,
        kernel_size=3, strides=2, training=training)

    concatenated = keras.layers.Concatenate(axis=-1)(
        [lateral_input, downsampling_input])

    # 拼接之后，特征通道数量增大了一倍，用 reversed_csp 可以将其减半，
    # 达到 target_filters。
    downsampling_output = reversed_csp(
        inputs=concatenated, target_filters=target_filters,
        reversed_csp_quantity=reversed_csp_quantity, training=training)

    return downsampling_output


def panet(inputs, training=None):
    """对 backbone 输入的 3 个张量，先使用上采样分支对其进行处理，然后用下采样分支
    进行处理，最后返回 3 个张量。

    Arguments:
        inputs: 一个元祖，包含来自 backbone 输出的 3个 3D 张量。分别表示为
            p5_backbone, p4_backbone, p3_backbone。使用时 Keras 将
            会自动插入一个第 0 维度，作为批量维度。
        training: 一个布尔值，表示处于训练模式或是推理 inference 模式。
    Returns:
        p5_neck: 一个 3D 张量，特征图最小。形状为 (19, 19, 512)。（3 个输出
            p5_neck，p4_neck， p3_neck，均假定输入图片大小为 (608, 608, 3)）
            使用时 Keras 将会自动插入一个第 0 维度，作为批量维度。
        p4_neck: 一个 3D 张量，特征图中等。形状为 (38, 38, 256)。使用时 Keras 将
            会自动插入一个第 0 维度，作为批量维度。
        p3_neck: 一个 3D 张量，特征图最大。形状为 (76, 76, 128)。使用时 Keras 将
            会自动插入一个第 0 维度，作为批量维度。
    """

    p5_backbone, p4_backbone, p3_backbone = inputs

    # 将 PANet 看做 2 个分支，upsampling_branch 和 downsampling_branch。
    # 先处理上采样分支，必须按 p5,p4,p3 的顺序，p5_upsampling 表示在上采样分支的 p5。
    p5_upsampling = reversed_csp(
        inputs=p5_backbone, target_filters=512,
        reversed_csp_quantity=2, reversed_spp_csp=True, training=training)

    p4_upsampling = upsampling_branch(
        upsampling_input=p5_upsampling, lateral_input=p4_backbone,
        target_filters=256, reversed_csp_quantity=2, training=training)

    p3_upsampling = upsampling_branch(
        upsampling_input=p4_upsampling, lateral_input=p3_backbone,
        target_filters=128, reversed_csp_quantity=2, training=training)

    # 再处理下采样分支，必须按 p3,p4,p5 的顺序，p5_neck 表示下采样分支的 p5，同时也是
    # PANet 的输出，所以叫 p5_neck。
    p3_neck = p3_upsampling
    p4_neck = downsampling_branch(
        downsampling_input=p3_neck, lateral_input=p4_upsampling,
        target_filters=256, reversed_csp_quantity=2, training=training)

    p5_neck = downsampling_branch(
        downsampling_input=p4_neck, lateral_input=p5_upsampling,
        target_filters=512, reversed_csp_quantity=2, training=training)

    return p5_neck, p4_neck, p3_neck


def heads(inputs, training=None):
    """将来自 neck 部分的 3 个输入转换为 3个 heads。

    Arguments:
        inputs: 一个元祖，包含来自 backbone 输出的 3 个 3D 张量。分别表示为
            p5_neck, p4_neck, p3_neck。使用时 Keras 将会自动插入一个第 0 维度，
            作为批量维度。
        training: 一个布尔值，表示处于训练模式或是推理 inference 模式。
    Returns:
        p5_head: 一个 3D 张量，特征图最小。形状为 (19, 19, 255)。（3 个输出
            p5_head, p4_head, p3_head，均假定输入图片大小为 (608, 608, 3)）
            使用时 Keras 将会自动插入一个第 0 维度，作为批量维度。
        p4_head: 一个 3D 张量，特征图中等。形状为 (38, 38, 255)。使用时 Keras 将
            会自动插入一个第 0 维度，作为批量维度。
        p3_head: 一个 3D 张量，特征图最大。形状为 (76, 76, 255)。使用时 Keras 将
            会自动插入一个第 0 维度，作为批量维度。
    """

    p5_neck, p4_neck, p3_neck = inputs

    p5_head = conv_bn_mish(inputs=p5_neck, filters=1024,
                           kernel_size=3, training=training)

    # 在模型的最终输出部分，只有一个卷积层，并且该层会使用 bias。
    p5_head = conv_bn_mish(inputs=p5_head, filters=255, separableconv=False,
                           kernel_size=1, convolution_only=True, use_bias=True,
                           conv_name='p5')

    p4_head = conv_bn_mish(inputs=p4_neck, filters=512,
                           kernel_size=3, training=training)
    p4_head = conv_bn_mish(inputs=p4_head, filters=255, separableconv=False,
                           kernel_size=1, convolution_only=True, use_bias=True,
                           conv_name='p4')

    p3_head = conv_bn_mish(inputs=p3_neck, filters=256,
                           kernel_size=3, training=training)
    p3_head = conv_bn_mish(inputs=p3_head, filters=255, separableconv=False,
                           kernel_size=1, convolution_only=True, use_bias=True,
                           conv_name='p3')

    return p5_head, p4_head, p3_head


def yolo_v4_csp(inputs, training=None):
    """YOLO V4 CSP module。

    Arguments：
        inputs：一个 4D 图片张量，形状为 (batch_size, 608, 608, 3)，数据类型为
            tf.float32。可以用全局变量 MODEL_IMAGE_SIZE 设置不同大小的图片输入。
        training: 一个布尔值，用于设置模型是处在训练模式或是推理 inference 模式。
            在预测时，如果不使用 predict 方法，而是直接调用模型的个体，则必须设置参
            数 training=False，比如 model(x, training=False)。因为这样才能让模
            型的 dropout 层和 BatchNormalization 层以 inference 模式运行。而如
            果是使用 predict 方法，则不需要设置该 training 参数。
    Returns:
        head_outputs: 一个元祖，包含 3 个 tf.float32 类型的张量，张量形状为
            (batch_size, 19, 19, 255), (batch_size, 38, 38, 255),
            (batch_size, 76, 76, 255)。最后 1 个维度大小为 255，可以转换为 (3, 85)，
            表示有 3 个预测框，每个预测结果是一个长度为 85 的向量。
            在这个长度为 85 的向量中，第 0 位是置信度，第 1 位到第 81 位，代表 80 个
            类别的 one-hot 编码，最后 4 位，则是预测框的位置和坐标，格式为
            (x, y, height, width)，其中 x，y 是预测框的中心点坐标，height, width
            是预测框的高度和宽度。对于一个训练好的模型，这 4 个数值的范围都应该在
            [0, 608] 之间。
    """

    backbone_outputs = csp_darknet53(inputs=inputs, training=training)
    neck_outputs = panet(inputs=backbone_outputs, training=training)
    head_outputs = heads(inputs=neck_outputs, training=training)

    return head_outputs


def create_model(input_shape=None):
    """创建一个新的 yolo_v4_csp 模型。"""

    if input_shape is None:
        input_shape = *MODEL_IMAGE_SIZE, 3

    keras.backend.clear_session()

    # inputs 是一个 Keras tensor，也叫符号张量 symbolic tensor，这种张量没有实际的值，
    # 只是在创建模型的第一步--构建计算图时会用到。模型创建好之后就不再使用符号张量。
    inputs = keras.Input(shape=input_shape)
    outputs = yolo_v4_csp(inputs=inputs)

    model = keras.Model(
        inputs=inputs, outputs=outputs, name='yolo_v4_csp_model')

    return model


def _transform_predictions(prediction, anchor_boxes):
    """将模型的预测结果转换为模型输入的图片大小，可以在全局变量 MODEL_IMAGE_SIZE 中
    设置模型输入图片大小。

    将每个长度为 85 的预测结果进行转换，第 0 位为置信度 confidence，第 1 位到第 81
    位为分类的 one-hot 编码。置信度和分类结果都需要用 sigmoid 转换为 [0, 1]之间的数，
    相当于转换为概率值。最后 4 位是探测框的预测结果，需要根据 YOLO V3 论文进行转换。
    倒数第 4 位到倒数第 3 位为预测框的中心点，需要对中心点用 sigmoid 函数进行转换。然
    后乘以一个比例，就得到中心点在模型输入图片中的实际坐标值。
    倒数第 2 位和最后一位是探测框的宽度和高度，需要先用指数函数转换为非负数，再乘以探测
    框的高度和宽度。

    Arguments:
        prediction: 一个 3D 张量，形状为 (N, *batch_size_feature_map, 255)，是
            模型的 3 个预测结果张量之一。height, width 是特征图大小。使用时 Keras
            将会自动插入一个第 0 维度，作为批量维度。
        anchor_boxes: 一个元祖，其中包含 3 个元祖，代表了当前 prediction 所对应
            的 3 个预设框大小。

    Returns:
        transformed_prediction: 一个 4D 张量，形状为 (*FEATURE_MAP_Px, 3, 85)。
            FEATURE_MAP_Px 是 p5, p4, p3 特征图大小。3 是特征图每个位置上，预设框
            的数量。85 是单个预测结果的长度。
            长度为 85 的预测向量，第 0 为表示是否有物体的概率， 第 1 位到第 80 位，
            是表示物体类别的 one-hot 编码，而最后 4 位，则分别是物体框 bbox 的参数
            (center_x, center_y, height, width)。
    """

    # 下面的注释以 p5 为例。在 p5 的 19x19 个位置上，将每个长度为 255 的向量，转换为
    #  3x85 的形状。
    #  85 位分别表示 [confidence, classification..., tx, ty, th, tw]，其中
    #  classification 部分，共包含 80 位，最后 4 位是探测框的中心点坐标和大小。

    # prediction 的形状为 (N, *batch_size_feature_map, 255), N 为 batch_size。
    # 进行 reshape时，必须带上 batch_size，所以用 batch_size_feature_map
    batch_size_feature_map = prediction.shape[: 3]
    prediction = tf.reshape(prediction,
                            shape=(*batch_size_feature_map, 3, 85))

    # get_probability 形状为 (N, 19, 19, 3, 81)，包括置信度和分类结果两部分。
    get_probability = tf.math.sigmoid(prediction[..., : 81])

    # confidence 形状为 (N, 19, 19, 3, 1) 需要配合使用 from_logits=False
    confidence = get_probability[..., : 1]

    # classification 形状为 (N, 19, 19, 3, 80)，需要配合使用 from_logits=False
    classification = get_probability[..., 1: 81]

    # prediction 的形状为 (N, 19, 19, 3, 85), N 为 batch_size。
    feature_map = prediction.shape[1: 3]

    # 根据 YOLO V3 论文中的 figure 2，需要对 bbox 坐标和尺寸进行转换。tx_ty 等标
    # 注记号和论文的记号一一对应。
    # tx_ty 形状为 (N, 19, 19, 3, 2)，分别代表 tx, ty。
    tx_ty = prediction[..., -4: -2]
    # 根据 YOLO V3论文，需要先取得 cx_cy。cx_cy 实际是一个比例值，在计算 IOU 和损失
    # 值之前，应该转换为 608x608 大小图片中的实际值。
    # 注意，根据论文 2.1 节第一段以及配图 figure 2，cx_cy 其实是每一个 cell
    # 的左上角点，这样预测框的中心点 bx_by 才能达到该 cell 中的每一个位置。
    grid = tf.ones(shape=feature_map)  # 构造一个 19x19 的网格
    cx_cy = tf.where(grid)  # where 函数可以获取张量的索引值，也就是 cx, cy

    cx_cy = tf.cast(x=cx_cy, dtype=tf.float32)  # cx_cy 原本是 int64 类型

    # cx_cy 的形状为 (361, 2)， 361 = 19 x 19，下面将其形状变为 (1, 19, 19, 1, 2)
    cx_cy = tf.reshape(cx_cy, shape=(1, *feature_map, 2))
    cx_cy = cx_cy[..., tf.newaxis, :]  # 展示一下 tf.newaxis 的用法

    #  cx_cy 的形状为 (1, 19, 19, 1, 2), tx_ty 的形状为 (N, 19, 19, 3, 2)
    bx_by = tf.math.sigmoid(tx_ty) + cx_cy

    # 下面根据 th, tw, 计算 bh, bw。th_tw 形状为 (N, 19, 19, 3, 2)
    th_tw = prediction[..., -2:]

    # anchor_boxes 是 Python 的 float64 类型数据，需要转换为 float32 类型才能参与后
    # 续的计算。
    ph_pw = tf.cast(anchor_boxes, dtype=tf.float32)
    # anchor_boxes 的形状为 (3, 2)，和上面的 cx_cy 同理，需要将 ph_pw 的形状变为
    # (1, 1, 3, 2)
    ph_pw = tf.reshape(ph_pw, shape=(1, 1, 1, 3, 2))
    # 此时 ph_pw 和 th_tw 的张量阶数 rank 相同，会自动扩展 broadcast，进行算术运算。
    bh_bw = ph_pw * tf.math.exp(th_tw)

    # 在计算 CIOU 损失时，如果高度宽度过大，计算预测框面积会产生 NaN 值，导致模型无法
    # 训练。所以把预测框的高度宽度限制到不超过图片大小即可。
    bh_bw = tf.clip_by_value(
        bh_bw, clip_value_min=0, clip_value_max=MODEL_IMAGE_SIZE[0])

    # bx_by，bh_bw 为比例值，需要转换为在 608x608 大小图片中的实际值。
    image_scale_height = MODEL_IMAGE_SIZE[0] / feature_map[0]
    image_scale_width = MODEL_IMAGE_SIZE[1] / feature_map[1]
    image_scale = image_scale_height, image_scale_width

    # bx_by 是一个比例值，乘以比例 image_scale 之后，bx_by 将代表图片中实际
    # 的长度数值。比如此时 bx, by 的数值可能是 520， 600 等，数值范围 [0, 608]
    # 而 bh_bw 已经是一个长度值，不需要再乘以比例。
    bx_by *= image_scale

    bx_by = tf.clip_by_value(
        bx_by, clip_value_min=0, clip_value_max=MODEL_IMAGE_SIZE[0])

    transformed_prediction = tf.concat(
        values=[confidence, classification, bx_by, bh_bw], axis=-1)

    return transformed_prediction


def predictor(inputs):
    """对模型输出的 1 个 head 进行转换。

    转换方式为：
    先将 head 的形状从 (batch_size, height, width, 255) 变为 (batch_size, height,
    width, 3, 85)。将每个长度为 85 的预测结果进行转换，第 0 位为置信度 confidence，
    第 1 位到第 81位为分类的 one-hot 编码，均需要用 sigmoid 转换为 [0, 1] 之间的数。
    最后 4 位是探测框的预测结果，需要根据 YOLO V3 论文进行转换。
    倒数第 4 位到倒数第 3 位为预测框的中心点，需要对中心点用 sigmoid 函数进行转换。然
    后乘以一个比例，就得到中心点在模型输入图片中的实际坐标值。
    倒数第 2 位和最后一位是探测框的宽度和高度，需要先用指数函数转换为非负数，再乘以探测
    框的高度和宽度。

    Arguments:
        inputs: 一个元祖，包含来自 Heads 输出的 3个 3D 张量。分别表示为
            p5_head, p4_head, p3_head。使用时 Keras 将会自动插入一个第 0 维度，
            作为批量维度。
    Returns:
        p5_prediction: 一个 5D 张量，形状为 (batch_size, height, width, 3, 85)。
            height, width 是特征图大小。3 是特征图的每个位置上，预设框的数量。
            85 是单个预测结果的长度。下面 p4_prediction， p3_prediction 也是一样。
            p5_prediction 的 height, width 为 19, 19.
        p4_prediction: 一个 5D 张量，形状为 (batch_size, 38, 38, 3, 85)。
        p3_prediction: 一个 5D 张量，形状为 (batch_size, 76, 76, 3, 85)。
    """

    # px_head 代表 p5_head, p4_head, p3_head
    px_head = inputs
    feature_map_size = px_head.shape[1: 3]

    anchor_boxes_px = None
    if feature_map_size == (*FEATURE_MAP_P5,):
        anchor_boxes_px = ANCHOR_BOXES_P5
    elif feature_map_size == (*FEATURE_MAP_P4,):
        anchor_boxes_px = ANCHOR_BOXES_P4
    elif feature_map_size == (*FEATURE_MAP_P3,):
        anchor_boxes_px = ANCHOR_BOXES_P3

    px_prediction = _transform_predictions(px_head, anchor_boxes_px)

    return px_prediction


class CheckModelWeight(keras.callbacks.Callback):
    """在训练过程中，每一个批次训练完之后，检查模型的权重，并给出最大权重，以监视 NaN 的
    发生过程。"""

    def __init__(self):
        super(CheckModelWeight, self).__init__()
        # 训练初期权重快速增长，所以为了避免频繁报告权重数值，设置大于 0.5 的权重才开始报告。
        self.max_weight = 0.5

    # noinspection PyUnusedLocal
    def on_epoch_end(self, epoch, logs=None):
        """检查每个 batch 结束之后，最大权重是否发生变化。主要目的是监视出现极大权重的
        情况。"""

        # self.model.weights 是一个列表，其中的每一个元素都是一个多维张量。
        for weight in self.model.weights:
            check_inf_nan(inputs=weight, name='weight')

            if np.amax(weight) > self.max_weight:
                self.max_weight = tf.math.reduce_max(weight)
                tf.print(f'\nHighest_weight changed to: '
                         f'{self.max_weight:.2e}, at epoch {epoch}. Weight '
                         f'shape:', weight.shape)


def check_weights(model_input):
    """检查每个 batch 结束之后，最大权重是否发生变化。主要目的是监视出现极大权重的
    情况。"""

    red_line_weight = 500
    max_weight = 0

    progress_bar = keras.utils.Progbar(
        len(model_input.weights), width=60, verbose=1,
        interval=0.01, stateful_metrics=None, unit_name='step')

    print(f'\nChecking the weights ...')
    # model_input.weights 是一个列表，其中的每一个元素都是一个多维张量。
    for i, weight in enumerate(model_input.weights):
        progress_bar.update(i)

        if max_weight < np.amax(weight):
            max_weight = np.amax(weight)

    if max_weight > red_line_weight:
        print(f'\nAlert! max_weight is: {max_weight:.1f}')
        print('\nVery high weight could lead to a big model output '
              'value, then cause the NaN loss. Please consider:\n'
              '1. use a smaller learning_rate;\n2. reduce the loss value.\n')
    else:
        print(f'\nThe status is OK, max_weight is: {max_weight:.1f}\n')

    return max_weight


def iou_calculator(label_bbox, prediction_bbox):
    """计算预测框和真实框的 IoU 。

    用法说明：使用时，要求输入的 label_bbox, prediction_bbox 形状相同，均为 4D 张量。将
    在两个输入的一一对应的位置上，计算 IoU。
    举例来说，假如两个输入的形状都是 (19, 19, 3, 4)，而标签 label_bbox 只在 (8, 12, 0)
    位置有一个物体框，则 iou_calculator 将会寻找 prediction_bbox 在同样位置
    (8, 12, 0) 的物体框，并计算这两个物体框之间的 IoU。prediction_bbox 中其它位置的物
    体框，并不会和 label_bbox 中 (8, 12, 0) 位置的物体框计算 IoU。
    计算结果的形状为 (19, 19, 3)，并且将在 (8, 12, 0) 位置有一个 IoU 值。

    Arguments:
        label_bbox: 一个 4D 张量，形状为 (input_height, input_width, 3, 4)，代表标
            签中的物体框。
            最后一个维度的 4 个值分别代表物体框的 (center_x, center_y, height_bbox,
            width_bbox)。第 2 个维度的 3 表示有 3 种不同宽高比的物体框。
            该 4 个值必须是实际值，而不是比例值。
        prediction_bbox: 一个 4D 张量，形状为 (input_height, input_width, 3, 4)，
            代表预测结果中的物体框。最后一个维度的 4 个值分别代表物体框的
            (center_x, center_y, height_bbox, width_bbox)。第 2 个维度的 3 表示
            有 3 种不同宽高比的物体框。该 4 个值必须是实际值，而不是比例值。
    Returns:
        iou: 一个 3D 张量，形状为 (input_height, input_width, 3)，代表交并比 IoU。
    """

    # 两个矩形框 a 和 b 相交时，要同时满足的 4 个条件是：
    # left_edge_a < right_edge_b , right_edge_a > left_edge_b
    # top_edge_a < bottom_edge_b , bottom_edge_a > top_edge_b

    # 对每个 bbox，先求出 4 条边。left_edge，right_edge 形状为
    # (input_height, input_width, 3)
    label_left_edge = label_bbox[..., -4] - label_bbox[..., -1] / 2
    label_right_edge = label_bbox[..., -4] + label_bbox[..., -1] / 2

    prediction_left_edge = (prediction_bbox[..., -4] -
                            prediction_bbox[..., -1] / 2)
    prediction_right_edge = (prediction_bbox[..., -4] +
                             prediction_bbox[..., -1] / 2)

    label_top_edge = label_bbox[..., -3] - label_bbox[..., -2] / 2
    label_bottom_edge = label_bbox[..., -3] + label_bbox[..., -2] / 2

    prediction_top_edge = (prediction_bbox[..., -3] -
                           prediction_bbox[..., -2] / 2)
    prediction_bottom_edge = (prediction_bbox[..., -3] +
                              prediction_bbox[..., -2] / 2)

    # left_right_condition 的形状为 (input_height, input_width, 3)
    # 表示 2 个条件：left_edge_a < right_edge_b , right_edge_a > left_edge_b
    left_right_condition = tf.math.logical_and(
        x=(label_left_edge < prediction_right_edge),
        y=(label_right_edge > prediction_left_edge))
    # top_bottom_condition 的形状为 (input_height, input_width, 3)
    # 表示 2 个条件：top_edge_a < bottom_edge_b , bottom_edge_a > top_edge_b
    top_bottom_condition = tf.math.logical_and(
        x=(label_top_edge < prediction_bottom_edge),
        y=(label_bottom_edge > prediction_top_edge))

    # intersection_condition 的形状为
    # (input_height, input_width, 3)，是 4 个条件的总和
    intersection_condition = tf.math.logical_and(x=left_right_condition,
                                                 y=top_bottom_condition)
    # 形状扩展为 (input_height, input_width, 3, 1)
    intersection_condition = tf.expand_dims(intersection_condition, axis=-1)
    # 形状扩展为 (input_height, input_width, 3, 4)
    intersection_condition = tf.repeat(input=intersection_condition,
                                       repeats=4, axis=-1)

    # horizontal_edges, vertical_edges 的形状为
    # (input_height, input_width, 3, 4)
    horizontal_edges = tf.stack(
        values=[label_top_edge, label_bottom_edge,
                prediction_top_edge, prediction_bottom_edge], axis=-1)

    vertical_edges = tf.stack(
        values=[label_left_edge, label_right_edge,
                prediction_left_edge, prediction_right_edge], axis=-1)

    zero_pad_edges = tf.zeros_like(input=horizontal_edges)
    # 下面使用 tf.where，可以使得 horizontal_edges 和 vertical_edges 的形状保持为
    # (input_height, input_width, 3, 4)，并且只保留相交 bbox 的边长值，其它设为 0
    horizontal_edges = tf.where(condition=intersection_condition,
                                x=horizontal_edges, y=zero_pad_edges)
    vertical_edges = tf.where(condition=intersection_condition,
                              x=vertical_edges, y=zero_pad_edges)

    horizontal_edges = tf.sort(values=horizontal_edges, axis=-1)
    vertical_edges = tf.sort(values=vertical_edges, axis=-1)

    # 4 条边按照从小到大的顺序排列后，就可以把第二大的减去第三大的边，得到边长。
    # intersection_height, intersection_width 的形状为
    # (input_height, input_width, 3)
    intersection_height = horizontal_edges[..., -2] - horizontal_edges[..., -3]
    intersection_width = vertical_edges[..., -2] - vertical_edges[..., -3]

    # intersection_area 的形状为 (input_height, input_width, 3)
    intersection_area = intersection_height * intersection_width

    prediction_bbox_width = prediction_bbox[..., -1]
    prediction_bbox_height = prediction_bbox[..., -2]

    # 不能使用混合精度计算。因为 float16 格式下，数值达到 65520 时，就会溢出变为 inf，
    # 从而导致 NaN。而 prediction_bbox_area 的数值是可能达到 320*320 甚至更大的。
    prediction_bbox_area = prediction_bbox_width * prediction_bbox_height

    label_bbox_area = label_bbox[..., -1] * label_bbox[..., -2]

    # union_area 的形状为 (input_height, input_width, 3)
    union_area = prediction_bbox_area + label_bbox_area - intersection_area

    # 为了计算的稳定性，避免出现 nan、inf 的情况，分母可能为 0 时应加上一个极小量 EPSILON
    # iou 的形状为 (input_height, input_width, 3)
    iou = intersection_area / (union_area + EPSILON)

    return iou


def diagonal_calculator(label_bbox, prediction_bbox):
    """计算预测框和真实框最小包络 smallest enclosing box 的对角线长度。

    要计算最小包络的对角线长度，需要先计算最小包络，4 个步骤如下：
    1. 2 个矩形框，共有 4 个水平边缘，按其纵坐标从小到大排列。
    2. 把另外 4 个竖直边缘，按横坐标从小到大排列。
    3. 将水平边缘的最小值和最大值，加上竖直边缘的最小值和最大值，就形成了最小包络。
    4. 根据最小包络的高度和宽度，计算对角线长度。

    Arguments:
        label_bbox: 一个 4D 张量，形状为 (height, width, 3, 4)，代表标签中的物体框。
        prediction_bbox: 一个 4D 张量，形状为 (height, width, 3, 4)，代表预测结果中
            的物体框。
    Returns:
        diagonal_length: 一个 3D 张量，形状为 (height, width, 3)，代表预测框和真实框
            最小包络的对角线长度。
    """

    # 对每个 bbox，4个参数分别是 (center_x, center_y, height_bbox, width_bbox)

    # 对每个 bbox，先求出 4 条边。left_edge，right_edge 形状为 (height, width, 3)
    label_left_edge = label_bbox[..., -4] - label_bbox[..., -1] / 2
    label_right_edge = label_bbox[..., -4] + label_bbox[..., -1] / 2

    prediction_left_edge = (prediction_bbox[..., -4] -
                            prediction_bbox[..., -1] / 2)
    prediction_right_edge = (prediction_bbox[..., -4] +
                             prediction_bbox[..., -1] / 2)

    label_top_edge = label_bbox[..., -3] - label_bbox[..., -2] / 2
    label_bottom_edge = label_bbox[..., -3] + label_bbox[..., -2] / 2

    prediction_top_edge = (prediction_bbox[..., -3] -
                           prediction_bbox[..., -2] / 2)
    prediction_bottom_edge = (prediction_bbox[..., -3] +
                              prediction_bbox[..., -2] / 2)

    # horizontal_edges, vertical_edges 的形状为 (height, width, 3, 4)
    horizontal_edges = tf.stack(
        values=[label_top_edge, label_bottom_edge,
                prediction_top_edge, prediction_bottom_edge], axis=-1)
    vertical_edges = tf.stack(
        values=[label_left_edge, label_right_edge,
                prediction_left_edge, prediction_right_edge], axis=-1)

    horizontal_edges = tf.sort(values=horizontal_edges, axis=-1)
    vertical_edges = tf.sort(values=vertical_edges, axis=-1)

    # 将水平边缘的最大值减去水平边缘的最小值(这里的最小，是指其坐标值最小)，就得到最小包络的
    # 高度 height_enclosing_box。 height_enclosing_box 的形状为 (height, width, 3)
    height_enclosing_box = horizontal_edges[..., -1] - horizontal_edges[..., 0]

    # 将竖直边缘的最大值减去竖直边缘的最小值(指其坐标值最大)，就得到最小包络的
    # 宽度 width_enclosing_box。 width_enclosing_box 的形状为 (height, width, 3)
    width_enclosing_box = vertical_edges[..., -1] - vertical_edges[..., 0]

    # 将水平边缘和竖直边缘进行 stack 组合，就得到 height_width_enclosing_box。
    # height_width_enclosing_box 的形状为 (height, width, 3, 2)
    height_width_enclosing_box = tf.stack(
        values=[height_enclosing_box, width_enclosing_box], axis=-1)

    # 计算欧氏距离，得到对角线长度 diagonal_length， 其形状为 (height, width, 3)
    diagonal_length = tf.math.reduce_euclidean_norm(
        input_tensor=height_width_enclosing_box, axis=-1)

    return diagonal_length


def ciou_calculator(label_bbox, prediction_bbox, get_diou=None):
    """计算预测框和真实框的 CIOU 损失。

    CIOU loss： loss_ciou = 1 − IoU + r_ciou  https://arxiv.org/abs/1911.08287
    CIOU 的正则项 regularization: r_ciou = r_diou + α * v

    Arguments:
        label_bbox: 一个 4D 张量，形状为 (height, width, 3, 4)，代表标签中的物体框。
        prediction_bbox: 一个 4D 张量，形状为 (height, width, 3, 4)，代表预测结果中
            的物体框。
        get_diou: 一个布尔值，如果为 True，则返回 diou 的值，在进行 DIOU-NMS 时会用到。
    Returns:
        loss_ciou: 一个 3D 张量，形状为 (height, width, 3)，代表 CIOU 损失。
    """

    iou = iou_calculator(label_bbox=label_bbox,
                         prediction_bbox=prediction_bbox)

    # 对每个 bbox，4 个参数分别是 (center_x, center_y, height_bbox, width_bbox)
    label_center = label_bbox[..., : 2]
    prediction_center = prediction_bbox[..., : 2]

    # deltas_x_y 的形状为 (height, width, 3, 2)，代表 2 个 bbox 之间中心点的 x，y差值
    deltas_x_y = label_center - prediction_center
    # 根据论文，用 rho 代表 2 个 bbox 中心点的欧氏距离，形状为 (height, width, 3)
    rho = tf.math.reduce_euclidean_norm(input_tensor=deltas_x_y, axis=-1)

    # c_diagonal_length 的形状为 (height, width, 3)
    c_diagonal_length = diagonal_calculator(label_bbox, prediction_bbox)

    # 为了计算的稳定性，避免出现 nan、inf 的情况，分母可能为 0 时应加上一个极小量 EPSILON
    # 根据论文中的公式 6 得到 r_diou，r_diou 的形状为 (height, width, 3)
    r_diou = tf.math.square(rho / (c_diagonal_length + EPSILON))

    # 因为论文中的 v 是一个控制宽高比的参数，所以这里将其命名为 v_aspect_ratio
    # 下面根据论文中的公式 9 计算参数 v_aspect_ratio，
    # atan_label_aspect_ratio 的形状为 (height, width, 3)

    atan_label_aspect_ratio = tf.math.atan(
        label_bbox[..., -1] / (label_bbox[..., -2] + EPSILON))
    atan_prediction_aspect_ratio = tf.math.atan(
        prediction_bbox[..., -1] / (prediction_bbox[..., -2] + EPSILON))

    squared_pi = tf.square(np.pi)
    # 把 squared_pi 转换为混合精度需要的数据类型。
    squared_pi = tf.cast(squared_pi, dtype=tf.float32)

    # v_aspect_ratio 的形状为 (height, width, 3)
    v_aspect_ratio = tf.math.square(
        atan_label_aspect_ratio -
        atan_prediction_aspect_ratio) * 4 / squared_pi

    # 根据论文中的公式 11 得到 alpha， alpha 的形状为 (height, width, 3)
    alpha = v_aspect_ratio / ((1 - iou) + v_aspect_ratio + EPSILON)

    # 根据论文中的公式 8 得到 r_ciou，r_ciou 的形状为 (height, width, 3)
    r_ciou = r_diou + alpha * v_aspect_ratio

    # 根据论文中的公式 10 得到 loss_ciou，loss_ciou 的形状为 (height, width, 3)
    loss_ciou = 1 - iou + r_ciou

    if get_diou:
        diou = iou - r_diou
        return diou

    return loss_ciou


def get_objectness_ignore_mask(y_true, y_pred):
    """根据 IoU，生成对应的 objectness_mask。

    根据 YOLO-V3 论文，当预测结果的物体框和标签的物体框 IoU 大于阈值 0.5 时，则可以
    忽略预测结果物体框预测损失，也就是不计算预测结果的 objectness 损失。

    Arguments:
        y_true: 一个数据类型为 float32 的 5D 张量，代表标签中的物体框。
            计算不同的 head 损失值时，形状不同。计算 p5_true, p4_true, p3_true 时，
            形状分别为 (batch_size, 19, 19, 3, 85)，(batch_size, 38, 38, 3, 85)，
            (batch_size, 76, 76, 3, 85)。
        y_pred: 一个数据类型为 float32 的 5D 张量，代表标签中的物体框。
            计算不同的 head 损失值时，形状不同。计算 p5_prediction, p4_prediction,
            p3_prediction 时，形状分别为 (batch_size, 19, 19, 3, 85)，
            (batch_size, 38, 38, 3, 85)，(batch_size, 76, 76, 3, 85)。

    Returns:
        objectness_mask: 一个形状为 (batch_size, 19, 19, 3) 的 4D 布尔张量（该形状以
            P5 特征层为例），代表可以忽略预测损失 objectness 的物体框。该张量是一个基本全
            为 False 的张量，但是对 IoU 大于阈值 0.5 的物体框，其对应的布尔值为 True。
            此外，对于有物体的预设框，它们的损失值不可以被忽略，所以这些有物体的预设框，它
            们的布尔值会被设置为 False。
    """

    batch_size = y_true.shape[0]

    # 初始的 objectness_mask  形状为 (0, 19, 19, 3)。
    objectness_mask = tf.zeros(shape=(0, *y_true.shape[1: 4]), dtype=tf.bool)

    # 1. 遍历每一张图片和标签。
    for i in range(batch_size):
        # one_label, one_prediction 形状为 (19, 19, 3, 85)。
        one_label = y_true[i]
        one_prediction = y_pred[i]

        # 以 P5 特征层为例，objectness_label 形状为 (19, 19, 3)。
        objectness_label = one_label[..., 0]

        # object_exist_label 形状为 (19, 19, 3)，表示标签中有物体的那些预设框。
        object_exist_label = tf.experimental.numpy.isclose(objectness_label, 1)

        # bboxes_indices_label 形状为 (x, 3)，表示标签中有 x 个预设框，其中有物体。
        bboxes_indices_label = tf.where(object_exist_label)

        # 2. 如果没有标签 bbox，创建全为 False 的布尔张量 objectness_mask_one_image。
        # objectness_mask_one_image 形状为 (19, 19, 3)，是全为 False 的布尔张量。
        objectness_mask_one_image = tf.zeros(
            shape=one_label.shape[: 3], dtype=tf.bool)

        # 3. 如果有标签 bbox，对每一个标签 bbox，计算预测结果中的 IoU。以 IoU 大于阈
        # 值 0.5 为判断条件，得到布尔张量 objectness_mask_one_image。
        if len(bboxes_indices_label) > 0:

            # bboxes_prediction 形状为 (19, 19, 3, 4)，是预测结果中 bboxes 的中心点
            # 坐标和高度宽度信息。
            bboxes_prediction = one_prediction[..., -4:]

            # 3.1 遍历每一个标签 bbox，计算预测结果中的 IoU。
            # indices 形状为 (3,)，是标签中一个 bbox 的索引值。
            for indices in bboxes_indices_label:
                # one_label_info 形状为 (85,)。indices 是标签中一个 bbox 的索引值，
                # 在图模式下，无法将张量转换为元祖，用下面这行代码来代替。
                one_label_info = one_label[indices[0], indices[1], indices[2]]
                # one_label_bbox 形状为 (4,)。
                one_label_bbox = one_label_info[-4:]

                # bbox_label 形状为 (19, 19, 3, 4)，张量中每一个长度为 (4,)的向量，
                # 都是 one_bbox_label 的中心点坐标和高度宽度信息。
                bbox_label = tf.ones_like(bboxes_prediction) * one_label_bbox

                # iou_result 形状为 (19, 19, 3)，是预测结果 bboxes 的 IoU。
                iou_result = iou_calculator(label_bbox=bbox_label,
                                            prediction_bbox=bboxes_prediction)

                # objectness_mask_one_label_bbox 形状为 (19, 19, 3)。
                objectness_mask_one_label_bbox = (iou_result > 0.5)

                # 3.2 对每个标签 bbox 得到的 objectness_mask_one_label_bbox，使用
                # 逻辑或操作，得到最终的 objectness_mask_one_image。
                # objectness_mask_one_image 形状为 (19, 19, 3)。
                objectness_mask_one_image = tf.math.logical_or(
                    objectness_mask_one_image, objectness_mask_one_label_bbox)

            # 3.3 把所有标签 bboxes 遍历完成之后，对于标签中有物体框的位置，还要把
            # objectness_mask_one_image 的对应位置设为 False。即这些有物体的预设框
            # 损失值不可以被忽略，必须计算二元交叉熵损失。

            # objectness_mask_one_image 形状为 (19, 19, 3)。
            objectness_mask_one_image = tf.where(
                condition=object_exist_label,
                x=False, y=objectness_mask_one_image)

        # 4. 将 batch_size 个 objectness_mask_one_image 进行 concatenate，得到最终
        # 的 objectness_mask。
        # objectness_mask_one_image 形状为 (1, 19, 19, 3)。
        objectness_mask_one_image = objectness_mask_one_image[tf.newaxis, ...]

        # objectness_mask 最终的形状为 (batch_size, 19, 19, 3)。
        objectness_mask = tf.concat(
            values=[objectness_mask, objectness_mask_one_image], axis=0)

    return objectness_mask


def yolo_v4_loss(y_true, y_pred, use_predictor=True):
    """YOLO-V4-CSP 的原始损失函数。因为有 3 个 heads，会自动分 3 次进行计算。每一次
    计算损失，y_true 和 y_pred 的形状均为 (batch_size, *FEATURE_MAP, 3, 85)。

    Arguments:
        y_true: 一个数据类型为 float32 的 5D 张量，代表标签中的物体框。
            计算不同的 head 损失值时，形状不同。计算 p5_true, p4_true, p3_true 时，
            形状分别为 (batch_size, 19, 19, 3, 85)，(batch_size, 38, 38, 3, 85)，
            (batch_size, 76, 76, 3, 85)。
        y_pred: 一个数据类型为 float32 的 5D 张量，代表标签中的物体框。
            计算不同的 head 损失值时，形状不同。计算 p5_prediction, p4_prediction,
            p3_prediction 时，形状分别为 (batch_size, 19, 19, 3, 85)，
            (batch_size, 38, 38, 3, 85)，(batch_size, 76, 76, 3, 85)。
        use_predictor: 一个布尔值。如果为 True，将使用 predictor 函数对预测结果进行转换。
    Returns:
        total_loss: 一个浮点数，代表该批次的平均损失值，等于总的损失值除以批次大小。
    """

    # 经过 predictor 转换，将 y_pred 的形状从 (batch_size, *FEATURE_MAP_Px, 255)
    # 改为 (batch_size, *FEATURE_MAP_Px, 3, 85)，并且进行了概率值和 bbox 参数的转换。
    if use_predictor:
        y_pred = predictor(inputs=y_pred)

    # YOLO V3 论文中使用了二元交叉熵来计算概率损失和分类损失，即把该任务作为多标签
    # multi-label 型任务。reduction 方式设定为暂不求和，需要根据标签中的第 0 位
    # probability 来求和，如果 probability==0，表示无物体，则无须计算分类和探测框损失。
    # 当输入值没有转换为概率时，使用 from_logits=True。如果模型的 prediction 部分，
    # 已经用 sigmoid 函数将数值转换到 [0, 1] 范围，相当于是概率值，所以这里就应该使用
    # 默认设置 from_logits=False。
    # 后续根据需要使用 label_smoothing。
    binary_crossentropy_logits_false = keras.losses.BinaryCrossentropy(
        from_logits=False, label_smoothing=0,
        reduction=keras.losses.Reduction.NONE)

    # 根据每个物体框的 85 位向量，计算 3 部分的损失，然后求损失之和。第一部分损失为
    # 第 0 位的 objectness 损失。

    # 以 p5 为例，objectness_label 的形状为 (batch_size, 19, 19, 3, 1)。
    # objectness 这个单词是基于 YOLO-V3 的原始论文，可以理解为“预设框内存在物体的概率”。
    objectness_label = y_true[..., : 1]
    objectness_prediction = y_pred[..., : 1]

    # loss_objectness 的形状为 (batch_size, 19, 19, 3)
    loss_objectness = binary_crossentropy_logits_false(
        objectness_label, objectness_prediction)

    # objectness_ignore_mask 是一个布尔张量，形状为 (batch_size, 19, 19, 3)，
    # 代表可以忽略 objectness 损失的物体框，张量中为 True 的位置，表示可以忽略其损失。
    objectness_ignore_mask = get_objectness_ignore_mask(y_true=y_true,
                                                        y_pred=y_pred)

    # 对于那些和标签 IoU 大于阈值 0.5 的物体框，忽略其 objectness 损失。
    loss_objectness = tf.where(condition=objectness_ignore_mask,
                               x=0., y=loss_objectness)

    # 求和之后，loss_objectness 是一个标量。
    loss_objectness = tf.math.reduce_sum(loss_objectness)

    # 参与计算 3 种不同的损失的 bbox 数量，各不相同，需要分开计算均值。
    # y_true 形状为 (batch_size, *FEATURE_MAP_P5, 3, 85)。
    prior_bboxes_quantity = np.prod((y_true.shape[: 4]))

    # 在计算均值时，理论上应该除去 objectness_ignore_mask 中那些被忽略的 bboxes 数量，
    # 但是因为这个数量相对 prior_bboxes_quantity 来说很小，所以从影响程度的大小，以及加
    # 快计算速度的角度出发，不考虑那些被忽略的 bboxes 数量。
    loss_probability_mean = loss_objectness / prior_bboxes_quantity

    # label_classification 的形状为 (batch_size, 19, 19, 3, 80)
    label_classification = y_true[..., 1: 81]
    prediction_classification = y_pred[..., 1: 81]

    # 经过 binary_crossentropy 计算后，loss_classification 的形状为
    # (batch_size, 19, 19, 3)
    loss_classification = binary_crossentropy_logits_false(
        label_classification, prediction_classification)

    # object_exist_tensor 是一个布尔张量，形状为 (batch_size, 19, 19, 3)，
    # 用于判断预设框内是否有物体。
    # 因为浮点数没有精度，不能直接比较是否相等，应该用 isclose 函数进行比较。
    object_exist_tensor = tf.experimental.numpy.isclose(y_true[..., 0], 1.0)
    object_exist_tensor = tf.convert_to_tensor(object_exist_tensor)

    # 只有在预设框有物体时，才计算分类损失 loss_classification，所以需要用
    # object_exist_tensor 作为索引。
    loss_classification = tf.math.reduce_sum(
        loss_classification[object_exist_tensor])

    # bounding box 张量的形状为 (batch_size, 19, 19, 3, 4)
    label_bbox = y_true[..., -4:]
    prediction_bbox = y_pred[..., -4:]

    loss_ciou = ciou_calculator(label_bbox=label_bbox,
                                prediction_bbox=prediction_bbox)

    # 返回的 loss_ciou 形状为 (batch_size, 19, 19, 3)，同样只在有物体的探测框，
    # 才计算其损失值，用 object_exist_tensor 进行索引。loss_ciou 是一个标量型张量。
    loss_ciou = tf.math.reduce_sum(loss_ciou[object_exist_tensor])

    # 计算损失的均值，先要统计有多少个样本参与了计算损失值。 existing_objects 是一个张量，
    # 包含若干个元祖，代表了 object_exist_tensor 中为 True 的元素的索引值。
    existing_objects = tf.where(object_exist_tensor)

    # object_exist_boxes_quantity 是一个整数型张量，表示标签中有物体的物体框数量。
    object_exist_boxes_quantity = len(existing_objects)

    # 把 object_exist_boxes_quantity 转换为 float32 型张量。
    object_exist_boxes_quantity = tf.cast(object_exist_boxes_quantity,
                                          dtype=tf.float32)

    # 注意！只有在标签中有物体时，才计算分类损失均值以及 ciou 损失均值。
    # 如果标签中没有物体也计算均值，损失中将产生 NaN 值。
    if object_exist_boxes_quantity > 0:
        loss_classification_mean = (loss_classification /
                                    object_exist_boxes_quantity)

        loss_ciou_mean = loss_ciou / object_exist_boxes_quantity
    else:
        loss_classification_mean = 0.0
        loss_ciou_mean = 0.0

    total_loss = (loss_probability_mean +
                  loss_classification_mean + loss_ciou_mean)

    return total_loss


def focal_categorical_crossentropy(y_true, y_pred, gamma=2, from_logits=False):
    """一个 focal loss 形式的多类别交叉熵损失函数。

    使用说明：
    根据 focal loss 论文的公式 4，该损失函数公式如下：
    focal_loss = -(1 - p) ** gamma * log(p) ，其中， p 为正确类别对应的预测概率。
    focal_loss 会把输入张量的最后一个维度消去。如果输入 y_true 和 y_pred 为 2D 张
    量，则返回的 focal_loss 将是 1D 张量。
    focal loss 论文地址： https://arxiv.org/abs/1708.02002

    Arguments:
        y_true: 一个数据类型为 float32 的张量，代表标签中的物体框。
            该张量的阶数应该大于等于 2，并且最后一个维度表示的是物体类别的 one-hot
            编码。如果是 COCO 数据集的 80 个类别，则 y_true 的张量形状是 (x, 80)。
        y_pred: 一个数据类型为 float32 的张量，代表预测结果中的物体框。
            该张量的阶数应该大于等于 2，并且最后一个维度表示的是物体类别的 one-hot
            编码。如果是 COCO 数据集的 80 个类别，则 y_pred 的张量形状是 (x, 80)。
        gamma: 一个浮点数，是 focal loss 论文公式中的 γ。默认为 2。
        from_logits: 一个布尔值，如果为 True，表示预测结果的数值范围是 [-∞, +∞]，
            需要先用 softmax 函数将其转换为概率值。

    Returns:
        focal_loss: 一个数据类型为 float32 的张量，代表 focal 形式的多类别交叉熵
            损失，形状为 (x,)。
    """

    if from_logits:
        y_pred = tf.nn.softmax(logits=y_pred, axis=-1)

    # label_index 形状为 (x, 80)，是一个布尔张量。
    label_index = tf.experimental.numpy.isclose(y_true, 1)
    # probability_pred 形状为 (x,)。
    probability_pred = y_pred[label_index]
    # modulating_factor 形状为 (x,)。
    modulating_factor = (1 - probability_pred) ** gamma
    # focal_loss 形状为 (x,)。
    focal_loss = - modulating_factor * tf.math.log(probability_pred)

    return focal_loss


def my_custom_loss(y_true, y_pred, focal_binary_loss=True,
                   categorical_classification_loss=False,
                   weight_classification=10, weight_ciou=0.01,
                   use_predictor=True):
    """用于 YOLO-V4-CSP 的自定义损失函数。

    用法说明：因为有 3 个 heads，会自动分 3 次进行计算。每一次计算损失，y_true 和 y_pred
    的形状均为 (batch_size, *FEATURE_MAP, 3, 85)。

    损失函数的 3 个组成部分如下：
    1. objectness 损失：对每个物体框，判断框内是否有物体，使用二元交叉熵损失。
    2. 分类损失：判断框内的物体，属于哪个类别的损失。使用二元交叉熵，或是多类别交叉熵损失函数。
    3. 物体框的位置损失：预测结果框和标签中物体框的位置差别，使用 CIOU 损失。

    Arguments:
        y_true: 一个数据类型为 float32 的 5D 张量，代表标签中的物体框。
            计算不同的 head 损失值时，形状不同。计算 p5_true, p4_true, p3_true 时，
            形状分别为 (batch_size, 19, 19, 3, 85)，(batch_size, 38, 38, 3, 85)，
            (batch_size, 76, 76, 3, 85)。
        y_pred: 一个数据类型为 float32 的 5D 张量，代表标签中的物体框。
            计算不同的 head 损失值时，形状不同。计算 p5_prediction, p4_prediction,
            p3_prediction 时，形状分别为 (batch_size, 19, 19, 3, 85)，
            (batch_size, 38, 38, 3, 85)，(batch_size, 76, 76, 3, 85)。
        focal_binary_loss: 一个布尔值，如果为 True，则使用 focal loss 形式的交叉熵。
        categorical_classification_loss: 一个布尔值，如果为 True，则对损失函数的类别
            损失部分使用多类别损失函数。
        weight_classification: 一个浮点数，是分类类别损失值的权重系数。默认为 1。
        weight_ciou: 一个浮点数，是探测框损失值的权重系数。后续可以根据训练的
            效果，改变 2 个损失值 classification，ciou 之间的相对比例，以达到更好的
            分类效果。objectness 的损失比例默认为 1，所以只需要改 classification
            和 ciou 的系数即可。
        use_predictor: 一个布尔值，如果为 True，将使用 predictor 函数对预测结果进行转换。

    Returns:
        total_loss: 一个浮点数，代表该批次的平均损失值，等于总的损失值除以批次大小。
    """

    # check_inf_nan(inputs=y_pred, name='check 1, before y_pred')
    if use_predictor:
        y_pred = predictor(inputs=y_pred)
    # check_inf_nan(inputs=y_pred, name='check 2, after y_pred')

    # YOLO V3 论文中使用了二元交叉熵来计算概率损失和分类损失，即把该任务作为多标签
    # multi-label 型任务。reduction 方式设定为暂不求和，需要根据标签中的第 0 位
    # probability 来求和，如果 probability==0，表示无物体，则无须计算分类和探测框损失。
    # 当输入值没有转换为概率时，使用 from_logits=True。如果模型的 prediction 部分，
    # 已经用 sigmoid 函数将数值转换到 [0, 1] 范围，相当于是概率值，所以这里就应该使用
    # 默认设置 from_logits=False 。

    if focal_binary_loss:
        # 尝试使用 BinaryFocalCrossentropy。
        binary_crossentropy_logits_false = keras.losses.BinaryFocalCrossentropy(
            from_logits=False, label_smoothing=0,
            gamma=2.0,
            reduction=keras.losses.Reduction.NONE)
    else:
        # 使用 BinaryCrossentropy，后续根据需要使用 label_smoothing。
        binary_crossentropy_logits_false = keras.losses.BinaryCrossentropy(
            from_logits=False, label_smoothing=0,
            reduction=keras.losses.Reduction.NONE)

    # label_objectness 的形状为 (batch_size, 19, 19, 3, 1)。
    label_objectness = y_true[..., 0: 1]
    prediction_objectness = y_pred[..., 0: 1]

    # 经过 binary_crossentropy 计算后，loss_objectness 的形状为
    # (batch_size, 19, 19, 3)，会去掉最后一个维度。
    loss_objectness = binary_crossentropy_logits_false(
        label_objectness, prediction_objectness)

    # loss_ignore_mask 是一个布尔张量，形状为 (batch_size, 19, 19, 3)，张量中为 True
    # 的位置，表示可以忽略其损失，因为它们和某个标签 bbox 的 IoU 大于阈值 0.5。
    loss_ignore_mask = get_objectness_ignore_mask(y_true=y_true, y_pred=y_pred)

    # loss_objectness 的形状为 (batch_size, 19, 19, 3)。
    loss_objectness = tf.where(condition=loss_ignore_mask,
                               x=0., y=loss_objectness)

    # object_exist_tensor 是一个布尔张量，形状为 (batch_size, 19, 19, 3)，
    # 用于判断预设框内是否有物体。如果有物体，则对应的位置为 True。
    # 因为浮点数没有精度，不能直接比较是否相等，应该用 isclose 函数进行比较。
    object_exist_tensor = tf.experimental.numpy.isclose(y_true[..., 0], 1.0)

    # 下面在求损失均值时，应该再除以 3，因为特征图的每个位置，都有 3 个预设框。不过
    # 后续计算损失时，还要乘以系数，所以这里提前进行了简化，不做除以 3 这个计算。
    loss_objectness_mean = tf.reduce_mean(loss_objectness)

    # 计算损失的均值，先要统计有多少个样本参与了计算损失值。 existing_objects 是一个张量，
    # 包含若干个元祖，代表了 object_exist_tensor 中为 True 的元素的索引值。
    existing_objects = tf.where(object_exist_tensor)

    # object_exist_boxes_quantity 是一个整数型张量，表示标签中正样本的数量。
    object_exist_boxes_quantity = len(existing_objects)

    # 只在有物体的探测框，才计算类别损失和 CIOU 损失，否则会产生 NaN 损失。
    if object_exist_boxes_quantity > 0:

        # label_classification 的形状为 (batch_size, 19, 19, 3, 80)。
        label_classification = y_true[..., 1: 81]
        # label_classification 的形状为 (object_exist_boxes_quantity, 80)。
        label_classification = label_classification[object_exist_tensor]

        # prediction_classification 的形状为 (batch_size, 19, 19, 3, 80)。
        prediction_classification = y_pred[..., 1: 81]
        # prediction_classification 的形状为 (object_exist_boxes_quantity, 80)。
        prediction_classification = prediction_classification[
            object_exist_tensor]

        # 经过多类别交叉熵或二元交叉熵计算后，loss_classification 的形状为
        # (object_exist_boxes_quantity,)。
        if categorical_classification_loss:
            # 使用 focal loss 形式的多类别交叉熵。
            loss_classification = focal_categorical_crossentropy(
                y_true=label_classification, y_pred=prediction_classification)
        else:
            loss_classification = binary_crossentropy_logits_false(
                y_true=label_classification, y_pred=prediction_classification)

        loss_classification_mean = tf.reduce_mean(loss_classification)

        # label_bbox 张量的形状为 (batch_size, 19, 19, 3, 4)。
        label_bbox = y_true[..., -4:]
        # label_bbox 的形状为 (object_exist_boxes_quantity, 4)。
        label_bbox = label_bbox[object_exist_tensor]

        # prediction_bbox 张量的形状为 (batch_size, 19, 19, 3, 4)。
        prediction_bbox = y_pred[..., -4:]
        # prediction_bbox 的形状为 (object_exist_boxes_quantity, 4)。
        prediction_bbox = prediction_bbox[object_exist_tensor]

        # loss_ciou 形状为 (object_exist_boxes_quantity,)。
        loss_ciou = ciou_calculator(label_bbox=label_bbox,
                                    prediction_bbox=prediction_bbox)

        loss_ciou_mean = tf.reduce_mean(loss_ciou)

    else:
        loss_classification_mean = 0.
        loss_ciou_mean = 0.

    total_loss = (loss_objectness_mean +
                  loss_classification_mean * weight_classification +
                  loss_ciou_mean * weight_ciou)

    return total_loss


class SaveModelHighestAP(keras.callbacks.Callback):
    """指标 AP 达到最高时，保存模型。

    因为计算 AP 指标花的时间很长，并且 AP 指标的计算图太大，个人 PC 上无法构建计算图，所
    以只在模型运行一定 epochs 次数之后，才计算一次 AP 指标，并且是在 eager 模式下计算。
    而模型本身则始终运行在图模式下，这样既能保证模型运行速度，又能实现 AP 指标的计算。
    具体 5 个操作如下：
    1. 先创建一个专用的模型 evaluation_ap，用于测试指标 AP。
    2. 当训练次数 epochs 满足 2 个条件时，把当前训练模型 self.model 的权重提取出来，可以
        用 get_weights。
        epochs 需要满足的 2 个条件是：
        a. epochs ≥ epochs_warm_up，epochs_warm_up 是一个整数，表示经过一定数量的训
        练之后才保存权重。
        b. epochs % skip_epochs == 0，表示每隔 skip_epochs 个 epochs 之后，
        保存权重。
    3. 把提取到的权重，用 set_weights 加载到指标测试模型 evaluation_ap 上，然后用模型
        evaluation_ap 来测试指标。
    4. 如果指标 AP 大于最好的 AP 记录，则把该模型保存为 highest_ap_model。
    5. 如果提供了路径 ongoing_training_model_path，还可以保存当前正在训练的模型。

    Attributes:
        evaluation_data: 用来计算 AP 的输入数据，一般应该使用验证集数据。
        highest_ap_model_path: 一个字符串，是保存最高 AP 指标模型的路径。
        evaluation_model: 一个 Keras 模型，专门用于计算 AP 指标。
        coco_average_precision: 一个 AP 指标，是类 MeanAveragePrecision 的个体。
        epochs_warm_up: 一个整数，表示从多少个 epochs 训练之后，开始计算 AP 指标。
        skip_epochs: 一个整数，表示每隔多少个 epochs，计算一次 AP 指标。
        ap_record: 一个列表，记录了所有的 AP 指标。
        ongoing_training_model_path: 一个字符串，是每个 epoch 之后保存当前模型的路径。
    """

    def __init__(self, evaluation_data, highest_ap_model_path,
                 epochs_warm_up=100, skip_epochs=50,
                 ongoing_training_model_path=None):
        super().__init__()
        self.evaluation_data = evaluation_data
        self.highest_ap_model_path = highest_ap_model_path

        self.evaluation_model = create_model()
        self.coco_average_precision = MeanAveragePrecision()

        self.epochs_warm_up = epochs_warm_up
        self.skip_epochs = skip_epochs
        # 应该先在 ap_record 中放入一个 0，否则后续从 ap_record 读到的 AP 值为 -inf。
        self.ap_record = [0.]
        self.ongoing_training_model_path = ongoing_training_model_path

    def compile_evaluation_model(self):
        """对模型进行编译。"""

        # 编译模型。没有找到 self.model.loss 的官方介绍，但是根据官方文档的
        # self.model.optimizer，推测损失函数应该是 self.model.loss，经过实际验证可行。
        # 还可以使用 self.model.outputs 和 self.model.metrics 等属性。
        self.evaluation_model.compile(
            run_eagerly=True,
            metrics=[self.coco_average_precision],
            loss=self.model.loss,
            optimizer=self.model.optimizer)

    # noinspection PyUnusedLocal
    def on_train_begin(self, logs=None):
        """在训练开始时，才可以编译模型，因为此时才会有 self.model.loss 等属性。"""
        self.compile_evaluation_model()

    # noinspection PyUnusedLocal
    def on_epoch_end(self, epoch, logs=None):
        """在一个 epoch 结束之后，计算 AP，并保存最高 AP 对应的模型。"""
        # 先保存一个持续训练的模型。
        if self.ongoing_training_model_path is not None:
            self.model.save(self.ongoing_training_model_path)

        # 使用 (epoch + 1) 是因为 epoch 是从 0 开始计算的，但实际上 epoch = 0 时，
        # 就已经是第一次迭代了。
        if tf.logical_and(
            ((epoch + 1) >= self.epochs_warm_up),
                ((epoch + 1 - self.epochs_warm_up) % self.skip_epochs == 0)):
            print(f'\nChecking the AP at epoch {epoch + 1}. The highest AP is: '
                  f'{tf.reduce_max(self.ap_record).numpy():.1%}, ')

            # 先获取当前模型的权重。
            current_weights = self.model.get_weights()
            # 把当前模型的权重，应用到模型 evaluation_ap 上。
            self.evaluation_model.set_weights(current_weights)

            # 因为指标的状态量还存储着上一个模型的状态，所以在计算新模型的指标之前，必须用
            # reset_state() 把指标的状态量全部复位，否则会发生计算错误。
            self.coco_average_precision.reset_state()

            evaluation = self.evaluation_model.evaluate(self.evaluation_data)
            # evaluation 是一个列表，包括 4 个损失值和 3 个 AP。
            current_ap = evaluation[-1]

            if current_ap > np.amax(self.ap_record):
                self.model.save(self.highest_ap_model_path)
                print(f'The highest AP changed to: {current_ap:.1%}, '
                      f'model is saved.')

            # 应该在保存模型之后，才把指标加到列表 ap_record 中去，否则程序逻辑会出错。
            self.ap_record.append(current_ap)


# 指标 MeanAveragePrecision 用到的全局变量，使用大写字母。
# OBJECTNESS_THRESHOLD: 一个浮点数，表示物体框内，是否存在物体的置信度阈值。
OBJECTNESS_THRESHOLD = 0.5
# CLASSIFICATION_CONFIDENCE_THRESHOLD: 一个浮点数，表示物体框的类别置信度阈值。
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.5


# LATEST_RELATED_IMAGES: 一个整数，表示最多使用多少张相关图片来计算一个类别的 AP。
# 验证集有 5000 张图片，平均每个类别有 63 张图片，因此可以使用 63 张以上的图片来计算 AP。
LATEST_RELATED_IMAGES = 10
# BBOXES_PER_IMAGE: 一个整数，表示对于一个类别的每张相关图片，最多使用
# BBOXES_PER_IMAGE 个 bboxes 来计算 AP。
BBOXES_PER_IMAGE = 10

# latest_positive_bboxes: 一个 tf.Variable 张量，用于存放最近的
# LATEST_RELATED_IMAGES 张相关图片，且每张图片只保留 BBOXES_PER_IMAGE 个
# positive bboxes，每个 bboxes 有 2 个数值，分别是类别置信度，以及 IoU 值。
latest_positive_bboxes = tf.Variable(
    tf.zeros(shape=(CLASSES, LATEST_RELATED_IMAGES, BBOXES_PER_IMAGE, 2)),
    trainable=False, name='latest_positive_bboxes')

# labels_quantity_per_image: 一个形状为 (CLASSES, BBOXES_PER_IMAGE) 的整数型
# 张量，表示每张图片中，该类别的标签 bboxes 数量。
labels_quantity_per_image = tf.Variable(
    tf.zeros(shape=(CLASSES, LATEST_RELATED_IMAGES)),
    trainable=False, name='labels_quantity_per_image')

# showed_up_classes：一个形状为 (CLASSES, ) 的布尔张量，用于记录所有出现过的类别。
# 每批次数据中，都会出现不同的类别，计算指标时，只使用出现过的类别进行计算。
showed_up_classes = tf.Variable(tf.zeros(shape=(CLASSES,), dtype=tf.bool),
                                trainable=False, name='showed_up_classes')


class MeanAveragePrecision(tf.keras.metrics.Metric):
    """计算 COCO 的 AP 指标。

    使用说明：COCO 的 AP 指标，是 10 个 IoU 阈值下，80 个类别 AP 的平均值，即 mean
    average precision。为了和单个类别的 AP 进行区分，这里使用 mAP 来代表 AP 的平均值。

    受内存大小的限制，对每一个类别，只使用最近 LATEST_RELATED_IMAGES 张相关图片计算其
    AP(COCO 实际是使用所有相关图片)。
    相关图片是指该图片的标签或是预测结果的正样本中，包含了该类别。对每个类别的每张图片，
    只保留 BBOXES_PER_IMAGE 个 bboxes 来计算 AP（COCO 实际是最多使用 100 个 bboxes）。
    """

    def __init__(self, name='AP', **kwargs):
        super().__init__(name=name, **kwargs)
        self.reset_state()

    # noinspection PyUnusedLocal, PyMethodMayBeStatic
    def update_state(self, y_true, y_pred, sample_weight=None,
                     use_predictor=True):
        """根据每个 batch 的计算结果，区分 4 种情况，更新状态 state。

        Arguments:
            y_true: 一个浮点类型张量，形状为 (batch_size, *Feature_Map_px, 3, 85)。
                是每个批次数据的标签。
            y_pred: 一个浮点类型张量，形状为 (batch_size, *Feature_Map_px, 3, 85)。
                是每个批次数据的预测结果。
            sample_weight: update_state 方法的必备参数，即使不使用该参数，也必须在此
                进行定义，否则程序会报错。
            use_predictor: 一个布尔值。当使用测试盒 testcase 时，在每个单元测试中设置
            use_predictor=False，因为测试盒的 y_pred 是已经转换完成后的结果，不需要用
            predictor 再次转换。
        """

        # 先将模型输出进行转换。y_pred 形状为 (batch_size, *Feature_Map_px, 3, 85)。
        if use_predictor:
            y_pred = predictor(inputs=y_pred)

        # 先更新第一个状态量 showed_up_classes，更新该状态量不需要逐个图片处理。
        # 1. 先从标签中提取所有出现过的类别。
        # objectness_label 形状为 (batch_size, *Feature_Map_px, 3)。
        objectness_label = y_true[..., 0]

        # showed_up_categories_index_label 形状为
        # (batch_size, *Feature_Map_px, 3)，是一个布尔张量。
        showed_up_categories_index_label = tf.experimental.numpy.isclose(
            objectness_label, 1)

        # showed_up_categories_label 形状为 (batch_size, *Feature_Map_px, 3)。
        showed_up_categories_label = tf.argmax(y_true[..., 1: 81], axis=-1)

        # showed_up_categories_label 形状为 (x,)，里面存放的是出现过的类别编号，表示
        # 有 x 个类别出现在了这批标签中。
        showed_up_categories_label = showed_up_categories_label[
            showed_up_categories_index_label]

        # showed_up_categories_label 形状为 (1, x)。
        showed_up_categories_label = tf.reshape(showed_up_categories_label,
                                                shape=(1, -1))

        # 2. 从预测结果中提取所有出现过的类别，操作方法和上面的步骤 1 类似。
        # objectness_pred 形状为 (batch_size, *Feature_Map_px, 3)。
        objectness_pred = y_pred[..., 0]

        # classification_confidence_pred 形状为 (batch_size, *Feature_Map_px, 3)。
        classification_confidence_pred = tf.reduce_max(
            y_pred[..., 1: 81], axis=-1)

        # showed_up_categories_index_pred 形状为
        # (batch_size, *Feature_Map_px, 3)，是一个布尔张量。和 y_true 不同的地方在于，
        # 它需要大于 2 个置信度阈值，才认为是做出了预测，得出正确的布尔张量。
        showed_up_categories_index_pred = tf.logical_and(
            x=(objectness_pred > OBJECTNESS_THRESHOLD),
            y=(classification_confidence_pred >
               CLASSIFICATION_CONFIDENCE_THRESHOLD))

        # showed_up_categories_pred 形状为 (batch_size, *Feature_Map_px, 3)。
        showed_up_categories_pred = tf.argmax(y_pred[..., 1: 81], axis=-1)

        # showed_up_categories_pred 形状为 (y,)，里面存放的是出现过的类别编号，表示
        # 有 y 个类别出现在了这批预测结果中。
        showed_up_categories_pred = showed_up_categories_pred[
            showed_up_categories_index_pred]

        # showed_up_categories_pred 形状为 (1, y)。
        showed_up_categories_pred = tf.reshape(showed_up_categories_pred,
                                               shape=(1, -1))

        # showed_up_categories 形状为 (z,)，是一个 sparse tensor。对出现过的类别求
        # 并集，数量从 x,y 变为 z。
        showed_up_categories = tf.sets.union(showed_up_categories_pred,
                                             showed_up_categories_label)

        # 将 showed_up_categories 从 sparse tensor 转化为 tf.tensor。
        showed_up_categories = showed_up_categories.values

        # 更新状态量 showed_up_classes。
        # 遍历该 batch 中的每一个类别，如果该类别是第一次出现，则需要将其记录下来。
        for showed_up_category in showed_up_categories:
            if not showed_up_classes[showed_up_category]:
                showed_up_classes[showed_up_category].assign(True)

        # 下面更新另外 2 个状态量 latest_positive_bboxes 和
        # labels_quantity_per_image，需要逐个图片处理。
        batch_size = y_true.shape[0]

        # 步骤 1，遍历每一张图片预测结果及其对应的标签。
        for sample in range(batch_size):

            # one_label， one_pred 形状为 (*Feature_Map_px, 3, 85).
            one_label = y_true[sample]
            one_pred = y_pred[sample]

            # 步骤 2.1，对于标签，构造 3 个张量：positives_index_label，
            # positives_label 和 category_label。
            # objectness_one_label 形状为 (*Feature_Map_px, 3).
            objectness_one_label = one_label[..., 0]

            # positives_index_label 形状为 (*Feature_Map_px, 3)，是一个布尔张量。
            # 因为用了 isclose 函数，对标签要慎用 label smoothing，标签值可能不再为 1。
            positives_index_label = tf.experimental.numpy.isclose(
                objectness_one_label, 1)

            # positives_label 形状为 (*Feature_Map_px, 3, 85)，是标签正样本的信息，
            # 在不是正样本的位置，其数值为 -8。
            positives_label = tf.where(
                condition=positives_index_label[..., tf.newaxis],
                x=one_label, y=-8.)

            # category_label 形状为 (*Feature_Map_px, 3)，是标签正样本的类别编号，
            # 在不是正样本的位置，其数值为 0。因为这个 0 会和类别编号 0 发生混淆，所以下面
            # 要用 tf.where 再次进行转换。
            category_label = tf.argmax(positives_label[..., 1: 81], axis=-1,
                                       output_type=tf.dtypes.int32)

            # category_label 形状为 (*Feature_Map_px, 3)，是标签正样本的类别编号，
            # 在不是正样本的位置，其数值为 -8。
            category_label = tf.where(condition=positives_index_label,
                                      x=category_label, y=-8)

            # 步骤 2.2，对于预测结果，构造 3 个张量：positives_index_pred，
            # positives_pred 和 category_pred。

            # objectness_one_pred 形状为 (*Feature_Map_px, 3).
            objectness_one_pred = one_pred[..., 0]
            # classification_confidence_one_pred 形状为 (*Feature_Map_px, 3)。
            classification_confidence_one_pred = tf.reduce_max(
                one_pred[..., 1: 81], axis=-1)

            # positives_index_pred 形状为 (*Feature_Map_px, 3)，是一个布尔张量。
            positives_index_pred = tf.logical_and(
                x=(objectness_one_pred > OBJECTNESS_THRESHOLD),
                y=(classification_confidence_one_pred >
                   CLASSIFICATION_CONFIDENCE_THRESHOLD))

            # positives_pred 形状为 (*Feature_Map_px, 3, 85)，是预测结果正样本的信息，
            # 在不是正样本的位置，其数值为 -8。
            positives_pred = tf.where(
                condition=positives_index_pred[..., tf.newaxis],
                x=one_pred, y=-8.)

            # category_pred 形状为 (*Feature_Map_px, 3)，是预测结果正样本的类别编号，
            # 在不是正样本的位置，其数值为 0。因为这个 0 会和类别编号 0 发生混淆，所以下面
            # 要用 tf.where 再次进行转换。
            category_pred = tf.argmax(positives_pred[..., 1: 81], axis=-1,
                                      output_type=tf.dtypes.int32)

            # category_pred 形状为 (*Feature_Map_px, 3)，是预测结果正样本的类别编号，
            # 在不是正样本的位置，其数值为 -8。
            category_pred = tf.where(condition=positives_index_pred,
                                     x=category_pred, y=-8)

            # 步骤 3，遍历所有 80 个类别，更新另外 2 个状态值。
            # 对于每一个类别，可能会在 y_true, y_pred 中出现，也可能不出现。组合起来
            # 有 4 种情况，需要对这 4 种情况进行区分，更新状态值。
            for category in range(CLASSES):

                # category_bool_label 和 category_bool_pred 形状都为
                # (*Feature_Map_px, 3)，所有属于当前类别的 bboxes，其布尔值为 True。
                # 这也是把 category_label，category_pred 的非正样本位置设为 -8 的原
                # 因，避免和 category 0 发生混淆。
                category_bool_label = tf.experimental.numpy.isclose(
                    category_label, category)
                category_bool_pred = tf.experimental.numpy.isclose(
                    category_pred, category)

                # category_bool_any_label 和 category_bool_any_pred 是单个布尔值，
                # 用于判断 4 种情况。
                category_bool_any_label = tf.reduce_any(category_bool_label)
                category_bool_any_pred = tf.reduce_any(category_bool_pred)

                # 下面要分 4 种情况，更新状态量。
                # 情况 a ：标签和预测结果中，都没有该类别。无须更新状态。

                # 情况 b ：预测结果中没有该类别，但是标签中有该类别。
                # 对于预测结果，要提取置信度和 IoU，且置信度和 IoU 都为 0。
                # 对于标签，则提取该类别的标签数量即可。
                # scenario_b 是单个布尔值。
                scenario_b = tf.logical_and((~category_bool_any_pred),
                                            category_bool_any_label)

                # 情况 c ：预测结果中有该类别，标签没有该类别。
                # 对于预测结果，要提取置信度，而因为没有标签，IoU 为 0。
                # 对于标签，提取该类别的标签数量为 0 即可。
                # scenario_c 是单个布尔值。
                scenario_c = tf.logical_and(category_bool_any_pred,
                                            (~category_bool_any_label))

                # 情况 d ：预测结果和标签中都有该类别，此时要计算 IoU，再提取预测结果的
                # 置信度和 IoU，标签中则要提取标签数量。scenario_d 是单个布尔值。
                scenario_d = tf.logical_and(category_bool_any_pred,
                                            category_bool_any_label)

                # 只有在情况 b, c, d 时，才需要更新状态，所以先要判断是否处在情况
                # b, c, d 下。under_scenarios_bcd 是单个布尔值。
                under_scenarios_bc = tf.logical_or(scenario_b, scenario_c)
                under_scenarios_bcd = tf.logical_or(under_scenarios_bc,
                                                    scenario_d)

                # 在情况 b, c, d 时，更新状态量。
                if under_scenarios_bcd:
                    # 更新第二个状态量 labels_quantity_per_image，其形状为
                    # (CLASSES, latest_related_images)。
                    # one_image_category_labels_quantity 是一个整数，表示在一张图
                    # 片中，属于当前类别的标签 bboxes 数量。
                    one_image_category_labels_quantity = tf.where(
                        category_bool_label).shape[0]

                    # 如果某个类别没有在标签中出现，标签数量会是个 None，需要改为 0 。
                    if one_image_category_labels_quantity is None:
                        one_image_category_labels_quantity = 0

                    # 先把 labels_quantity_per_image 整体后移一位。
                    labels_quantity_per_image[category, 1:].assign(
                        labels_quantity_per_image[category, :-1])

                    # 把最近一个标签数量更新到 labels_quantity_per_image 的第 0 位。
                    labels_quantity_per_image[category, 0].assign(
                        one_image_category_labels_quantity)

                    # 最后更新第三个状态量 latest_positive_bboxes，形状为
                    # (CLASSES, latest_related_images, bboxes_per_image, 2)。
                    # 需要对 3 种情况 b,c,d 分别进行更新。

                    # 情况 b ：预测结果中没有该类别，但是标签中有该类别。
                    # 对于预测结果，要提取置信度和 IoU，且置信度和 IoU 都为 0。
                    if scenario_b:

                        # one_image_positive_bboxes 形状为 (BBOXES_PER_IMAGE, 2)。
                        one_image_positive_bboxes = tf.zeros(
                            shape=(BBOXES_PER_IMAGE, 2))

                    # 情况 c ：预测结果中有该类别，标签没有该类别。
                    # 对于预测结果的状态，要提取置信度，而因为没有标签，IoU 为 0。
                    elif scenario_c:
                        # scenario_c_positives_pred 形状为
                        # (scenario_c_bboxes, 85)。
                        scenario_c_positives_pred = positives_pred[
                            category_bool_pred]

                        # scenario_c_class_confidence_pred 形状为
                        # (scenario_c_bboxes,)。
                        scenario_c_class_confidence_pred = tf.reduce_max(
                                scenario_c_positives_pred[:, 1: 81], axis=-1)

                        scenario_c_bboxes = (
                            scenario_c_class_confidence_pred.shape[0])

                        if scenario_c_bboxes is None:
                            scenario_c_bboxes = 0

                        # 如果 scenario_c_bboxes 数量少于规定的数量，则进行补零。
                        if scenario_c_bboxes < BBOXES_PER_IMAGE:
                            # scenario_c_paddings 形状为 (1, 2)。
                            scenario_c_paddings = tf.constant(
                                (0, (BBOXES_PER_IMAGE - scenario_c_bboxes)),
                                shape=(1, 2))

                            # one_image_positive_bboxes 形状为
                            # (BBOXES_PER_IMAGE,)。
                            one_image_positive_bboxes = tf.pad(
                                tensor=scenario_c_class_confidence_pred,
                                paddings=scenario_c_paddings,
                                mode='CONSTANT', constant_values=0)

                        # 如果 scenario_c_bboxes 数量大于等于规定的数量，则应该先按
                        # 类别置信度从大到小的顺序进行排序，然后保留规定的数量 bboxes。
                        else:
                            # scenario_c_sorted_pred 形状为
                            # (BBOXES_PER_IMAGE,)。
                            scenario_c_sorted_pred = tf.sort(
                                scenario_c_class_confidence_pred,
                                direction='DESCENDING')

                            # one_image_positive_bboxes 形状为
                            # (BBOXES_PER_IMAGE,)。
                            one_image_positive_bboxes = (
                                scenario_c_sorted_pred[: BBOXES_PER_IMAGE])

                        # scenario_c_ious_pred 形状为 (BBOXES_PER_IMAGE,)。
                        scenario_c_ious_pred = tf.zeros_like(
                            one_image_positive_bboxes)

                        # one_image_positive_bboxes 形状为 (BBOXES_PER_IMAGE, 2)。
                        one_image_positive_bboxes = tf.stack(
                            values=[one_image_positive_bboxes,
                                    scenario_c_ious_pred], axis=1)

                    # 情况 d ：预测结果和标签中都有该类别，此时要计算 IoU，再提取预测结果
                    # 的置信度和 IoU，标签中则要提取标签数量。scenario_d 是单个布尔值。
                    else:
                        # 1. bboxes_iou_pred 形状为 (*Feature_Map_px, 3, 4)。
                        bboxes_iou_pred = tf.where(
                            condition=category_bool_pred[..., tf.newaxis],
                            x=positives_pred[..., -4:], y=0.)

                        # 2. 构造 bboxes_category_label， 形状为
                        # (scenario_d_bboxes_label, 4)。
                        bboxes_category_label = positives_label[..., -4:][
                            category_bool_label]

                        # bboxes_area_label 形状为 (scenario_d_bboxes_label,)，
                        # 是当前类别中，各个 bbox 的面积。
                        bboxes_area_label = (bboxes_category_label[:, -1] *
                                             bboxes_category_label[:, -2])

                        # 把标签的 bboxes 按照面积从小到大排序。
                        # sort_by_area 形状为 (scenario_d_bboxes_label,)
                        sort_by_area = tf.argsort(values=bboxes_area_label,
                                                  axis=0, direction='ASCENDING')

                        # 3. 构造 sorted_bboxes_label， 形状为
                        # (scenario_d_bboxes_label, 4)。
                        sorted_bboxes_label = tf.gather(
                            params=bboxes_category_label,
                            indices=sort_by_area, axis=0)

                        # 4. 用 one_image_positive_bboxes 记录下新预测的且命中标签的
                        # bboxes，直接设置其为空，后续用 concat 方式添加新的 bboxes。
                        one_image_positive_bboxes = tf.zeros(
                            shape=(BBOXES_PER_IMAGE, 2))

                        # 用 new_bboxes_quantity 作为标识 flag，每向
                        # one_image_positive_bboxes 增加一个 bbox 信息，则变大 1.
                        new_bboxes_quantity = 0

                        # 5. 遍历 sorted_bboxes_label。
                        for bbox_info in sorted_bboxes_label:

                            # carried_over_shape 形状为 (*Feature_Map_px, 3, 4)
                            carried_over_shape = tf.ones_like(bboxes_iou_pred)

                            # 5.1 构造 bbox_iou_label，
                            # 其形状为 (*Feature_Map_px, 3, 4)。
                            bbox_iou_label = carried_over_shape * bbox_info

                            # 5.2 ious_category 形状为 (*Feature_Map_px, 3)。
                            ious_category = iou_calculator(
                                label_bbox=bbox_iou_label,
                                prediction_bbox=bboxes_iou_pred)

                            # max_iou_category 是一个标量，表示当前类别所有 bboxes，
                            # 计算得到的最大 IoU。
                            max_iou_category = tf.reduce_max(ious_category)

                            # 5.3 当最大 IoU 大于 0.5 时，则认为预测的 bbox 命中了该
                            # 标签，需要把置信度和 IoU 记录到 category_new_bboxes 中。
                            if tf.logical_and(
                                    (max_iou_category > 0.5),
                                    (new_bboxes_quantity < BBOXES_PER_IMAGE)):
                                # 记录 new_bboxes_quantity，当达到设定的固定数量后，
                                # 停止记录新的 bboxes。
                                new_bboxes_quantity += 1

                                # max_iou_position 形状为 (*Feature_Map_px, 3)，
                                # 是一个布尔张量，仅最大 IoU 位置为 True。
                                max_iou_position = (
                                    tf.experimental.numpy.isclose(
                                        ious_category, max_iou_category))

                                # max_iou_bbox_pred 形状为 (1, 85)，是预测结果中
                                # IoU 最大的那个 bbox。
                                max_iou_bbox_pred = positives_pred[
                                    max_iou_position]

                                # max_iou_bbox_confidence 是一个标量型张量。
                                max_iou_bbox_class_confidence = (
                                    tf.reduce_max(max_iou_bbox_pred[0, 1: 81]))

                                # new_bbox 是一个元祖，包含类别置信度和 IoU。
                                new_bbox = (max_iou_bbox_class_confidence,
                                            max_iou_category)

                                # new_bbox 形状为 (1, 2)。
                                new_bbox = tf.ones(shape=(1, 2)) * new_bbox

                                # 记录这个命中标签的 bbox 信息。append_new_bboxes
                                # 形状为 (BBOXES_PER_IMAGE + 1, 2)。
                                append_new_bboxes = tf.concat(
                                    values=[one_image_positive_bboxes,
                                            new_bbox], axis=0)

                                # 5.3.1 记录到 one_image_positive_bboxes， 形状为
                                # (BBOXES_PER_IMAGE, 2)。
                                one_image_positive_bboxes = (
                                    append_new_bboxes[-BBOXES_PER_IMAGE:])

                                # 5.3.2 需要将该 bbox 从 bboxes_iou_pred
                                # 中移除，再进行后续的 IoU 计算。remove_max_iou_bbox
                                # 形状为 (*Feature_Map_px, 3, 1)，在最大 IoU 的位
                                # 置为 True，其它为 False。
                                remove_max_iou_bbox = max_iou_position[
                                    ..., tf.newaxis]

                                # bboxes_iou_pred 形状为 (*Feature_Map_px,
                                # 3, 4)。把被去除的 bbox 替换为 0。
                                bboxes_iou_pred = tf.where(
                                    condition=remove_max_iou_bbox,
                                    x=0., y=bboxes_iou_pred)

                        # 6. 遍历 sorted_bboxes_label 完成之后，处理
                        # bboxes_iou_pred 中剩余的 bboxes。

                        # left_bboxes_sum 形状为 (*Feature_Map_px, 3)，是剩
                        # 余的没有命中标签的 bboxes。
                        # 下面用求和，是为了确定该 bbox 中是否有物体。如果一个
                        # bbox 中没有物体，那么它的中心点坐标，高度宽度这 4 个
                        # 参数之和依然等于 0。
                        left_bboxes_sum = tf.math.reduce_sum(
                            bboxes_iou_pred, axis=-1)

                        # left_bboxes_bool 形状为 (*Feature_Map_px, 3)，
                        # 是一个布尔张量，剩余 bboxes 位置为 True。
                        left_bboxes_bool = (left_bboxes_sum > 0)

                        # left_bboxes_pred 形状为 (left_bboxes_quantity, 85)。
                        left_bboxes_pred = positives_pred[left_bboxes_bool]

                        # left_bboxes_confidence_pred 是剩余 bboxes 的类别
                        # 置信度，形状为 (left_bboxes_quantity,)。
                        left_bboxes_confidence_pred = tf.reduce_max(
                            left_bboxes_pred[:, 1: 81], axis=-1)

                        # left_bboxes_quantity 是一个标量型张量。
                        left_bboxes_quantity = left_bboxes_pred.shape[0]

                        if left_bboxes_quantity is None:
                            left_bboxes_quantity = 0

                        # 把没有命中标签的正样本 bboxes 也记录下来。
                        if tf.math.logical_and(
                                (left_bboxes_quantity > 0),
                                (new_bboxes_quantity < BBOXES_PER_IMAGE)):

                            # scenario_d_bboxes 是一个标量型张量。
                            scenario_d_bboxes = (new_bboxes_quantity +
                                                 left_bboxes_quantity)

                            # 6.1 scenario_d_bboxes > BBOXES_PER_IMAGE，需
                            # 要对剩余的 bboxes，按类别置信度进行排序。
                            if scenario_d_bboxes > BBOXES_PER_IMAGE:
                                # left_bboxes_sorted_confidence 形状为
                                # (left_bboxes_quantity,)。
                                left_bboxes_sorted_confidence = tf.sort(
                                    left_bboxes_confidence_pred,
                                    direction='DESCENDING')

                                # vacant_seats 是一个整数，表示还有多少个空位，
                                # 可以用于填充剩余的 bboxes。
                                vacant_seats = (
                                    BBOXES_PER_IMAGE - new_bboxes_quantity)

                                # left_bboxes_confidence_pred 形状为
                                # (vacant_seats,)。
                                left_bboxes_confidence_pred = (
                                    left_bboxes_sorted_confidence[
                                        : vacant_seats])

                            # left_bboxes_ious_pred 形状为 (vacant_seats,)，
                            # 或者是 (left_bboxes_quantity,)。
                            left_bboxes_ious_pred = tf.zeros_like(
                                left_bboxes_confidence_pred)

                            # left_positive_bboxes_pred 形状为
                            # (left_bboxes_quantity, 2)。
                            left_positive_bboxes_pred = tf.stack(
                                values=[left_bboxes_confidence_pred,
                                        left_bboxes_ious_pred], axis=1)

                            # 记录剩余 bboxes 信息。append_left_bboxes
                            # 形状为 (BBOXES_PER_IMAGE +
                            # left_bboxes_quantity, 2)。
                            append_left_bboxes = tf.concat(
                                values=[one_image_positive_bboxes,
                                        left_positive_bboxes_pred],
                                axis=0)

                            # one_image_positive_bboxes，形状为
                            # (BBOXES_PER_IMAGE, 2)。
                            one_image_positive_bboxes = (
                                append_left_bboxes[-BBOXES_PER_IMAGE:])

                    # 更新最后一个状态量 latest_positive_bboxes。 形状为 (CLASSES,
                    # LATEST_RELATED_IMAGES, BBOXES_PER_IMAGE, 2)。
                    latest_positive_bboxes[category, 1:].assign(
                        latest_positive_bboxes[category, :-1])

                    # latest_positive_bboxes 形状为 (CLASSES,
                    # LATEST_RELATED_IMAGES, BBOXES_PER_IMAGE, 2)。
                    latest_positive_bboxes[category, 0].assign(
                        one_image_positive_bboxes)

    def result(self):
        """对于当前所有已出现类别，使用状态值 state，计算 mean average precision。"""

        # 只在 P3 时计算 AP，因为 P5, P4 实际上是无效计算，跳开 P5, P4 还可以节省时间。
        # 和 YOLOv4-CSP 等 YOLO 系列模型配合使用时，注意要将 3 个输出分别命名为 p5,
        # p4, p3，它们会自动和 AP 指标连接，得到指标名字 self.name 为 'p3_AP' 等。
        if self.name == 'p3_AP':
            # 不能直接使用 tf.Variable 进行索引，需要将其转换为布尔张量。
            # showed_up_classes 形状为 (CLASSES,)。
            showed_up_classes_tensor = tf.convert_to_tensor(
                showed_up_classes, dtype=tf.bool)

            # average_precision_per_iou 形状为 (10,)。
            average_precision_per_iou = tf.zeros(shape=(10,))
            # 把 10 个不同 IoU 阈值情况下的 AP，放入张量 average_precision_per_iou
            # 中，然后再求均值。
            for iou_threshold in np.linspace(0.5, 0.95, num=10):

                # average_precisions 形状为 (80,)，存放的是每一个类别的 AP。
                average_precisions = tf.zeros(shape=(CLASSES,))
                # 对所有出现过的类别，将其 AP 放入 average_precisions 中，然后再求均值。
                for category in range(CLASSES):

                    # 只使用出现过的类别计算 AP。
                    if showed_up_classes[category]:
                        # 1. 计算 recall_precisions。
                        recall_precisions = tf.ones(shape=(1,))
                        true_positives = tf.constant(0., shape=(1,))
                        false_positives = tf.constant(0., shape=(1,))

                        # 下面按照类别置信度从大到小的顺序，对 bboxes 进行排序。
                        # positive_bboxes_category 形状为
                        # (LATEST_RELATED_IMAGES, BBOXES_PER_IMAGE, 2)
                        positive_bboxes_category = latest_positive_bboxes[
                            category]

                        # positive_bboxes_category 形状为
                        # (LATEST_RELATED_IMAGES * BBOXES_PER_IMAGE, 2)
                        positive_bboxes_category = tf.reshape(
                            positive_bboxes_category, shape=(-1, 2))

                        # confidence_category 形状为
                        # (LATEST_RELATED_IMAGES * BBOXES_PER_IMAGE,)。
                        confidence_category = positive_bboxes_category[:, 0]

                        # sorted_classification_confidence 形状为
                        # (LATEST_RELATED_IMAGES * BBOXES_PER_IMAGE,)。
                        sorted_classification_confidence = tf.argsort(
                            values=confidence_category,
                            axis=0, direction='DESCENDING')

                        # sorted_bboxes_category 形状为
                        # (LATEST_RELATED_IMAGES * BBOXES_PER_IMAGE, 2)。
                        sorted_bboxes_category = tf.gather(
                            params=positive_bboxes_category,
                            indices=sorted_classification_confidence, axis=0)

                        # 一个奇怪的事情是，使用 for bbox in sorted_bboxes_category，
                        # 它将不允许对 recall_precisions 使用 tf.concat。
                        # 下面更新 recall_precisions。
                        for i in range(len(sorted_bboxes_category)):
                            bbox = sorted_bboxes_category[i]
                            # sorted_bboxes_category 中，有一些是空的 bboxes，是既
                            # 没有标签，也没有预测结果。当遇到这些 bboxes 时，说明已经遍
                            # 历完预测结果，此时应跳出循环。空的 bboxes 类别置信度为 0.
                            bbox_classification_confidence = bbox[0]

                            if bbox_classification_confidence > 0:
                                bbox_iou = bbox[1]
                                # 根据当前的 iou_threshold，判断该 bbox 是否命中标签。
                                if bbox_iou > iou_threshold:
                                    true_positives += 1
                                    # 如果增加了一个 recall ，则记录下来。
                                    recall_increased = True
                                else:
                                    false_positives += 1
                                    recall_increased = False

                                # 计算精度 precision。
                                precision = true_positives / (true_positives +
                                                              false_positives)

                                # recall_precisions 形状为 (x,)。如果有新增加了一个
                                # recall，则增加一个新的精度值。反之如果 recall 没有
                                # 增加，则把当前的精度值更新即可。
                                recall_precisions = tf.cond(
                                    pred=recall_increased,
                                    true_fn=lambda: tf.concat(
                                        values=[recall_precisions, precision],
                                        axis=0),
                                    false_fn=lambda: tf.concat(
                                        values=[recall_precisions[:-1],
                                                precision], axis=0))

                        # 2. 计算当前类别的 AP。使用累加多个小梯形面积的方式来计算 AP。

                        # labels_quantity 是当前类别中，所有标签的总数。
                        labels_quantity = tf.math.reduce_sum(
                            labels_quantity_per_image[category])

                        # update_state 方法中区分了 a,b,c,d 共 4 种情况，scenario_d
                        # 属于下面这种，即有预测结果和标签，需要计算 AP 的情况。
                        # 如果有标签，即 labels_quantity > 0，要计算 AP。
                        if labels_quantity > 0:

                            # trapezoid_height 是每一个小梯形的高度。
                            # 注意！！！如果没有标签也计算小梯形高度，trapezoid_height
                            # 将会是 inf，并最终导致 NaN。所以要设置
                            # labels_quantity > 0.
                            trapezoid_height = 1 / labels_quantity

                            # accumulated_edge_length 是每一个小梯形的上下边长总和。
                            # accumulated_edge_length = 0.
                            accumulated_edge_length = tf.constant(
                                0., dtype=tf.float32)

                            # recalls 是总的 recall 数量。因为第 0 位并不是真正的
                            # recall，所以要减去 1.
                            recalls = len(recall_precisions) - 1

                            if recalls == 0:
                                # scenario_b 是有标签但是没有预测结果，包括在这种情况
                                # recalls==0，累计的梯形面积应该等于 0，AP 也将等于0。
                                accumulated_area_trapezoid = tf.constant(
                                    0, dtype=tf.float32)

                            else:
                                for i in range(recalls):
                                    top_edge_length = recall_precisions[i]
                                    bottom_edge_length = recall_precisions[
                                        i + 1]

                                    accumulated_edge_length += (
                                            top_edge_length +
                                            bottom_edge_length)

                                # 计算梯形面积：(上边长 + 下边长) * 梯形高度 / 2 。
                                accumulated_area_trapezoid = (
                                    accumulated_edge_length *
                                    trapezoid_height) / 2

                        # 而如果没有标签，则 average_precision=0。
                        # accumulated_area_trapezoid 就是当前类别的
                        # average_precision。scenario_c 属于这种情况。
                        else:
                            accumulated_area_trapezoid = tf.constant(
                                0, dtype=tf.float32)

                        # 构造索引 category_index，使它指向当前类别。
                        category_index = np.zeros(shape=(CLASSES,))
                        category_index[category] = 1
                        # category_index 形状为 (CLASSES,)。
                        category_index = tf.convert_to_tensor(category_index,
                                                              dtype=tf.bool)

                        # average_precisions 形状为 (CLASSES,)。
                        average_precisions = tf.where(
                            condition=category_index,
                            x=accumulated_area_trapezoid[tf.newaxis],
                            y=average_precisions)

                # 把出现过的类别过滤出来，用于计算 average_precision。
                # average_precision_showed_up_categories 形状为 (x,)，即一共有 x
                # 个类别出现过，需要参与计算 AP。
                average_precision_showed_up_categories = average_precisions[
                    showed_up_classes_tensor]

                # 一种特殊情况是，没有标签，预测结果也没有任何物体，此时不可以计算均值，否
                # 则会得出 NaN。这时直接把均值设置为 0 即可。
                if len(average_precision_showed_up_categories) != 0:
                    # average_precision_over_categories 形状为 (1,)。
                    average_precision_over_categories = tf.math.reduce_mean(
                        average_precision_showed_up_categories, keepdims=True)

                else:
                    # 因为下面要进行拼接操作 concat，所以应该设置形状 (1,)，否则会出错。
                    average_precision_over_categories = tf.constant(0.,
                                                                    shape=(1,))

                # average_precision_per_iou 形状始终保持为 (10,)。
                average_precision_per_iou = tf.concat(
                    values=[average_precision_per_iou[1:],
                            average_precision_over_categories], axis=0)

            mean_average_precision = tf.math.reduce_mean(
                average_precision_per_iou)

        # 虽然跳开了 P5, P4，但是还需要给它们设置一个浮点数，否则会报错。
        else:
            mean_average_precision = 0.

        return mean_average_precision

    # noinspection PyMethodMayBeStatic
    def reset_state(self):
        """每个 epoch 开始时，需要重新把状态初始化。"""
        latest_positive_bboxes.assign(
            tf.zeros_like(latest_positive_bboxes))

        labels_quantity_per_image.assign(
            tf.zeros_like(labels_quantity_per_image))

        showed_up_classes.assign(tf.zeros_like(showed_up_classes))


def diou_nms(predictions, objectness_threshold=OBJECTNESS_THRESHOLD,
             classification_threshold=CLASSIFICATION_CONFIDENCE_THRESHOLD,
             diou_threshold=0.5,
             bboxes_quantity=100, is_prediction=True):
    """对输入的模型预测结果，实施 DIOU Non Max Suppression 操作。如果输入的是标签，则只
    提取 bboxes 信息，不实施 DIOU NMS 操作。

    Arguments:
        predictions: 是一个元祖，包含 3 个 float32 型数组，按先后顺序依次如下，
            p5_prediction，形状为 (batch_size, *FEATURE_MAP_P5, 3, 85)。
            p4_prediction，形状为 (batch_size, *FEATURE_MAP_P4, 3, 85)。
            p3_prediction，形状为 (batch_size, *FEATURE_MAP_P3, 3, 85)。
        objectness_threshold: 一个 [0, 1] 之间的浮点数，是物体框内是否有物体的置信度
            阈值。如果预测结果的置信度小于该值，则对应的探测框不显示。
        classification_threshold: 一个 [0, 1] 之间的浮点数，是对于物体框内的物体，属于
            某个类别的置信度。如果预测结果的置信度小于该值，则对应的探测框不显示。
        diou_threshold: 一个 [0, 1] 之间的浮点数，是计算 diou 时使用的阈值。如果 2 个
            bboxes 的 DIOU 小于该阈值，则认为这两个 bboxes 对应的是两个不同的物体。
        bboxes_quantity: 一个整数，表示对于每张图片，只显示 bboxes_quantity 个 bboxes。
        is_prediction: 一个布尔值，如果是模型的预测结果，该布尔值为 True。
            如果不是预测结果而是标签，则该布尔值为 False。

    Returns:
        bboxes_batch: 一个 float32 型数组，代表所有保留下来的 bboxes，形状为
            (batch_size, BBOXES_PER_IMAGE, 6)。其中每个长度为 6 的向量，第 0 为类别
            的 id，第 1 位是类别置信度，后四位是 bbox 的参数，分别是
            (center_x, center_y, height_bbox, width_bbox)。
    """

    batch_size = len(predictions[0])
    # 创建一个 bboxes_batch，把各个图片对应的 bboxes 放入其中。
    bboxes_batch = np.zeros(shape=(batch_size, bboxes_quantity, 6),
                            dtype=np.float32)

    for i in range(batch_size):
        # 1. 找出预测结果中，物体存在置信度大于阈值 0.5 的物体框。可以把 3 个特征层结果
        # concatenate，得到形状为 (n, 85) 的数组。

        # 建立 object_exist_bboxes，把 prediction 中所有的 bboxes 都放入该数组中.
        object_exist_bboxes = None
        for prediction in predictions:
            # objectness_pred 是预测结果中，每个 bbox 是否有物体的概率，
            # 形状为 (*FEATURE_MAP_Px, 3)。因为使用了索引 [i]，消去了批次维度。
            objectness_pred = prediction[i][..., 0]

            # objectness_mask 是布尔数组，用于把物体存在置信度大于阈值 0.5 的物体框
            # 选出来。 objectness_mask 形状为 (*FEATURE_MAP_Px, 3)。
            objectness_mask = objectness_pred > objectness_threshold

            # classification_pred 形状为 (*FEATURE_MAP_Px, 3, 80)。
            classification_pred = prediction[i][..., 1: 81]

            # classification_pred 形状为 (*FEATURE_MAP_Px, 3)，是类别概率。
            classification_pred = np.amax(classification_pred, axis=-1)

            # classification_pred 形状为 (*FEATURE_MAP_Px, 3)。
            classification_mask = classification_pred > classification_threshold

            positive_bboxes_mask = np.logical_and(objectness_mask,
                                                  classification_mask)

            # 如果存在有物体框，则加入到 object_exist_bboxes 中。
            if np.any(positive_bboxes_mask):
                # positive_bboxes 是两个置信度都大于阈值的物体框，形状为 (x, 85)
                positive_bboxes = prediction[i][positive_bboxes_mask]

                if object_exist_bboxes is None:
                    object_exist_bboxes = positive_bboxes
                else:
                    # object_exist_bboxes 包括了所有置信度大于阈值的物体框，形状为
                    # (n, 85)
                    object_exist_bboxes = np.concatenate(
                        (object_exist_bboxes, positive_bboxes), axis=0)

        # object_exist_bboxes 中有 bboxes 时，需要将其形状从 (n, 85) 转换为  (n, 6)。
        if object_exist_bboxes is not None:
            # 建立 kept_bboxes_array，每一张图片的所有 bboxes， 都将用 DIOU NMS 过滤，
            # 然后保留在该数组中。
            kept_bboxes_array = None

            # 2. 根据第 1 位到第 81 位 one-hot 编码，使用 argmax 找出类别置信度最高的位，
            # 得出具体的类别 id；使用 amax 找出该类别对应的类别置信度。

            # classification 是 n 个 bbox 的类别 one-hot 编码， 形状为 (n, 80)
            classification = object_exist_bboxes[:, 1: 81]

            # class_id 是 n 个 bbox 各自的类别 id， 形状为 (n)。老版本 Numpy 没有
            # keepdims 参数
            class_id = np.argmax(classification, axis=1)
            # class_id 数据类型为 int64，需要将其转换为 float32
            class_id = class_id.astype(np.float32)

            # 取得类别 id 的集合 categories，在 class_id 还是一维数组时才可以执行 set
            # 函数。categories 的长度为 m，表示这一批 prediction 中包含有 m 个类别
            categories = set(class_id)

            # class_id 形状为 (n, 1)
            class_id = class_id.reshape(-1, 1)

            # class_confidence 是 n 个 bbox 各自的类别置信度， 形状为 (n, 1)
            class_confidence = np.amax(classification,
                                       axis=1, keepdims=True)

            # bboxes_parameters 是 n 个 bbox 的位置和大小， 形状为 (n, 4)
            bboxes_parameters = object_exist_bboxes[:, 81:]

            # 3. 拼接得到数组 id_confidence_bboxes， 包含了 n 个 bbox 的类别 id，
            # 置信度和 bbox 信息，id_confidence_bboxes 形状为 (n, 6)
            id_confidence_bboxes = np.concatenate(
                (class_id, class_confidence, bboxes_parameters), axis=1)

            # 4. 取得类别 id 的集合 categories，已经在前面提前执行。

            # 需要区分 2 种情况，即输入是模型预测结果还是标签，对这两种不同的输入，处理方法
            # 不同。如果是模型预测结果，需要经过 diou 计算。
            if is_prediction:

                # 5. 遍历 categories， each_class 是一个 [0, 79] 之间的整数，代表一
                # 个类别 id。
                for each_class in categories:

                    # 5.1 each_class_mask 是布尔数组， 形状为 (n, 1)，当前类别的
                    # bboxes 将为 True
                    each_class_mask = np.isclose(class_id, each_class)
                    # each_class_mask 形状为 (n)，必须指定 axis=-1，否则 (1, 1) 的
                    # 数组会变成一个标量。
                    each_class_mask = np.squeeze(each_class_mask, axis=-1)

                    # each_class_unsorted 形状为 (classes, 6)，表示在这批
                    # predictions 中，有 classes 个物体属于这个类别
                    each_class_unsorted = id_confidence_bboxes[
                        each_class_mask, :]

                    # each_class_unsorted_confidence 形状为 (classes)，是各个 bbox
                    # 的类别置信度
                    each_class_unsorted_confidence = each_class_unsorted[:, 1]

                    # 按照类别置信度，进行排序。argsort 默认排序是由小到大。
                    # each_class_sorted_confidence 形状为 (classes)
                    each_class_sorted_confidence = np.argsort(
                        each_class_unsorted_confidence, axis=-1)
                    # 进行由大到小的排序
                    each_class_sorted_confidence = each_class_sorted_confidence[
                                                   ::-1]

                    # 5.2 按照类别置信度，进行由大到小的排序，each_class_sorted 形状为
                    # (classes, 6)
                    each_class_sorted = each_class_unsorted[
                        each_class_sorted_confidence]

                    # 5.3.0 创建一个空的列表 kept_bboxes，将用于存放挑选好的 bbox。
                    each_class_kept_bboxes = []

                    # 5.3.4 检查 each_class_sorted 的元素数量，如果元素数量 ≤ 1，则无
                    # 须再检查，可以跳出循环。该步骤用 while 循环实现，所以要提前放在这里。
                    while len(each_class_sorted) > 0:

                        # highest_confidence_bbox 是最高置信度的 bbox，形状为 (6)
                        highest_confidence_bbox = each_class_sorted[0]

                        # 如果恰好 highest_confidence_bbox 是最后一个元素，则可以跳
                        # 出循环。
                        if len(each_class_sorted) == 0:
                            break

                        # 5.3.1 存入列表 kept_bboxes
                        each_class_kept_bboxes.append(highest_confidence_bbox)
                        # 将 highest_confidence_bbox 从 each_class_sorted 中移出
                        # each_class_sorted 形状为 (classes, 6)
                        each_class_sorted = each_class_sorted[1:]

                        # 5.3.2 计算 each_class_sorted 中所有 bbox，和
                        # highest_confidence_bbox 的 DIOU。

                        # prediction_bbox 形状为 (classes, 4)
                        prediction_bbox = each_class_sorted[:, 2:]

                        # label_bbox 形状为 (4)
                        label_bbox = highest_confidence_bbox[2:]
                        # 将 label_bbox 扩展到和 prediction_bbox 相同形状
                        # (classes, 4)
                        label_bbox = np.broadcast_to(
                            label_bbox, shape=prediction_bbox.shape)

                        # ciou_calculator 的 2 个输入张量，可以是任意形状，只需要最后
                        # 一个维度的大小是 4，并且4个参数分别是 (center_x, center_y,
                        # height_bbox, width_bbox)
                        # 5.3.2 计算 diou_loss，diou_loss 形状为 (classes)
                        diou_loss = ciou_calculator(
                            label_bbox=label_bbox,
                            prediction_bbox=prediction_bbox, get_diou=True)

                        # 5.3.3 把 DIOU 小于阈值 0.5 作为 mask，进行过滤
                        diou_mask = diou_loss < diou_threshold

                        # 去掉了和 highest_confidence_bbox 重复的那些 bbox，得到
                        # DIOU 小于阈值的 each_class_sorted，形状为
                        # (classes_filtered, 6)
                        # 注意下面的索引方式，应该把第 1 个维度的分号 ：也加进来。
                        # 一个形状为 (1, 6)的数组被 [False] 索引之后，才会得到形状为
                        #  (0, 6) 的空数组。如果不加入第一个维度的分号 ：，索引之后会从
                        #  (1, 6) 变成 (6)，最终导致错误。
                        each_class_sorted = each_class_sorted[diou_mask, :]

                    # 5.4 将 each_class_kept_bboxes 转换为数组，形状为
                    # (classes_filtered, 6)
                    each_class_kept_bboxes = np.array(each_class_kept_bboxes)

                    if kept_bboxes_array is None:
                        kept_bboxes_array = each_class_kept_bboxes
                    else:
                        # 最终 kept_bboxes_array 的形状为 (m, 6)，表示经过 diou
                        # 计算，保留下了 m 个物体框
                        kept_bboxes_array = np.concatenate(
                            (kept_bboxes_array, each_class_kept_bboxes), axis=0)

            # 如果不是模型预测结果而是标签，则不需要进行 diou 计算，直接保留所有的 bboxes。
            else:
                kept_bboxes_array = id_confidence_bboxes

            # 把物体框信息输入到 bboxes_batch 中。需要先对 kept_bboxes_array
            # 中的数量进行修剪。
            if len(kept_bboxes_array) >= bboxes_quantity:
                kept_bboxes_array = kept_bboxes_array[: bboxes_quantity]

            # 如果 kept_bboxes_array 中的数量不够，需要对其进行补零操作。
            else:
                padding_rows = bboxes_quantity - len(kept_bboxes_array)
                kept_bboxes_array = np.pad(
                    kept_bboxes_array, ((0, padding_rows), (0, 0)),
                    'constant', constant_values=0)

            # kept_bboxes_array 修剪完成后，可以直接赋值给 bboxes_batch[i]。如果
            # kept_bboxes_array 为空，则 bboxes_batch 的第 i 个元素保持为 0.
            bboxes_batch[i] = kept_bboxes_array

    return bboxes_batch


def _visualize_one_batch_prediction(images_batch, bboxes_batch,
                                    show_classification_confidence=True,
                                    categories_to_detect=None,
                                    is_image=True, is_video=False):
    """将 YOLO-V4-CSP 的预测结果显示为图片。

    Arguments:
        images_batch：一个图片张量，形状为 (batch_size, *MODEL_IMAGE_SIZE, 3)。
        bboxes_batch：一个 float32 型数组，代表所有保留下来的 bboxes，形状为
            (batch_size, BBOXES_PER_IMAGE, 6)。其中每个长度为 6 的向量，第 0 为类别
            的 id，第 1 位是类别置信度，后四位是 bbox 的参数，分别是
            (center_x, center_y, height_bbox, width_bbox)。
        show_classification_confidence：一个布尔值。如果该值为 False，此时在输出的图
            片中，只显示类别的名字，不显示类别置信度。
        categories_to_detect：一个 Pandas 的 DataFrame 对象，包括了所有需要探测的类别。
        is_image：一个布尔值。如果用户输入图片，则该值为真，否则为假。
        is_video：一个布尔值。如果用户输入视频，则该值为真，否则为假。

    Returns:
        image_bgr: 对输入的每一个图片，都显示 3 个探测结果图片。3 个结果图片对应 p5, p4
            和 p3 共 3 个不同的特征层，结果图片内显示探测到的物体。
            如果在某个特征层没有探测到物体，则不显示该特征层的图片。
    """

    # 留给显示信息的高度和宽度。
    text_height = 20
    text_width = 60

    image_bgr = None

    for image_tensor, bboxes in zip(images_batch, bboxes_batch):

        # 此时 image_tensor 的数值范围是 [-1, 1]，需要转换为 [0, 255]
        image_tensor += 1
        image_tensor *= 127.5
        image_pillow = keras.preprocessing.image.array_to_img(
            image_tensor)

        # 因为 OpenCV 要求图片数组的数据类型为 uint8 。需要对数据类型进行转换，
        # np.asarray 会将图片对象默认生成 uint8 类型的数据。
        array_rgb = np.asarray(image_pillow, dtype=np.uint8)

        # 注意这里要用 numpy 数组的深度拷贝。因为后续的画框和写字操作，会对数组进行修改，
        # 所以无法直接在原始数组上进行，需要用 copy 新建一个图片数组。
        # 同时将 rgb 通道调整为 OpenCV 的 bgr 通道
        image_bgr = array_rgb[..., [2, 1, 0]].copy()

        if image_bgr is None:
            sys.exit(f"Could not read the image_bgr in detection_results.")
        # 只需要 image_bgr 是3D张量即可。PIL，Keras 和 OpenCV 的图片数组，都是
        #  height 在前，width 在后。
        image_height, image_width = image_bgr.shape[: 2]

        image_bgr = array_rgb[..., [2, 1, 0]].copy()

        # 如果该图片有物体框，则显示这些物体框。如果没有物体框，则只显示图片。
        if bboxes is not None:
            # 遍历每一个物体框.
            for each_bbox in bboxes:
                id_in_model = each_bbox[0]
                category_name = categories_to_detect.at[id_in_model, 'name']

                class_confidence = each_bbox[1]
                # 如果类别置信度等于 0，说明已经遍历完所有物体框，应该跳出循环，不显示剩余的
                # 空物体框。
                if np.isclose(class_confidence, 0):
                    break

                class_confidence = f'{class_confidence:.0%}'

                if show_classification_confidence:
                    show_text = category_name + ' ' + class_confidence

                # 如果是标签，则不需要显示置信度，因为此时置信度完全为 100%。
                else:
                    show_text = category_name

                bbox_center_x = each_bbox[-4]
                bbox_center_y = each_bbox[-3]
                bbox_height = each_bbox[-2]
                bbox_width = each_bbox[-1]

                # 此处获得预测框和真实的坐标。
                top_left_point_x = bbox_center_x - bbox_width / 2
                top_left_point_y = bbox_center_y - bbox_height / 2
                top_left_point_x = int(top_left_point_x)
                top_left_point_y = int(top_left_point_y)

                # 需要把物体框坐标限制在图片的范围之内，避免出现负数坐标。
                top_left_point_x = np.clip(top_left_point_x,
                                           a_min=0, a_max=image_width)
                top_left_point_y = np.clip(top_left_point_y,
                                           a_min=0, a_max=image_height)

                top_left_point = top_left_point_x, top_left_point_y

                bottom_right_point_x = bbox_center_x + bbox_width / 2
                bottom_right_point_y = bbox_center_y + bbox_height / 2
                bottom_right_point_x = int(bottom_right_point_x)
                bottom_right_point_y = int(bottom_right_point_y)

                # 需要把物体框坐标限制在图片的范围之内，避免出现负数坐标。
                bottom_right_point_x = np.clip(bottom_right_point_x,
                                               a_min=0, a_max=image_width)
                bottom_right_point_y = np.clip(bottom_right_point_y,
                                               a_min=0, a_max=image_height)

                bottom_right_point = (bottom_right_point_x,
                                      bottom_right_point_y)

                cv.rectangle(img=image_bgr, pt1=top_left_point,
                             pt2=bottom_right_point,
                             color=(0, 255, 0), thickness=2)

                text_point = list(top_left_point)
                text_point[1] -= 6
                if top_left_point[1] < text_height:
                    text_point[1] = text_height
                if image_width - top_left_point[0] < text_width:
                    text_point[0] = image_width - text_width

                cv.putText(img=image_bgr, text=show_text, org=text_point,
                           fontFace=cv.FONT_HERSHEY_TRIPLEX,
                           fontScale=0.5, color=(0, 255, 0))

        if is_image:
            print('\nPress key "q" to close all image windows.\n'
                  'Press key "s" to save the detected image.')
            cv.imshow(f'detection result', image_bgr)
            keyboard = cv.waitKey(0)
            if keyboard == ord("s"):
                cv.imwrite(r'tests\image_test.png', image_bgr)
            if keyboard == ord('q') or keyboard == 27:  # 27 为 esc 键
                cv.destroyAllWindows()

            print('\nPress any key to close all image windows.')
            cv.waitKey(0)
            cv.destroyAllWindows()
        elif is_video:
            # 该部分留待后续处理视频时使用。可能需要在上一级的程序里显示图片。
            # cv.imshow(f'{j}', image_bgr)
            pass

    return image_bgr


def visualize_predictions(
        image_input, predictions=None,
        objectness_threshold=OBJECTNESS_THRESHOLD,
        classification_threshold=CLASSIFICATION_CONFIDENCE_THRESHOLD,
        show_classification_confidence=True,
        diou_threshold=0.5, bboxes_quantity=100,
        categories_to_detect=None,
        is_image=True, is_video=False):
    """将 YOLO-v4-CSP 的预测结果显示为图片。

    Arguments:
        image_input: 一个用 coco_data_yolov4_csp 函数生成的 tf.data.Dataset。或是
            一个图片张量，形状为 (batch_size, *MODEL_IMAGE_SIZE, 3).
        predictions: 是图片对应的预测结果。如果 image_input 是 tf.data.Dataset，则
            predictions 可以缺省。
            如果 image_input 是一个图片张量，则必须有 predictions，并且 predictions
            应该是一个元祖，包含 3 个预测结果，按先后顺序依次是：
                p5_prediction， 一个 float32 型张量，形状为
                    (batch_size, *FEATURE_MAP_P5, 3, 85)。
                p4_prediction， 一个 float32 型张量，形状为
                    (batch_size, *FEATURE_MAP_P4, 3, 85)。
                p3_prediction， 一个 float32 型张量，形状为
                    (batch_size, *FEATURE_MAP_P3, 3, 85)。
        objectness_threshold: 一个 [0, 1] 之间的浮点数，是物体框内是否有物体的置信度
            阈值。如果预测结果的置信度小于该值，则对应的探测框不显示。
        classification_threshold: 一个 [0, 1] 之间的浮点数，是对于物体框内的物体，属于
            某个类别的置信度。如果预测结果的置信度小于该值，则对应的探测框不显示。
        show_classification_confidence: 一个布尔值。如果该值为 False，此时在输出的图
            片中，只显示类别的名字，不显示类别置信度。
        diou_threshold: 一个 [0, 1] 之间的浮点数，是计算 diou 时使用的阈值。如果 2 个
            bboxes 的 DIOU小于该阈值，则认为这两个 bboxes 对应的是两个不同的物体。
        bboxes_quantity: 一个整数，表示对于每张图片，只显示 bboxes_quantity 个 bboxes。
        categories_to_detect: 一个 Pandas 的 DataFrame 对象，包括了所有需要探测的类别。
        is_image: 一个布尔值。如果用户输入图片，则该值为真，否则为假。
        is_video: 一个布尔值。如果用户输入视频，则该值为真，否则为假。

    Returns:
        image_bgr: 对输入的每一个图片，都显示 3 个探测结果图片。3 个结果图片对应 p5, p4
            和 p3 共 3 个不同的特征层，结果图片内显示探测到的物体。
            如果在某个特征层没有探测到物体，则不显示该特征层的图片。
    """

    # ignore_types 是为了避免使用一些错误的类型，包括字符串或字节类型的数据。
    ignore_types = str, bytes

    # 如果没有 predictions，说明是标签数据。
    if predictions is None:
        # 如果是 tf.data.Dataset 的数据，则需要进行逐个取出。Dataset 属于 Iterable 。
        if isinstance(image_input, Iterable) and (
                not isinstance(image_input, ignore_types)):
            # 对于 Dataset 中的每一个 batch，对进行显示.
            for element in image_input:
                # 取出图片张量，形状为 (batch_size, *MODEL_IMAGE_SIZE, 3)
                images_batch = element[0]
                # 取出标签元祖。
                labels = element[1]

                # labels 不需要进行 DIOU NMS 操作，但是要用 diou_nms 将 labels
                # 转换成形状为 (n, 6) 的数组，所以设置 is_prediction 为 False。
                bboxes_batch = diou_nms(
                    predictions=labels,
                    objectness_threshold=objectness_threshold,
                    classification_threshold=classification_threshold,
                    diou_threshold=diou_threshold,
                    bboxes_quantity=bboxes_quantity,
                    is_prediction=False)

                _visualize_one_batch_prediction(
                    images_batch=images_batch,
                    bboxes_batch=bboxes_batch,
                    show_classification_confidence=False,
                    categories_to_detect=categories_to_detect,
                    is_image=is_image, is_video=is_video)

    # 如果提供了 predictions，则默认为输入的是一个图片张量，以及一个预测结果元祖，其中包含
    #  3 个预测结果张量。
    else:
        # 需要对模型的输出结果进行解码，将其预测值转换为概率，并把 bbox 转换到 608x608
        # 图片中的等效大小。需要把预测结果元祖里的 3 个结果，分别使用 predictor 进行转换。

        transformed_predictions = (predictor(predictions[0]),
                                   predictor(predictions[1]),
                                   predictor(predictions[2]))

        # 将模型的预测结果进行 DIOU NMS 操作。
        bboxes_batch = diou_nms(
            predictions=transformed_predictions,
            objectness_threshold=objectness_threshold,
            classification_threshold=classification_threshold,
            diou_threshold=diou_threshold,
            bboxes_quantity=bboxes_quantity,
            is_prediction=True)

        _visualize_one_batch_prediction(
            images_batch=image_input,
            bboxes_batch=bboxes_batch,
            show_classification_confidence=show_classification_confidence,
            categories_to_detect=categories_to_detect,
            is_image=is_image, is_video=is_video)
