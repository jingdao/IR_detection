#coding=utf-8
# MIT License
#
#Copyright (c) 2018 Thalles Santos Silva
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
# ==============================================================================
"""

Adapted from DeepLab V3 with Atrous Convolutions:
	Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam
	Rethinking Atrous Convolution for Semantic Image Segmentation
	https://arxiv.org/pdf/1706.05587.pdf
"""

import tensorflow as tf
slim = tf.contrib.slim
from resnet import resnet_v2, resnet_utils

# ImageNet mean statistics
_R_MEAN = 72
_G_MEAN = 72
_B_MEAN = 72

@slim.add_arg_scope
def atrous_spatial_pyramid_pooling(net, scope, depth=256, reuse=None):
    """
    ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
    (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
    :param net: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
    :param scope: scope name of the aspp layer
    :return: network layer with aspp applyed to it.
    """

    with tf.variable_scope(scope, reuse=reuse):
        feature_map_size = tf.shape(net)

        # apply global average pooling
        image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keepdims=True)
        image_level_features = slim.conv2d(image_level_features, depth, [1, 1], scope="image_level_conv_1x1",
                                           activation_fn=None)
        image_level_features = tf.image.resize_bilinear(image_level_features, (feature_map_size[1], feature_map_size[2]))

        at_pool1x1 = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_0", activation_fn=None)

        at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1", rate=6, activation_fn=None)

        at_pool3x3_2 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_2", rate=12, activation_fn=None)

        at_pool3x3_3 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_3", rate=18, activation_fn=None)

        net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3,
                        name="concat")
        net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)
        return net


def deeplab_v3(inputs, args, is_training, reuse):

    # mean subtraction normalization
    inputs = inputs - 72

    # inputs has shape - Original: [batch, 513, 513, 3]
    with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training,
                                                      args.batch_norm_decay,
                                                      args.batch_norm_epsilon)):
        resnet = getattr(resnet_v2, args.resnet_model)
        _, end_points = resnet(inputs,
                               args.number_of_classes,
                               is_training=is_training,
                               global_pool=False,
                               spatial_squeeze=False,
                               output_stride=args.output_stride,
                               reuse=reuse)

        with tf.variable_scope("DeepLab_v3", reuse=reuse):

            # get block 4 feature outputs
            net = end_points[args.resnet_model + '/block4']

            net = atrous_spatial_pyramid_pooling(net, "ASPP_layer", depth=256, reuse=reuse)

            net = slim.conv2d(net, args.number_of_classes, [1, 1], activation_fn=None,
                              normalizer_fn=None, scope='logits')

            size = tf.shape(inputs)[1:3]
            # resize the output logits to match the labels dimensions
            #net = tf.image.resize_nearest_neighbor(net, size)
            net = tf.image.resize_bilinear(net, size)
            return net
