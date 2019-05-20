from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import  numpy as np
import tensorflow as tf
from dltk.core.residual_unit import vanilla_residual_unit_3d
from dltk.groupy.gconv.tensorflow_gconv.splitgconv3d import gconv3d_unit, group_norm


def groupnet_3d(inputs,
              num_classes,
              filters=(16, 32, 64, 128),
              strides=((2, 2, 2), (1, 1, 1), (2, 2, 2), (1, 1, 1)),
              mode=tf.estimator.ModeKeys.EVAL,
              use_bias=False,
              activation=tf.nn.relu6,
              kernel_initializer=tf.initializers.variance_scaling(distribution='uniform'),
              bias_initializer=tf.zeros_initializer(),
              kernel_regularizer=None, bias_regularizer=None):
    """
    Regression/classification network based on group convolution networks


    Args:
        inputs (tf.Tensor): Input feature tensor to the network (rank 5
            required).
        num_classes (int): Number of output channels or classes.
        num_res_units (int, optional): Number of residual units per resolution
            scale.
        filters (tuple, optional): Number of filters for all residual units at
            each resolution scale.
        strides (tuple, optional): Stride of the first unit on a resolution
            scale.
        mode (TYPE, optional): One of the tf.estimator.ModeKeys strings: TRAIN,
            EVAL or PREDICT
        use_bias (bool, optional): Boolean, whether the layer uses a bias.
        activation (optional): A function to use as activation function.
        kernel_initializer (TYPE, optional): An initializer for the convolution
            kernel.
        bias_initializer (TYPE, optional): An initializer for the bias vector.
            If None, no bias will be applied.
        kernel_regularizer (None, optional): Optional regularizer for the
            convolution kernel.
        bias_regularizer (None, optional): Optional regularizer for the bias
            vector.

    Returns:
        dict: dictionary of output tensors

    """
    outputs = {}
    assert len(strides) == len(filters)
    assert len(inputs.get_shape().as_list()) == 5, \
        'inputs are required to have a rank of 5.'

    relu_op = tf.nn.relu6

    conv_params = {'padding': 'same',
                   'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}

    x = inputs
    #print('input shape: ', x.get_shape())

    for i in range(len(filters)):
        x = gconv3d_unit(x, filters[i])
        ##batch normalization can be done groupwise ('group_norm') or as usual on the whole batch ('batch_normalization').
        x = group_norm(x, G=24)
        #x = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN)
        x = activation(x)
        x = tf.layers.dropout(x,rate=0.2,training=mode == tf.estimator.ModeKeys.TRAIN)


    # Global pool and last unit
    with tf.variable_scope('pool'):
        x = tf.layers.batch_normalization(
            x, training=mode == tf.estimator.ModeKeys.TRAIN)
        x = relu_op(x)

        axis = tuple(range(len(x.get_shape().as_list())))[1:-1]
        x = tf.reduce_mean(x, axis=axis, name='global_avg_pool')

        tf.logging.info('Global pool shape {}'.format(x.get_shape()))

    with tf.variable_scope('last'):
        x = tf.layers.dense(inputs=x,
                            units=num_classes,
                            activation=None,
                            use_bias=conv_params['use_bias'],
                            kernel_initializer=conv_params['kernel_initializer'],
                            bias_initializer=conv_params['bias_initializer'],
                            kernel_regularizer=conv_params['kernel_regularizer'],
                            bias_regularizer=conv_params['bias_regularizer'],
                            name='hidden_units')

        tf.logging.info('Output tensor shape {}'.format(x.get_shape()))

    # Define the outputs
    outputs['logits'] = x

    with tf.variable_scope('pred'):

        y_prob = tf.nn.softmax(x)
        outputs['y_prob'] = y_prob

        y_ = tf.argmax(x, axis=-1) \
            if num_classes > 1 \
            else tf.cast(tf.greater_equal(x[..., 0], 0.5), tf.int32)
        outputs['y_'] = y_

    return outputs
