#!/usr/bin/env python

from __future__ import print_function

import lasagne
from lasagne.layers import MaxPool2DLayer, ElemwiseSumLayer, TransposedConv2DLayer, batch_norm, DropoutLayer
from lasagne.layers import Conv2DLayer as Conv2DLayer
from lasagne.nonlinearities import elu, sigmoid, rectify


INPUT_SHAPE = [1, 512, 512]


def conv_bn(in_layer, num_filters, filter_size, nonlinearity=rectify, pad='same'):
    """ convolution block with with batch normalization """
    in_layer = Conv2DLayer(in_layer, num_filters=num_filters, filter_size=filter_size,
                           nonlinearity=nonlinearity, pad=pad, name='conv')
    in_layer = batch_norm(in_layer)
    return in_layer


def build_model(in_shape=INPUT_SHAPE):
    """ Compile net architecture """
    nonlin = elu

    net1 = lasagne.layers.InputLayer(shape=(None, in_shape[0], in_shape[1], in_shape[2]), name='Input')

    # number of filters in first layer
    # decreased by factor 2 in each block
    nf0 = 8

    # --- encoder ---
    net1 = conv_bn(net1, num_filters=nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    net1 = conv_bn(net1, num_filters=nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    p1 = net1
    net1 = MaxPool2DLayer(net1, pool_size=2, stride=2)

    net1 = conv_bn(net1, num_filters=2 * nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    net1 = conv_bn(net1, num_filters=2 * nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    p2 = net1
    net1 = MaxPool2DLayer(net1, pool_size=2, stride=2)

    net1 = conv_bn(net1, num_filters=4 * nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    net1 = conv_bn(net1, num_filters=4 * nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    p3 = net1
    net1 = MaxPool2DLayer(net1, pool_size=2, stride=2)

    net1 = conv_bn(net1, num_filters=8 * nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    net1 = conv_bn(net1, num_filters=8 * nf0, filter_size=3, nonlinearity=nonlin, pad='same')

    # --- decoder ---
    net1 = TransposedConv2DLayer(net1, num_filters=4 * nf0, filter_size=2, stride=2)
    net1 = batch_norm(net1)
    net1 = ElemwiseSumLayer((p3, net1))
    net1 = batch_norm(net1)
    net1 = conv_bn(net1, num_filters=4 * nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    net1 = conv_bn(net1, num_filters=4 * nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    net1 = DropoutLayer(net1, p=0.2)

    net1 = TransposedConv2DLayer(net1, num_filters=2 * nf0, filter_size=2, stride=2)
    net1 = batch_norm(net1)
    net1 = ElemwiseSumLayer((p2, net1))
    net1 = batch_norm(net1)
    net1 = conv_bn(net1, num_filters=2 * nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    net1 = conv_bn(net1, num_filters=2 * nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    net1 = DropoutLayer(net1, p=0.1)

    net1 = TransposedConv2DLayer(net1, num_filters=nf0, filter_size=2, stride=2)
    net1 = batch_norm(net1)
    net1 = ElemwiseSumLayer((p1, net1))
    net1 = batch_norm(net1)
    net1 = conv_bn(net1, num_filters=nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    net1 = conv_bn(net1, num_filters=nf0, filter_size=3, nonlinearity=nonlin, pad='same')

    net1 = Conv2DLayer(net1, num_filters=1, filter_size=1, nonlinearity=sigmoid, pad='same')
    return net1
