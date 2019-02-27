#!/usr/bin/env python
import theano.tensor as T
from lasagne.layers import MergeLayer


class ApplyAttentionLayer(MergeLayer):
    """
    Attention Layer
    """

    def __init__(self, incomings, scale=1, loss_weight=0, **kwargs):
        super(ApplyAttentionLayer, self).__init__(incomings, **kwargs)
        self.scale = scale
        self.loss = T.constant(0.0)
        self.loss_weight = loss_weight

    def get_output_shape_for(self, input_shapes):
        output_shape = input_shapes[0]
        return output_shape

    def get_output_for(self, inputs, **kwargs):

        # apply attention
        attention = inputs[1]
        output = inputs[0] * attention * self.scale

        # apply entropy regularization on attention
        if self.loss_weight > 0:
            eps = 1e-9
            log_probs = T.log(attention + eps)
            entropy = -(log_probs * attention).sum(axis=-1).mean() * self.loss_weight
            self.loss = -entropy

        return output

    def get_loss(self):
        return self.loss

