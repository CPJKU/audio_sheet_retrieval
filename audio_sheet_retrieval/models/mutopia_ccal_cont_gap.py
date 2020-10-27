import theano.tensor as T

import lasagne
from lasagne.layers import SliceLayer
from lasagne.nonlinearities import elu, identity
from audio_sheet_retrieval.utils.monitoring import print_architecture
from audio_sheet_retrieval.models.lasagne_extensions.layers.cca import LengthNormLayer, LearnedCCALayer
from audio_sheet_retrieval.models.lasagne_extensions.layers.cca import CCALayer

try:
    from lasagne.layers import dnn
    Conv2DLayer = dnn.Conv2DDNNLayer
    MaxPool2DLayer = dnn.MaxPool2DDNNLayer
    batch_norm = dnn.batch_norm_dnn
except:
    from lasagne.layers import Conv2DLayer, MaxPool2DLayer, batch_norm


INI_LEARNING_RATE = 0.002
REFINEMENT_STEPS = 10
LR_MULTIPLIER = 0.5
BATCH_SIZE = 100
MOMENTUM = 0.9
MAX_EPOCHS = 1000
PATIENCE = 15
X_TENSOR_TYPE = T.tensor4
Y_TENSOR_TYPE = T.ivector

DIM_LATENT = 32

L1 = None
L2 = 0.00001  # None
GRAD_NORM = None

r1 = r2 = 1e-3
rT = 1e-3

nonlin = elu
init = lasagne.init.HeUniform

FIT_CCA = False
ALPHA = 1.0
WEIGHT_TNO = 0.0
USE_CCAL = True
GAMMA = 0.7


def conv_bn(net_in, num_filters, nonlinearity):
    """ Compile convolution layer with batch norm """
    net = Conv2DLayer(net_in, num_filters=num_filters, filter_size=3, pad=1, W=init(),
                      nonlinearity=nonlinearity, name='conv_bn')
    return batch_norm(net)


def get_build_model(weight_tno, alpha, dim_latent, use_ccal):
    """ Get model_name function """

    def model(input_shape_1, input_shape_2, show_model):
        """ Compile net architecture """

        # --- input layers ---
        l_view1 = lasagne.layers.InputLayer(shape=(None, input_shape_1[0], input_shape_1[1] // 2, input_shape_1[2] // 2))
        l_view2 = lasagne.layers.InputLayer(shape=(None, input_shape_2[0], input_shape_2[1], input_shape_2[2]))

        net1 = l_view1
        net2 = l_view2

        # --- feed forward part view 1 ---
        num_filters_1 = 12

        net1 = conv_bn(net1, num_filters_1, nonlin)
        net1 = conv_bn(net1, num_filters_1, nonlin)
        net1 = MaxPool2DLayer(net1, pool_size=2)

        net1 = conv_bn(net1, 2 * num_filters_1, nonlin)
        net1 = conv_bn(net1, 2 * num_filters_1, nonlin)
        net1 = MaxPool2DLayer(net1, pool_size=2)

        net1 = conv_bn(net1, 4 * num_filters_1, nonlin)
        net1 = conv_bn(net1, 4 * num_filters_1, nonlin)
        net1 = MaxPool2DLayer(net1, pool_size=2)

        net1 = conv_bn(net1, 4 * num_filters_1, nonlin)
        net1 = conv_bn(net1, 4 * num_filters_1, nonlin)
        net1 = MaxPool2DLayer(net1, pool_size=2)

        net1 = Conv2DLayer(net1, num_filters=dim_latent, filter_size=1, pad=0, W=init(), nonlinearity=identity)
        net1 = batch_norm(net1)

        net1 = lasagne.layers.GlobalPoolLayer(net1)
        l_v1latent = lasagne.layers.FlattenLayer(net1, name='Flatten')

        # --- feed forward part view 2 ---
        num_filters_2 = num_filters_1

        net2 = conv_bn(net2, num_filters_2, nonlin)
        net2 = conv_bn(net2, num_filters_2, nonlin)
        net2 = MaxPool2DLayer(net2, pool_size=2)

        net2 = conv_bn(net2, 2 * num_filters_2, nonlin)
        net2 = conv_bn(net2, 2 * num_filters_2, nonlin)
        net2 = MaxPool2DLayer(net2, pool_size=2)

        net2 = conv_bn(net2, 4 * num_filters_2, nonlin)
        net2 = conv_bn(net2, 4 * num_filters_2, nonlin)
        net2 = MaxPool2DLayer(net2, pool_size=2)

        net2 = conv_bn(net2, 4 * num_filters_2, nonlin)
        net2 = conv_bn(net2, 4 * num_filters_2, nonlin)
        net2 = MaxPool2DLayer(net2, pool_size=2)

        net2 = Conv2DLayer(net2, num_filters=dim_latent, filter_size=1, pad=0, W=init(), nonlinearity=identity)
        net2 = batch_norm(net2)

        net2 = lasagne.layers.GlobalPoolLayer(net2)
        l_v2latent = lasagne.layers.FlattenLayer(net2, name='Flatten')

        # --- multi modality part ---

        # merge modalities by cca projection or learned embedding layer
        if use_ccal:
            net = CCALayer([l_v1latent, l_v2latent], r1, r2, rT, alpha=alpha, wl=weight_tno)
        else:
            net = LearnedCCALayer([l_v1latent, l_v2latent], U=init(), V=init(), alpha=alpha)

        # split modalities again
        l_v1latent = SliceLayer(net, slice(0, dim_latent), axis=1)
        l_v2latent = SliceLayer(net, slice(dim_latent, 2 * dim_latent), axis=1)

        # normalize (per row) output to length 1.0
        l_v1latent = LengthNormLayer(l_v1latent)
        l_v2latent = LengthNormLayer(l_v2latent)

        # --- print architectures ---
        if show_model:
            print_architecture(l_v1latent)
            print_architecture(l_v2latent)

        return l_view1, l_view2, l_v1latent, l_v2latent

    return model

build_model = get_build_model(weight_tno=WEIGHT_TNO, alpha=ALPHA, dim_latent=DIM_LATENT, use_ccal=USE_CCAL)


def objectives():
    """ Compile objectives """
    from .objectives import get_contrastive_cos_loss
    return get_contrastive_cos_loss(1.0 - WEIGHT_TNO, GAMMA)


def compute_updates(all_grads, all_params, learning_rate):
    """
    Compute gradients for updates
    """
    return lasagne.updates.adam(all_grads, all_params, learning_rate)


def update_learning_rate(lr, epoch=None):
    """ Update learning rate """
    return lr


def prepare(x, y=None):
    """ prepare images for training """
    import cv2
    import numpy as np

    # convert sheet snippet to float
    x = x.astype(np.float32)
    x /= 255

    # # resize sheet image
    # sheet_shape = [x.shape[2] // 2, x.shape[3] // 2]
    # new_shape = [x.shape[0], x.shape[1], ] + sheet_shape
    # x_new = np.zeros(new_shape, np.float32)
    # for i in range(len(x)):
    #     x_new[i, 0] = cv2.resize(x[i, 0], (sheet_shape[1], sheet_shape[0]))
    # x = x_new

    if y is None:
        return x
    else:
        return x, y


def valid_batch_iterator():
    """ Compile batch iterator """
    from audio_sheet_retrieval.utils.batch_iterators import MultiviewPoolIteratorUnsupervised
    batch_iterator = MultiviewPoolIteratorUnsupervised(batch_size=BATCH_SIZE, prepare=prepare, shuffle=False)
    return batch_iterator


def train_batch_iterator(batch_size=BATCH_SIZE):
    """ Compile batch iterator """
    from audio_sheet_retrieval.utils.batch_iterators import MultiviewPoolIteratorUnsupervised
    batch_iterator = MultiviewPoolIteratorUnsupervised(batch_size=batch_size, prepare=prepare, k_samples=10000)
    return batch_iterator
