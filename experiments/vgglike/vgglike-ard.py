from __future__ import print_function

import warnings
warnings.simplefilter("ignore")

import sys
from nets import objectives
from theano import tensor as T
from nets import optpolicy, layers
from lasagne import init, nonlinearities as nl, layers as ll
from lasagne.layers.dnn import Pool2DDNNLayer as MaxPool2DLayer, BatchNormDNNLayer as BatchNormLayer
from experiments.utils import run_experiment, build_params_from_init


def conv_bn_rectify(net, num_filters):
    net = layers.Conv2DVarDropOutARD(net, int(num_filters), 3, W=init.Normal(), pad=1, nonlinearity=nl.linear)
    net = BatchNormLayer(net, epsilon=1e-3)
    net = ll.NonlinearityLayer(net)

    return net

def net_vgglike(k, input_shape, nclass):
    input_x, target_y, Winit = T.tensor4("input"), T.vector("target", dtype='int32'), init.Normal()

    net = ll.InputLayer(input_shape, input_x)
    net = conv_bn_rectify(net, 64 * k)
    net = conv_bn_rectify(net, 64 * k)
    net = MaxPool2DLayer(net, 2, 2)

    net = conv_bn_rectify(net, 128 * k)
    net = conv_bn_rectify(net, 128 * k)
    net = MaxPool2DLayer(net, 2, 2)

    net = conv_bn_rectify(net, 256 * k)
    net = conv_bn_rectify(net, 256 * k)
    net = conv_bn_rectify(net, 256 * k)
    net = MaxPool2DLayer(net, 2, 2)

    net = conv_bn_rectify(net, 512 * k)
    net = conv_bn_rectify(net, 512 * k)
    net = conv_bn_rectify(net, 512 * k)
    net = MaxPool2DLayer(net, 2, 2)

    net = conv_bn_rectify(net, 512 * k)
    net = conv_bn_rectify(net, 512 * k)
    net = conv_bn_rectify(net, 512 * k)
    net = MaxPool2DLayer(net, 2, 2)

    net = layers.DenseVarDropOutARD(net, int(512 * k), W=init.Normal(), nonlinearity=nl.linear)
    net = BatchNormLayer(net, epsilon=1e-3)
    net = ll.NonlinearityLayer(net)
    net = layers.DenseVarDropOutARD(net, nclass, W=init.Normal(), nonlinearity=nl.softmax)

    return net, input_x, target_y, k

k = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
dataset = str(sys.argv[2]) if len(sys.argv) > 2 else 'cifar10'
iparam = str(sys.argv[3]) if len(sys.argv) > 3 else None
averaging = int(sys.argv[4]) if len(sys.argv) > 4 else 0
print('k = ', k, 'dataset = ', dataset, 'params = ', iparam)

num_epochs, batch_size, verbose = 200, 100, 1
optpol = lambda epoch: optpolicy.lr_linear(epoch, 1e-5)
arch = lambda input_shape, s: net_vgglike(k, input_shape, s)

net = run_experiment(
    dataset, 0, batch_size, arch, objectives.sgvlb, False,
    optpol, optpolicy.rw_linear, optimizer='adam')

paramsv = build_params_from_init(net, iparam) if iparam else None

net = run_experiment(
    dataset, num_epochs, batch_size, arch, objectives.sgvlb, verbose,
    optpol, optpolicy.rw_linear, params=paramsv, optimizer='adam')
