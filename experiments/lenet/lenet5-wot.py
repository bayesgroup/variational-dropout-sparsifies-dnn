from __future__ import print_function

import warnings
from nets import objectives
from theano import tensor as T
from nets import optpolicy
from lasagne import init, nonlinearities as nl, layers as ll
from experiments.utils import run_experiment
from lasagne.layers.dnn import Pool2DDNNLayer as MaxPool2DLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer


warnings.simplefilter("ignore")

def net_lenet5(input_shape, nclass):
    input_x, target_y, Winit = T.tensor4("input"), T.vector("target", dtype='int32'), init.Normal()

    net = ll.InputLayer(input_shape, input_x)

    net = ConvLayer(net, 20, 5, W=init.Normal())
    net = MaxPool2DLayer(net, 2)

    net = ConvLayer(net, 50, 5, W=init.Normal())
    net = MaxPool2DLayer(net, 2)

    net = ll.DenseLayer(net, 500, W=init.Normal())
    net = ll.DenseLayer(net, nclass, W=init.Normal(), nonlinearity=nl.softmax)

    return net, input_x, target_y, 1

num_epochs, batch_size, verbose, dataset = 200, 100, 1, 'mnist'
optp = lambda epoch: optpolicy.lr_linear(epoch, 1e-4)
arch = net_lenet5

net = run_experiment(
    dataset, num_epochs, batch_size, arch, objectives.sgvlb,
    verbose, optp, optpolicy.rw_linear, optimizer='adam', da=True)