from __future__ import print_function

import sys
import warnings
from nets import objectives
from theano import tensor as T
from nets import optpolicy, layers
from lasagne import init, nonlinearities as nl, layers as ll
from lasagne.layers.dnn import Pool2DDNNLayer as MaxPool2DLayer
from experiments.utils import run_experiment, build_params_from_init

warnings.simplefilter("ignore")

def net_lenet5(input_shape, nclass):
    input_x, target_y, Winit = T.tensor4("input"), T.vector("target", dtype='int32'), init.Normal()

    net = ll.InputLayer(input_shape, input_x)

    net = layers.Conv2DVarDropOutARD(net, 20, 5, W=init.Normal())
    net = MaxPool2DLayer(net, 2)

    net = layers.Conv2DVarDropOutARD(net, 50, 5, W=init.Normal())
    net = MaxPool2DLayer(net, 2)

    net = layers.DenseVarDropOutARD(net, 500, W=init.Normal())
    net = layers.DenseVarDropOutARD(net, nclass, W=init.Normal(), nonlinearity=nl.softmax)

    return net, input_x, target_y, 1

k = float(sys.argv[1]) if len(sys.argv) > 1 else 0
dataset = str(sys.argv[2]) if len(sys.argv) > 2 else 'mnist'
iparam = str(sys.argv[3]) if len(sys.argv) > 3 else None
print('k = ', k, 'dataset = ', dataset, 'params = ', iparam)

num_epochs, batch_size, verbose = 200, 100, 1
optpol = lambda epoch: optpolicy.lr_linear_to0(epoch, 1e-3)
arch = net_lenet5

net = run_experiment(
    dataset, 0, batch_size, arch, objectives.sgvlb, False,
    optpol, optpolicy.rw_linear, optimizer='adam')

paramsv = build_params_from_init(net, iparam, lsinit=-10) if iparam else None

net = run_experiment(
    dataset, num_epochs, batch_size, arch, objectives.sgvlb, verbose,
    optpol, optpolicy.rw_linear, params=paramsv, optimizer='adam', train_clip=True)