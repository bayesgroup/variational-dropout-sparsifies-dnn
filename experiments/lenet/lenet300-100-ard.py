from __future__ import print_function

import sys
import warnings
from nets import objectives
from theano import tensor as T
from nets import optpolicy, layers
from lasagne import init, nonlinearities as nl, layers as ll
from experiments.utils import run_experiment

warnings.simplefilter("ignore")

def net_lenet5(input_shape, nclass):
    input_x, target_y, Winit = T.tensor4("input"), T.vector("target", dtype='int32'), init.Normal()

    net = ll.InputLayer(input_shape, input_x)

    net = layers.DenseVarDropOutARD(net, 300, W=init.Normal())
    net = layers.DenseVarDropOutARD(net, 100, W=init.Normal())
    net = layers.DenseVarDropOutARD(net, nclass, W=init.Normal(), nonlinearity=nl.softmax)

    return net, input_x, target_y, 1

dataset = 'mnist'
iparam = str(sys.argv[1]) if len(sys.argv) > 1 else None
print('dataset = ', dataset, 'params = ', iparam)

num_epochs, batch_size, verbose = 200, 100, 1
optpol = lambda epoch: optpolicy.lr_linear(epoch, 1e-3)
arch = net_lenet5

net = run_experiment(
    dataset, num_epochs, batch_size, arch, objectives.sgvlb, verbose,
    optpol, optpolicy.rw_linear, params=None, optimizer='adam', train_clip=True)
