from __future__ import print_function

import warnings
import sys
from theano import tensor as T
from nets import objectives, layers, optpolicy
from lasagne import init, nonlinearities as nl, layers as ll
from zexperiments.utils import run_experiment, build_params_from_init, save_net

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

paramsv = build_params_from_init(net, iparam, lsinit=-15) if iparam else None

net = run_experiment(
    dataset, num_epochs, batch_size, arch, objectives.sgvlb, verbose,
    optpol, optpolicy.rw_linear__, params=paramsv, optimizer='adam', train_clip=True)
