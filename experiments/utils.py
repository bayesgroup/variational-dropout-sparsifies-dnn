from __future__ import print_function

import os
import sys
import random
import datetime
import numpy as np

from nets import utils
from data import reader
from lasagne import layers as ll
from time import gmtime, strftime


def get_logging_print(fname):
    cur_time = strftime("%m-%d_%H:%M:%S", gmtime())

    def prin(*args):
        str_to_write = ' '.join(map(str, args))
        with open(fname % cur_time, 'a') as f:
            f.write(str_to_write + '\n')
            f.flush()

        print(str_to_write)
        sys.stdout.flush()

    return prin


def experiment_info(dataset, num_epochs, batch_size, arch, obj, optpolicy_lr, optpolicy_rw, **params):
    info = [
        '\n', '=' * 80, '\n>> Experiment parameters:\n',
        'dataset:       ', dataset, '\n',
        'num_epochs:    ', num_epochs, '\n',
        'batch_size:    ', batch_size, '\n',
        'arch:          ', arch.__name__, '\n',
        'objective:     ', obj.__name__, '\n',
        'optpolicy_lr:     ', optpolicy_lr.__name__, '\n',
        'optpolicy_rw:     ', optpolicy_rw.__name__, '\n',
        'Commandline:         ', ' '.join(sys.argv) + '\n']

    return ''.join([str(x) for x in info])


def run_experiment(dataset, num_epochs, batch_size, arch, obj, verbose,
                   optpolicy_lr, optpolicy_rw, log_fname=None, params=None,
                   train_clip=False, thresh=3, optimizer='adam', da=False):
    data, train_size, test_size, input_shape, nclass = reader.load(dataset)
    net, input_x, target_y, k = arch(input_shape, nclass)

    if num_epochs == 0:
        return net

    if params is not None:
        ll.set_all_param_values(net, params)

    # Default log file name = experiment script file name
    if log_fname is None:
        log_fname = sys.argv[0].split('/')[-1][:-3]

    if not os.path.exists('./experiments/logs'):
        os.mkdir('./experiments/logs')

    base_fname = './experiments/logs/{fname}-{dataset}-%s.txt'
    print = get_logging_print(base_fname.format(dataset=dataset, fname=log_fname))
    print(experiment_info(**locals()))
    print(utils.net_configuration(net, short=(not verbose)))

    print('start compile', datetime.datetime.now().isoformat()[:16].replace('T', ' '))
    trainf, testf, predictf, up_opt, up_rw = utils.get_functions(**locals())
    print('finish compile', datetime.datetime.now().isoformat()[:16].replace('T', ' '))
    net, tr_info, te_info = utils.train(
        net, trainf, testf, up_opt, optpolicy_lr, up_rw, optpolicy_rw,
        data, num_epochs, batch_size, verbose, printf=print, thresh=thresh, da=da)

    print(save_net(net, dataset, k))
    print(utils.test_net(net, testf, data, 'ard' in sys.argv[0].split('/')[-1][:-3]))

    return net

def save_net(net, dataset, k):
    params = ll.get_all_param_values(net)
    fname = sys.argv[0].split('/')[-1][:-3]
    hash = ''.join([chr(random.randint(97, 122)) for _ in range(3)])

    if not os.path.exists('./experiments/weights'):
        os.mkdir('./experiments/weights')

    name = './experiments/weights/%s-%s-%s-%s' % (dataset, fname, k, hash)
    print('save model: ' + name)
    np.save(name, params)
    return name + '.npy'

def build_params_from_init(net, init_name, lsinit=-15, verbose=False):
    init_paramsv = list(np.load(init_name))
    params, paramsv, ardi = ll.get_all_params(net), [], 0

    for i in range(len(params)):
        if params[i].name in ['W', 'beta', 'gamma', 'mean', 'inv_std', 'b']:
            paramsv.append(init_paramsv[ardi])
            ardi += 1
        elif params[i].name == 'ls2':
            sh = paramsv[-1] if len(paramsv[-1].shape) == 4 else paramsv[-2]
            init_ = np.zeros_like(sh)
            paramsv.append(init_ + lsinit)
        else:
            raise Exception('wtf' + params[i].name)

        if verbose:
            print(params[i].name, paramsv[-1].shape)

    return paramsv



