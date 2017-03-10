from __future__ import print_function

import sys
import time
import random
import lasagne
import numpy as np

from objectives import *
from tabulate import tabulate

np.set_printoptions(precision=4, linewidth=150)


def get_functions(net, obj, input_x, target_y, batch_size, train_size, test_size, optimizer='nesterov',
                  train_clip=False, thresh=3, **params):

    predict_train = lasagne.layers.get_output(net, deterministic=False, train_clip=train_clip, thresh=thresh)
    accuracy_train = lasagne.objectives.categorical_accuracy(predict_train, target_y).mean()
    loss_train, rw = obj(
        predict_train, target_y, net, batch_size=batch_size, num_samples=train_size,
        train_clip=train_clip, thresh=thresh, **params)
    nll_train = ell(predict_train, target_y)
    reg_train = rw*reg(net, train_clip=train_clip, thresh=thresh)

    predict_test = lasagne.layers.get_output(net, thresh=thresh, deterministic=True)
    accuracy_test = lasagne.objectives.categorical_accuracy(predict_test, target_y).mean()
    loss_test, _ = obj(predict_test, target_y, net, batch_size=batch_size, num_samples=test_size, rw=rw, thresh=thresh, **params)
    nll_test = ell(predict_test, target_y)
    reg_test = rw*reg(net, train_clip=train_clip, thresh=thresh)

    weights = lasagne.layers.get_all_params(net, trainable=True)
    lr, beta = theano.shared(np.cast[theano.config.floatX](0)), theano.shared(np.cast[theano.config.floatX](0))
    if optimizer == 'nesterov':
        updates = lasagne.updates.nesterov_momentum(loss_train, weights, learning_rate=lr, momentum=beta)
    elif optimizer == 'adam':
        updates = lasagne.updates.adam(loss_train, weights, learning_rate=lr, beta1=beta)
    else:
        raise Exception('opt wtf')

    train_func = theano.function(
        [input_x, target_y], [loss_train, accuracy_train, nll_train, reg_train],
        allow_input_downcast=True, updates=updates)
    test_func = theano.function(
        [input_x, target_y], [loss_test,  accuracy_test,  nll_test,  reg_test],
        allow_input_downcast=True)

    def update_optimizer(new_lr, new_beta):
        lr.set_value(np.cast[theano.config.floatX](new_lr))
        beta.set_value(np.cast[theano.config.floatX](new_beta))

    def update_regweight(new_rw):
        rw.set_value(np.cast[theano.config.floatX](new_rw))

    return train_func, test_func, predict_test, update_optimizer, update_regweight

def net_configuration(net, short=False):
    if short:
        nl = net.input_layer.nonlinearity.func_name if hasattr(net.input_layer, 'nonlinearity') else 'linear'
        return "%s, %s, %s:" % (net.input_layer.name, net.input_layer.input_layer.output_shape, nl)

    table = []
    header = ['Layer', 'output_shape', 'parameters', 'nonlinearity']
    while hasattr(net, 'input_layer'):
        if hasattr(net, 'nonlinearity') and hasattr(net.nonlinearity, 'func_name'):
            nl = net.nonlinearity.func_name
        else:
            nl = 'linear'

        if net.name is not None:
            table.append((net.name, net.output_shape, net.params.keys(), nl))
        else:
            table.append((str(net.__class__).split('.')[-1][:-2],
                          net.output_shape, net.params.keys(), nl))
        net = net.input_layer

    if hasattr(net, 'nonlinearity') and hasattr(net.nonlinearity, 'func_name'):
        nl = net.nonlinearity.func_name
    else:
        nl = 'linear'
    table.append((net.name, net.output_shape, net.params.keys(), nl))

    return ">> Net Architecture\n" + tabulate(reversed(table), header, floatfmt=u'.3f') + '\n'


def iter_info(verbose, epoch, start_time, num_epochs, updates, train_info, test_info, printf, net,
              optpolicy_lr, optpolicy_rw, thresh=3, **params):
    if verbose and epoch % verbose == 0:
        train_loss, train_acc, train_nll, train_reg = train_info[-1]
        test_loss,  test_acc,  test_nll,  test_reg = test_info[-1]
        epoch_time, start_time = int(time.time() - start_time), time.time()


        ard_layers = map(lambda l: l.get_ard(thresh=thresh) if 'reg' in l.__dict__ else None,
                 lasagne.layers.get_all_layers(net))

        he = ['epo', 'upd', 'lr', 'beta', 'tr_loss', 'tr_nll', 'tr_acc',
              'te_loss', 'te_nll', 'te_acc', 'reg', 'rw', 'ard', 'sec']
        info = ('%s/%s' % (str(epoch).zfill(3) , num_epochs),
                updates,
                optpolicy_lr(epoch)[0], '\'%.2f' % optpolicy_lr(epoch)[1],
                '\'%.3f' % train_loss, train_nll, '\'%.3f'  % train_acc,
                '\'%.3f' % test_loss,  test_nll,  '\'%.3f'  % test_acc,
                train_reg,  '\'%.2f'  % optpolicy_rw(epoch),
                str(filter(None, ard_layers)).replace('\'', ''), epoch_time)

        if epoch == 0:
            printf(">> Start Learning")
            printf(tabulate([info], he, floatfmt='1.1e'))
        else:
            printf(tabulate([info], he, tablefmt="plain", floatfmt='1.1e').split('\n')[1])

    return start_time


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in xrange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def batch_iterator_train_crop_flip(data, y, batchsize, shuffle=False):
    PIXELS = 28
    PAD_CROP = 4
    n_samples = data.shape[0]
    # Shuffles indicies of training data, so we can draw batches from random indicies instead of shuffling whole data
    indx = np.random.permutation(xrange(n_samples))
    for i in range((n_samples + batchsize - 1) // batchsize):
        sl = slice(i * batchsize, (i + 1) * batchsize)
        X_batch = data[indx[sl]]
        y_batch = y[indx[sl]]

        # pad and crop settings
        trans_1 = random.randint(0, (PAD_CROP*2))
        trans_2 = random.randint(0, (PAD_CROP*2))
        crop_x1 = trans_1
        crop_x2 = (PIXELS + trans_1)
        crop_y1 = trans_2
        crop_y2 = (PIXELS + trans_2)

        # flip left-right choice
        flip_lr = random.randint(0,1)

        # set empty copy to hold augmented images so that we don't overwrite
        X_batch_aug = np.copy(X_batch)

        # for each image in the batch do the augmentation
        for j in range(X_batch.shape[0]):
            # for each image channel
            for k in range(X_batch.shape[1]):
                # pad and crop images
                img_pad = np.pad(
                    X_batch_aug[j, k], pad_width=((PAD_CROP, PAD_CROP), (PAD_CROP, PAD_CROP)), mode='constant')
                X_batch_aug[j, k] = img_pad[crop_x1:crop_x2, crop_y1:crop_y2]

                # flip left-right if chosen
                if flip_lr == 1:
                    X_batch_aug[j, k] = np.fliplr(X_batch_aug[j,k])

        # fit model on each batch
        yield X_batch_aug, y_batch


def train(net, train_fun, test_fun, up_opt, optpolicy_lr, up_rw, optpolicy_rw, data, num_epochs, batch_size,
          verbose=1, printf=print, thresh=3, da=False):
    sys.stdout.flush()
    train_info, test_info = [], []
    start_time, updates = time.time(), 0
    X_train, y_train, X_test, y_test = data

    try:
        for epoch in xrange(num_epochs+1):
            up_opt(*optpolicy_lr(epoch))
            up_rw(optpolicy_rw(epoch))
            batches, info = 0, np.zeros(4)

            itera = batch_iterator_train_crop_flip if da else iterate_minibatches

            end = time.time()
            for inputs, targets in itera(X_train, y_train, batch_size, shuffle=True):
                prev = end
                begin = time.time()
                info += train_fun(inputs, targets)
                batches += 1
                updates += 1
            train_info.append(info/batches)

            batches, info = 0, np.zeros(4)
            for inputs, targets in itera(X_test, y_test, batch_size, shuffle=False):
                info += test_fun(inputs, targets)
                batches += 1
            test_info.append(info/batches)

            start_time = iter_info(**locals())
    except KeyboardInterrupt:
        print('stop train')

    return net, train_info, test_info


def test_net(net, test_fun, data, ard=False):
    params = ll.get_all_params(net)
    paramsv = ll.get_all_param_values(net)
    save_w, atallw, batches, info  = 0, 0, 0, np.zeros(4)

    if ard:
        for i in range(len(params)):
            if params[i].name == 'W':
                ls2 = paramsv[i + 1] if len(paramsv[i + 1].shape) == 4 else paramsv[i + 2]
                log_alpha = ls2 - np.log(paramsv[i] ** 2)
                paramsv[i][log_alpha > 3] = 0

                save_w += np.sum(paramsv[i] == 0)
                atallw += paramsv[i].size

        totalp = np.sum([p.flatten().shape[0] for p in paramsv]) - atallw
        compression = 'compr_w = %s, compr_full = %s' % (save_w*1.0/atallw, save_w*1.0/totalp)
    else:
        compression = ''

    X_train, y_train, X_test, y_test = data
    for inputs, targets in iterate_minibatches(X_test, y_test, 100, shuffle=False):
        info += test_fun(inputs, targets)
        batches += 1

    return 'acc %s ' % info[1] + compression
