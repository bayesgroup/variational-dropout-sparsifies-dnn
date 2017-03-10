import theano
import numpy as np
import theano.tensor as T
from lasagne import layers as ll, regularization as rg
from lasagne.objectives import categorical_crossentropy


def ell(predictions, targets):
    return -categorical_crossentropy(predictions, targets).sum()


def reg(net, train_clip=False, thresh=3):
    return T.sum([layer.eval_reg(train_clip=train_clip, thresh=thresh)
                  for layer in ll.get_all_layers(net) if 'reg' in layer.__dict__])


def nll(predictions, targets, net, batch_size, num_samples, **kwargs):
    rw = theano.shared(np.cast[theano.config.floatX](0))
    return categorical_crossentropy(predictions, targets).sum() + rw * 0, rw


def sgvlb(predictions, targets, net, batch_size, num_samples, rw=None, train_clip=False, thresh=3, **kwargs):
    if rw is None:
        rw = theano.shared(np.cast[theano.config.floatX](1))
    return -(num_samples * 1.0 / batch_size * ell(predictions, targets) - rw * reg(net, train_clip, thresh)), rw


def nll_l2(predictions, targets, net, batch_size, num_samples, rw=None, train_clip=False, thresh=3,
           weight_decay=0.00001, **kwargs):
    if rw is None:
        rw = theano.shared(np.cast[theano.config.floatX](0))

    print('Weight decay:', weight_decay)

    loss = categorical_crossentropy(predictions, targets).mean()
    loss += rg.regularize_layer_params(ll.get_all_layers(net), rg.l2) * weight_decay

    return loss, rw
