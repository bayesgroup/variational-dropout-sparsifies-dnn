import numpy as np
import theano as th

def lr_rn(epoch):
    if epoch < 10:
        lr = 0.01
    elif epoch < 80:
        lr = 0.1
    elif epoch < 120:
        lr = 0.01
    else:
        lr = 0.001
    return lr, 0.9

def lr_wrn(epoch):
    if epoch < 60:
        lr = 0.1
    elif epoch < 120:
        lr = 0.1 * 0.2
    elif epoch < 160:
        lr = 0.1 * 0.2 * 0.2
    else:
        lr = 0.1 * 0.2 * 0.2
    return lr, 0.9

def lr_linear(epoch, k):
    return max(0, np.cast[th.config.floatX](k * np.minimum(2. - 1.0 * epoch / 100., 1.))), 0.9

def lr_linear_to0(epoch, lr):
    return lr * max(0, (200. - epoch) / 200.), 0.9

def lr_const(epoch):
    return 1e-3, 0.9

def rw_linear(epoch):
    if epoch < 5:
        return 0.0
    if epoch < 20:
        return (epoch-5)/15.0
    return 1

def rw_linear_(epoch):
    if epoch < 10:
        return 0.2
    if epoch < 20:
        return 0.5
    return 1

def rw_const(epoch):
    return 1

