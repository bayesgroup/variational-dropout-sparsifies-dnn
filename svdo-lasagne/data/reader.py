import cPickle as pickle
import gzip
import os

import numpy as np
import theano
from scipy import linalg
from theano import tensor as T

from data.downloader import download_mnist, download_cifar10, download_cifar100


class ZCA(object):
    def __init__(self, regularization=1e-5, x=None):
        self.regularization = regularization
        if x is not None:
            self.fit(x)

    def fit(self, x):
        s = x.shape
        x = x.copy().reshape((s[0], np.prod(s[1:])))
        m = np.mean(x, axis=0)
        x -= m
        sigma = np.dot(x.T, x) / x.shape[0]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1. / np.sqrt(S + self.regularization)))
        tmp2 = np.dot(U, np.diag(np.sqrt(S + self.regularization)))
        self.ZCA_mat = theano.shared(np.dot(tmp, U.T).astype(theano.config.floatX))
        self.inv_ZCA_mat = theano.shared(np.dot(tmp2, U.T).astype(theano.config.floatX))
        self.mean = theano.shared(m.astype(theano.config.floatX))

    def apply(self, x):
        s = x.shape
        if isinstance(x, np.ndarray):
            return np.dot(x.reshape((s[0], np.prod(s[1:]))) - self.mean.get_value(), self.ZCA_mat.get_value()).reshape(
                s)
        elif isinstance(x, T.TensorVariable):
            return T.dot(x.flatten(2) - self.mean.dimshuffle('x', 0), self.ZCA_mat).reshape(s)
        else:
            raise NotImplementedError("Whitening only implemented for numpy arrays or Theano TensorVariables")

    def invert(self, x):
        s = x.shape
        if isinstance(x, np.ndarray):
            return (
            np.dot(x.reshape((s[0], np.prod(s[1:]))), self.inv_ZCA_mat.get_value()) + self.mean.get_value()).reshape(s)
        elif isinstance(x, T.TensorVariable):
            return (T.dot(x.flatten(2), self.inv_ZCA_mat) + self.mean.dimshuffle('x', 0)).reshape(s)
        else:
            raise NotImplementedError("Whitening only implemented for numpy arrays or Theano TensorVariables")

def load(dataset):
    if dataset == 'mnist':
        return load_mnist()
    if dataset == 'mnist-random':
        return load_mnist_random()
    if dataset == 'cifar10':
        return load_cifar10()
    if dataset == 'cifar10-random':
        return load_cifar10_random()
    if dataset == 'cifar100':
        return load_cifar100()

    raise Exception('Load of %s not implimented yet' % dataset)

def load_mnist():
    """
    load_mnist taken from https://github.com/Lasagne/Lasagne/blob/master/examples/images.py
    :param base: base path to images dataset
    """

    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    base = './data/mnist'

    if not os.path.exists(base):
        download_mnist()

    # We can now download and read the training and test set image and labels.
    X_train = load_mnist_images(base + '/train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(base + '/train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(base + '/t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(base + '/t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    return (X_train, y_train, X_test, y_test), X_train.shape[0], X_test.shape[0], (None, 1, 28, 28), 10

def load_mnist_random(base='./data/mnist'):
    X_train, y_train, X_test, y_test = load_mnist(base)[0]
    np.random.seed(74632)
    y_train = np.random.choice(10, len(y_train))
    return (X_train, y_train, X_test, y_test), X_train.shape[0], X_test.shape[0], (None, 1, 28, 28), 10

def load_cifar10():
    def load_CIFAR_batch(filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f)
            Y = np.array(datadict['labels'])
            X = datadict['data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            return X, Y

    def load_CIFAR10(ROOT):
        xs, ys = [], []
        for b in range(1, 6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b,))
            X, Y = load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr, Ytr = np.concatenate(xs), np.concatenate(ys)
        del X, Y
        Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
        return Xtr, Ytr, Xte, Yte

    base = './data/cifar10'
    if not os.path.exists(base):
        download_cifar10()


    # Load the raw CIFAR-10 data
    cifar10_dir = os.path.join(base, 'cifar-10-batches-py')
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Normalize the data: subtract the mean image
    whitener = ZCA(x=X_train)
    X_train = whitener.apply(X_train)
    X_test = whitener.apply(X_test)

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    return (X_train, y_train, X_test, y_test), X_train.shape[0], X_test.shape[0], (None, 3, 32, 32), 10

def load_cifar100():
    def load_CIFAR_batch(filename, num):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = pickle.load(f)
            X = datadict['data']
            Y = datadict['coarse_labels']
            X = X.reshape(num, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            Y = np.array(Y)
            return X, Y

    base = './data/cifar100/'
    if not os.path.exists(base):
        download_cifar100()

    Xtr, Ytr = load_CIFAR_batch(os.path.join(base+'/cifar-100-python', 'train'), 50000)
    Xte, Yte = load_CIFAR_batch(os.path.join(base+'/cifar-100-python', 'test'), 10000)

    # Normalize the data: subtract the mean image
    whitener = ZCA(x=Xtr)
    Xtr = whitener.apply(Xtr)
    Xte = whitener.apply(Xte)

    # Transpose so that channels come first
    Xtr = Xtr.transpose(0, 3, 1, 2).copy()
    Xte = Xte.transpose(0, 3, 1, 2).copy()

    return (Xtr, Ytr, Xte, Yte), X_train.shape[0], X_test.shape[0], (None, 3, 32, 32), 10

def load_cifar10_random(base='./data/cifar10'):
    X_train, y_train, X_test, y_test = load_cifar10(base)[0]
    np.random.seed(74632)
    y_train = np.random.choice(10, len(y_train))
    return (X_train, y_train, X_test, y_test), X_train.shape[0], X_test.shape[0], (None, 3, 32, 32), 10

def load_cifar100(base='./data/cifar100/cifar-100-python/'):
    def load_CIFAR_batch(filename, num):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = pickle.load(f)
            X = datadict['data']
            Y = datadict['coarse_labels']
            X = X.reshape(num, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            Y = np.array(Y)
            return X, Y

    Xtr, Ytr = load_CIFAR_batch(os.path.join(base, 'train'), 50000)
    Xte, Yte = load_CIFAR_batch(os.path.join(base, 'test'), 10000)

    # Normalize the data: subtract the mean image
    whitener = ZCA(x=Xtr)
    Xtr = whitener.apply(Xtr)
    Xte = whitener.apply(Xte)

    # Transpose so that channels come first
    Xtr = Xtr.transpose(0, 3, 1, 2).copy()
    Xte = Xte.transpose(0, 3, 1, 2).copy()

    return (Xtr, Ytr, Xte, Yte), Xtr.shape[0], Xte.shape[0], (None, 3, 32, 32), 100


