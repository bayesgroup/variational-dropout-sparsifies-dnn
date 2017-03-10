import theano.tensor as T

from lasagne import init
from lasagne import nonlinearities
from lasagne import updates
from lasagne.init import *
from lasagne.layers.base import Layer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import rectify
from lasagne.random import get_rng
from scipy.special import expit
from theano.sandbox.cuda import dnn
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class DenseLayer(Layer):
    def __init__(self, incoming, num_units, Wfc, nonlinearity=rectify, mnc=False, b=init.Constant(0.), **kwargs):
        super(DenseLayer, self).__init__(incoming)
        self.num_units = num_units
        self.nonlinearity = nonlinearity
        self.num_inputs = int(np.prod(self.input_shape[1:]))
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.W = self.add_param(Wfc, (self.num_inputs, self.num_units), name="W")
        if mnc:
            self.W = updates.norm_constraint(self.W, mnc)

        self.b = self.add_param(b, (num_units,), name="b", regularizable=False)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.num_units

    def get_output_for(self, input, deterministic=False, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)
        return self.get_output_for_(input, deterministic, **kwargs)

    def get_output_for_(self, input, deterministic, **kwargs):
        return self.nonlinearity(T.dot(input, self.W) + self.b)


class DenseBinaryDropOut(DenseLayer):
    def __init__(self, incoming, num_units, Wfc=Normal(), nonlinearity=rectify, p=0.5, **kwargs):
        super(DenseBinaryDropOut, self).__init__(incoming, num_units, Wfc, nonlinearity, mnc=3, **kwargs)
        self.p = p
        self.reg = True
        self.num_updates = 0
        self.name = 'BDropOut'

    def get_output_for_(self, input, deterministic, **kwargs):
        input_shape = input.shape if any(s is None for s in self.input_shape) else self.input_shape

        if not (deterministic or self.p == 0):
            input /= (1 - self.p)
            input *= self._srng.binomial(input_shape, p=1 - self.p, dtype=input.dtype)

        return self.nonlinearity(T.dot(input, self.W) + self.b)

    def eval_reg(self, **kwargs):
        return 0

    def get_ard(self, **kwargs):
        return None

    def get_reg(self):
        return str(self.p/(1-self.p))


class DenseVarDropOut(DenseLayer):
    def __init__(self, incoming, num_units, Wfc=Normal(), nonlinearity=rectify, **kwargs):
        super(DenseVarDropOut, self).__init__(incoming, num_units, Wfc, nonlinearity, mnc=3, **kwargs)
        self.reg = True
        self.name = 'VDropOut'
        self.logit_alpha = self.add_param(Constant(0), (1,), name="alpha")

    def get_output_for_(self, input, deterministic, **kwargs):
        alpha = T.nnet.sigmoid(self.logit_alpha)
        if deterministic:
            activation = T.dot(input, self.W)
        else:
            mu, si = T.dot(input, self.W), T.sqrt(T.dot(input * input, alpha * self.W * self.W) + 1e-8)
            activation = mu + self._srng.normal(mu.shape, avg=0, std=1) * si
        return self.nonlinearity(activation + self.b)

    def eval_reg(self, **kwargs):
        alpha = T.nnet.sigmoid(self.logit_alpha)
        regf = lambda a: 0.5 * T.log(a) + 1.16145124 * a + -1.50204118 * a * a + 0.58629921 * a * a * a
        reg = T.sum(regf(alpha))*self.num_inputs*self.num_units

        return -reg

    @staticmethod
    def get_ard(**kwargs):
        return 0

    def get_reg(self):
        alpha = expit(self.logit_alpha.get_value())
        return '%.1f' % alpha


class DenseVarDropOutARD(DenseLayer):
    def __init__(self, incoming, num_units, Wfc=Normal(), nonlinearity=rectify, ard_init=-10, **kwargs):
        super(DenseVarDropOutARD, self).__init__(incoming, num_units, Wfc, nonlinearity, **kwargs)
        self.reg = True
        self.log_sigma2 = self.add_param(Constant(ard_init), (self.num_inputs, self.num_units), name="ls2")
        self.name = 'VDropOutARD'

    @staticmethod
    def clip(mtx, to=8):
        mtx = T.switch(T.le(mtx, -to), -to, mtx)
        mtx = T.switch(T.ge(mtx, to), to, mtx)

        return mtx

    def get_output_for_(self, input, deterministic, train_clip=False, thresh=3, **kwargs):
        log_alpha = self.clip(self.log_sigma2 - T.log(self.W ** 2))
        clip_mask = T.ge(log_alpha, thresh)

        if deterministic:
            activation = T.dot(input, T.switch(clip_mask, 0, self.W))
        else:
            W = self.W
            if train_clip:
                W = T.switch(clip_mask, 0, self.W)
            mu = T.dot(input, W)
            si = T.sqrt(T.dot(input * input, T.exp(log_alpha) * self.W * self.W)+1e-8)
            activation = mu + self._srng.normal(mu.shape, avg=0, std=1) * si
        return self.nonlinearity(activation + self.b)

    def eval_reg(self, train_clip=False, thresh=3, **kwargs):
        log_alpha = self.clip(self.log_sigma2 - T.log(self.W ** 2))
        clip_mask = T.ge(log_alpha, thresh)
        reg = (0.5 * T.log1p(T.exp(-log_alpha)) - (0.03 + 1.0 / (1.0 + T.exp(-(1.5 * (log_alpha + 1.3)))) * 0.64))
        if train_clip:
            reg = T.switch(clip_mask, -0.67, reg)
        return reg.sum()

    def get_ard(self, thresh=3, **kwargs):
        log_alpha = self.log_sigma2.get_value() - 2 * np.log(np.abs(self.W.get_value()))
        return '%.4f' % (np.sum(log_alpha > thresh) * 1.0 / log_alpha.size)

    def get_reg(self):
        log_alpha = self.log_sigma2.get_value() - 2 * np.log(np.abs(self.W.get_value()))
        return '%.1f, %.1f' % (log_alpha.min(), log_alpha.max())


class Conv2DVarDropOutARD(ConvLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False,
                 Wconv=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=False,
                 convolution=T.nnet.conv2d, ard_init=-10, **kwargs):
        super(Conv2DVarDropOutARD, self).__init__(incoming, num_filters, filter_size,
                                                  stride, pad, untie_biases, Wconv, b,
                                                  nonlinearity, flip_filters)
        self.convolution = convolution
        self.reg = True
        self.shape = self.get_W_shape()
        self.log_sigma2 = self.add_param(Constant(ard_init), self.shape, name="ls2")
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))

    @staticmethod
    def clip(mtx, to=8):
        mtx = T.switch(T.le(mtx, -to), -to, mtx)
        mtx = T.switch(T.ge(mtx, to), to, mtx)
        return mtx

    def convolve(self, input, deterministic=False, train_clip=False, thresh=3, **kwargs):
        log_alpha = self.clip(self.log_sigma2 - T.log(self.W ** 2 + 1e-8))
        conv_mode = 'conv' if self.flip_filters else 'cross'
        border_mode = self.pad
        clip_mask = T.ge(log_alpha, thresh)
        if border_mode == 'same':
            border_mode = tuple(s // 2 for s in self.filter_size)

        if deterministic:
            conved = dnn.dnn_conv(img=input, kerns=T.switch(T.ge(log_alpha, thresh), 0, self.W),
                                  subsample=self.stride, border_mode=border_mode,
                                  conv_mode=conv_mode)
        else:
            W = self.W
            if train_clip:
                W = T.switch(clip_mask, 0, W)
            conved_mu = dnn.dnn_conv(img=input, kerns=W,
                                  subsample=self.stride, border_mode=border_mode,
                                  conv_mode=conv_mode)
            conved_si = T.sqrt(1e-8+dnn.dnn_conv(img=input * input, kerns=T.exp(log_alpha) * W * W,
                                  subsample=self.stride, border_mode=border_mode,
                                  conv_mode=conv_mode))
            conved = conved_mu + conved_si * self._srng.normal(conved_mu.shape, avg=0, std=1)
        return conved

    def eval_reg(self, train_clip=False, thresh=3, **kwargs):
        log_alpha = self.clip(self.log_sigma2 - T.log(self.W ** 2))
        clip_mask = T.ge(log_alpha, thresh)
        reg = (0.5 * T.log1p(T.exp(-log_alpha)) - (0.03 + 1.0 / (1.0 + T.exp(-(1.5 * (log_alpha + 1.3)))) * 0.64))
        if train_clip:
            reg = T.switch(clip_mask, -0.67, reg)
        return reg.sum()

    def get_ard(self, thresh=3, **kwargs):
        log_alpha = self.log_sigma2.get_value() - 2 * np.log(np.abs(self.W.get_value()))
        return '%.4f' % (np.sum(log_alpha > thresh) * 1.0 / log_alpha.size)

    def get_reg(self):
        log_alpha = self.log_sigma2.get_value() - 2 * np.log(np.abs(self.W.get_value()))
        return '%.1f, %.1f' % (log_alpha.min(), log_alpha.max())
