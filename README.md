# Variational Dropout Sparsifies Deep Neural Networks

### Tensorflow implementation
Google AI Research has released [State of Sparsity in Deep Neural Networks](https://arxiv.org/abs/1902.09574) - a nice large scale study of sparsification methods. The code contains an implementation of [Sparse variational dropout on Tensorflow](https://github.com/google-research/google-research/blob/master/state_of_sparsity/layers/variational_dropout/nn.py#L585).

### Play around w/ SparseVD (PyTorch)
You can play with compression of a small neural network using the following [IPython notebook @ Colab](https://colab.research.google.com/github/bayesgroup/deepbayes-2019/blob/master/seminars/day6/SparseVD-solution.ipynb), which is also available as an [assigment @ Colab](https://colab.research.google.com/github/bayesgroup/deepbayes-2019/blob/master/seminars/day6/SparseVD-assignment-colab.ipynb) from [DeepBayes](https://deepbayes.ru/) Summer School.
The code is not highly tuned but it is simple.

This repo contains the code for our ICML17 paper, [Variational Dropout Sparsifies Deep Neural Networks](https://arxiv.org/abs/1701.05369) ([talk](https://vimeo.com/238221185), [slides](https://docs.google.com/presentation/d/1Lg86MnGbksn3AtehADxSG-FcrbT2DMmGmjHUqG-EQYw/edit?usp=sharing), [poster](https://senya-ashukha.github.io/projects/svdo_icml17/svdo-poster-icml17.pdf), [blog-post](https://research.yandex.com/news/yandex-at-icml-2017-variational-dropout-sparsifies-deep-neural-networks)). 
We showed that Variational Dropout leads to extremely sparse solutions both in fully-connected and convolutional layers. 
Sparse VD reduced the number of parameters up to 280 times on LeNet architectures and up to 68 times on VGG-like networks with a negligible decrease of accuracy. 
This effect is similar to the Automatic Relevance Determination effect in empirical Bayes.
However, in Sparse VD the prior distribution remaines fixed, so there is no additional risk of overfitting.

We visualize the weights of Sparse VD LeNet-5-Caffe network and demonstrate several filters of the first convolutional layer and a piece of the fully-connected layer :)

<p align="center">
<img height="318" src="http://senya-ashukha.github.io/projects/svdo_icml17/conv.gif"/>
<img height="320" src="http://senya-ashukha.github.io/projects/svdo_icml17/fc.gif"/>
</p>

## ICML 2017 Oral Presentation by Dmitry Molchanov

[![ICML 2017 Oral Presentation by Dmitry Molchanov](http://senya-ashukha.github.io/projects/svdo_icml17/icml2017-oral.png)](https://vimeo.com/238221185)

## MNIST Experiments 

The table containes the comparison of different sparsity-inducing techniques (Pruning (Han et al., 2015b;a), DNS (Guo et al., 2016), SWS (Ullrich et al., 2017)) on LeNet architectures.
Our method provides the highest level of sparsity with a similar accuracy

| Network       | Method   | Error | Sparsity per Layer  |  Compression |
| -------------: | -------- | ----- | ------------------- | :--------------: |
|               | Original | 1.64  |                     | 1              |
|               | Pruning  | 1.59  | 92.0 − 91.0 − 74.0  | 12             |
| LeNet-300-100 | DNS      | 1.99  | 98.2 − 98.2 − 94.5  | 56             |
|               | SWS      | 1.94  |                     | 23             |
| (ours)        | SparseVD | 1.92  | 98.9 − 97.2 − 62.0  | **68**         |
||||||
|               | Original | 0.8   |                     | 1              |
|               | Pruning  | 0.77  | 34 − 88 − 92.0 − 81 | 12             |
| LeNet-5       | DNS      | 0.91  | 86 − 97 − 99.3 − 96 | 111            |
|               | SWH      | 0.97  |                     | 200            |
| (ours)        | SparseVD | 0.75  | 67 − 98 − 99.8 − 95 | **280**        |


## CIFAR Experiments

The plot contains the accuracy and sparsity level for VGG-like architectures of different sizes.
The number of neurons and filters scales as _k_.
Dense networks were trained with Binary Dropout, and Sparse VD networks were trained with Sparse Variational Dropout on all layers.
The overall sparsity level, achieved by our method, is reported as a dashed line.
The accuracy drop is negligible in most cases, and the sparsity level is high, especially in larger networks.

<p align="center">
<img height="318" src="http://senya-ashukha.github.io/projects/svdo_icml17/vgg-compression.png"/>
</p>

# Environment setup

```(bash)
sudo apt install virtualenv python-pip python-dev
virtualenv venv --system-site-packages
source venv/bin/activate

pip install numpy tabulate 'ipython[all]' sklearn matplotlib seaborn  
pip install --upgrade https://github.com/Theano/Theano/archive/rel-0.9.0.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```

# Launch experiments 

```(bash)
source ~/venv/bin/activate
cd variational-dropout-sparsifies-dnn
THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=1' ipython ./experiments/<experiment>.py
```

- If you have CuDNN problem please look at this [issue](https://github.com/ars-ashuha/variational-dropout-sparsifies-dnn/issues/3).
- This [repo](https://github.com/nnnet/Variational-dropout-sparsifies-dnn) seems to use more up-to-date libs (Python 3.5 and Theano 1.0.0).

# Further extensions

These two papers heavily rely on the Sparse Variational Dropout technique and extend it to other applications:
* [Structured Bayesian Pruning via Log-Normal Multiplicative Noise](https://arxiv.org/abs/1705.07283) provides a way to enforse _structured_ sparsity using a similar technique. This method allows to remove entire neurons and convolutional filters, which results in lighter architectures and a significant inference speed-up with standard deep learning frameworks.
* [Bayesian Compression for Natural Language Processing](https://arxiv.org/abs/1810.10927) adapts the Sparse Variational Dropout techniques for sparsification of various recurrent architectures. 

# Citation

If you found this code useful please cite our paper
```
@InProceedings{
  molchanov2017variational,
  title={Variational Dropout Sparsifies Deep Neural Networks},
  author={Dmitry Molchanov and Arsenii Ashukha and Dmitry Vetrov},
  booktitle={Proceedings of the 34th International Conference on Machine Learning},
  year={2017}
}
```
