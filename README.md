# About

The code for a paper on [Variational Dropout Sparsifies Deep Neural Networks](https://arxiv.org/abs/1701.05369). We showed that Variational Dropout leads to extremely sparse solutions both in fully-connected and convolutional layers. This effect is similar to automatic relevance determination effect, but prior distribution is fixed, so there is no additional overfitting risk. 

We visualize weights of Sparse VS LeNet-5 network and demonstrate several filters of a convolutional layer and piece of a fully-connected layer.

<p align="center">
<img src="http://ars-ashuha.ru/pdf/vdsdnn/conv.gif"/>
</p>

<p align="center">
<img src="http://ars-ashuha.ru/pdf/vdsdnn/animated_fc.gif"/>
</p>




# Technical Details

## Environment setup

```(bash)
sudo apt install virtualenv python-pip python-dev
virtualenv venv --system-site-packages
source venv/bin/activate

pip install numpy tabulate 'ipython[all]' sklearn 
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```

## Download Data

```(bash)
# Download MNIST dataset
mkdir ./data/mnist
curl -o ./data/mnist/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -o ./data/mnist/train-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -o ./data/mnist/t10k-images-idx3-ubyte.gz  http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -o ./data/mnist/t10k-labels-idx1-ubyte.gz  http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

# Download CIFAR-10 dataset
mkdir ./data/cifar10
curl -o ./data/cifar10/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf ./data/cifar10/cifar-10-python.tar.gz -C ./data/cifar10/

# Download CIFAR-100 dataset
mkdir ./data/cifar100
curl -o ./data/cifar10/cifar-100-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xvzf ./data/cifar10/cifar-100-python.tar.gz -C ./data/cifar100/
```

## Launch experiments 

```(bash)
source ~/venv/bin/activate
THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=1' ipython ./zexperiments/<experiment>.py
```
