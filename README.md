# About

Code for Variational Dropout Sparsifies Deep Neural Networks https://arxiv.org/abs/1701.05369

# Environment setup

```(bash)
#!/usr/bin/env bash

export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
sudo dpkg-reconfigure locales

sudo apt install virtualenv python-pip python-dev
virtualenv venv --system-site-packages
source venv/bin/activate

pip install numpy tabulate 'ipython[all]' sklearn 
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```

# Download Data

```(bash)
#!/bin/bash

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

# Launch experiments 

```(bash)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/home/ashuha/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/home/ashuha/cuda/include:$CPATH
export LIBRARY_PATH=/home/ashuha/cuda/lib64:$LD_LIBRARY_PATH 

source ~/venv/bin/activate
THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=1' ipython ./zexperiments/test.py 
```