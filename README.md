# Variational Dropout Sparsifies Deep Neural Networks

The code for ICML'17 paper on [Variational Dropout Sparsifies Deep Neural Networks](https://arxiv.org/abs/1701.05369). 
We showed that Variational Dropout leads to extremely sparse solutions both in fully-connected and convolutional layers. 
This effect is similar to automatic relevance determination effect, but prior distribution is fixed, so there is no additional overfitting risk. 

We visualize weights of Sparse VD LeNet-5-Caffe network and demonstrate several filters of the first convolutional layer and the piece of a fully-connected layer :) 

<p align="center">
<img height="318" src="http://ars-ashuha.ru/pdf/vdsdnn/conv.gif"/>
<img height="320" src="http://ars-ashuha.ru/pdf/vdsdnn/animated_fc.gif"/>
</p>


### Environment setup

```(bash)
sudo apt install virtualenv python-pip python-dev
virtualenv venv --system-site-packages
source venv/bin/activate

pip install numpy tabulate 'ipython[all]' sklearn matplotlib seaborn  
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```

### Launch experiments 

```(bash)
source ~/venv/bin/activate
cd variational-dropout-sparsifies-dnn
THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=1' ipython ./experiments/<experiment>.py
```
