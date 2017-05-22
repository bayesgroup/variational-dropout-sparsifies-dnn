import os
import subprocess

def download_mnist():
    if not os.path.exists('./data'): os.mkdir('./data')
    if not os.path.exists('./data/mnist'): os.mkdir('./data/mnist')

    base, lecun = './data/mnist', 'http://yann.lecun.com/exdb/mnist'

    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz']

    print 'loading mnist'

    for file in files:
        cmd = 'curl -o {base}/{file}  {lecun}/{file}'.format(base=base, lecun=lecun, file=file)
        subprocess.call(cmd.split())

    print 'loading finish'

def download_cifar10():
    if not os.path.exists('./data'): os.mkdir('./data')
    if not os.path.exists('./data/cifar10'): os.mkdir('./data/cifar10')

    cmd = 'curl -o ./data/cifar10/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    subprocess.call(cmd.split())

    cmd = 'tar -xvzf ./data/cifar10/cifar-10-python.tar.gz -C ./data/cifar10/'
    subprocess.call(cmd.split())

def download_cifar100():
    if not os.path.exists('./data'): os.mkdir('./data')
    if not os.path.exists('./data/cifar100'): os.mkdir('./data/cifar100')

    cmd = 'curl -o ./data/cifar100/cifar-100-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    subprocess.call(cmd.split())

    cmd = 'tar -xvzf ./data/cifar100/cifar-100-python.tar.gz -C ./data/cifar100/'
    subprocess.call(cmd.split())