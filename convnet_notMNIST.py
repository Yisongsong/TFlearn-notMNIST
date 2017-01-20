# -*- coding: utf-8 -*-

""" Convolutional Neural Network for MNIST dataset classification task.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Data loading and preprocessing
import notMNIST_data as notMNIST
X, Y, testX, testY, validaX,validaY = notMNIST.load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d(network, 16, 5,strides =2, activation='relu', regularizer="L2",weights_init= "truncated_normal")


network = conv_2d(network, 16, 5,strides =2, activation='relu', regularizer="L2",weights_init= "truncated_normal")



network = local_response_normalization(network)

network = fully_connected(network, 64, activation='relu')


network = fully_connected(network, 10, activation='softmax')

network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=20,
           validation_set=({'input': testX}, {'target': testY}),
         show_metric=True, run_id='convnet_notmnist')
