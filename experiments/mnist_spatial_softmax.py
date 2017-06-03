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

import matplotlib.pylab as plt
from deeprl.approximators.layers import spatialsoftmax, neural_tile_coding
import tensorflow as tf
# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d(network, 32, 3, activation='elu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='linear', regularizer="L2")#, bias=False)
#ss = spatialsoftmax(tf.reshape(network, (-1, 14, 14, 64)), hierarchical=False, safe_softmax=True,
                    #trainable_temperature=True, temp_init=0.1, epsilon=0.1)
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = tflearn.flatten(network)
network, _ = neural_tile_coding(network, [16] * 16, [1024] * 16, name='ntc1')
network = dropout(network, 0.5)
network, _ = neural_tile_coding(network, [16] * 16, [128] * 16, name='ntc2')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=1e-3,
                     loss='categorical_crossentropy', name='target')


# Training
model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='/home/jos/tensorflowlogs/mnist/ss')
model.fit({'input': X}, {'target': Y}, n_epoch=10,
           validation_set=({'input': testX}, {'target': testY}), show_metric=True, run_id='convnet_mnist')

softmax_features = model.session.run(ss.sm, feed_dict={model.inputs[0]: X[:10]})

for i in range(10):
    for j in range(64):
        print(softmax_features[i, :, :, j].sum())
        #print("x: {}, y: {}".format(pos[i, j], pos[i, j+64]))
        plt.imshow(softmax_features[i, :, :, j], cmap='gray', vmin=0., vmax=1.)
        plt.show()

#print(model.session.run(ss, feed_dict={model.inputs[0]: X[:1]}))
