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

from deeprl.approximators.layers import spatialsoftmax, spatial_weight_sharing, fully_connected_weight_sharing
import tensorflow as tf

import matplotlib.pyplot as plt
import scipy.misc

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')
#network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = spatial_weight_sharing(network, [2, 2], 24, 7, 1, tf.nn.relu, centroids_trainable=False, scaling=1,
                                 distance_fn='EXP', local_normalization=True, sigma_trainable=False)
visiual_summary = network.visual_summary
bias = network.b
W_list = network.W_list
network = max_pool_2d(network, 2)
#network = local_response_normalization(network)
#network = conv_2d(network, 64, 3, activation='linear', regularizer="L2")
#network = spatial_weight_sharing(network, [3, 3], 48, 3, 1, tf.nn.relu, centroids_trainable=True, scaling=.001,
#                                 distance_fn='EXP', local_normalization=True, sigma_trainable=True)
#network = max_pool_2d(network, 2)
#network = local_response_normalization(network)
#network = spatialsoftmax(network)
network = fully_connected(network, 128, activation='relu')
#network = fully_connected_weight_sharing(network, 11, 3, dimensionality='square')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='relu')
#network = fully_connected_weight_sharing(network, 15, 3, dimensionality='square')
network = dropout(network, 0.8)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in col:
    tf.add_to_collection(tf.GraphKeys.VARIABLES, x )


# Training

model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='/home/jos/mproj/deeprl/experiments/mnist_logs')


model.fit({'input': X}, {'target': Y}, n_epoch=5,
           validation_set=({'input': testX}, {'target': testY}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')

#image_sum = tf.summary.image("SpatialWeightSharingFilters", visiual_summary)
weighted_filters = model.session.run(visiual_summary)
Ws = model.session.run(W_list)
print("Biases: \n{}\n".format(model.session.run(bias)[-9*2:-9]))
print("Sigmas: \n{}\n".format(model.session.run(bias)[-9:]))
for i in range(32):
    #image = Image.fromarray(weighted_filters[i, :, :, 0]).convert('RGB')
    #image.save('mnist_logs/weighted_filters{0}.png'.format(i), format='png')
    scipy.misc.imsave('mnist_logs/weighted_filters{0}.png'.format(i), weighted_filters[i, :, :, 0])

    for j, W in enumerate(Ws):
        scipy.misc.imsave('mnist_logs/plain_filter{0}_{1}.png'.format(i, j), W[:, :, 0, i])
    #scipy.misc.imsave('mnist_logs/plain_filter{0}_0.png'.format(i), W0[:, :, 0, i])
    #scipy.misc.imsave('mnist_logs/plain_filter{0}_1.png'.format(i), W1[:, :, 0, i])
    #scipy.misc.imsave('mnist_logs/plain_filter{0}_2.png'.format(i), W2[:, :, 0, i])
    #scipy.misc.imsave('mnist_logs/plain_filter{0}_3.png'.format(i), W3[:, :, 0, i])

ax1 = plt.subplot2grid((9,9), (0,0), colspan=9, rowspan=8)
ax1.imshow(model.session.run(visiual_summary)[0, :, :, 0], cmap='gray')
print(model.session.run(bias)[-2*2:].reshape([2, 2]))
ax2 = plt.subplot2grid((9,9), (8,0), colspan=2)
ax2.imshow(W0[:, :, 0, 0], cmap='gray')
ax3 = plt.subplot2grid((9,9), (8,2), colspan=2)
ax3.imshow(W1[:, :, 0, 0], cmap='gray')
plt.show()

#model.trainer.summ_writer.add_summary(model.session.run(image_sum))
