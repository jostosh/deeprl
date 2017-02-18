import h5py
import tflearn
import tensorflow as tf
import argparse
from deeprl.common.hyper_parameters import HyperParameters
import pickle as pkl
import os
from deeprl.common.logger import get_log_dir
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import convolution2d
from deeprl.approximators.layers import spatial_weight_sharing


class LogDir(object):

    def __init__(self, model):
        self._logdir_base = self._get_logdir()

    def _get_logdir(self):
        hyperparameters = HyperParameters({
            'model': args.model,
            'env': 'adience',
            'frame_prediction': False,
            'feedback': False,
            'optimality_tightening': False,
            'residual_prediction': False,
        })
        logdir = os.path.join(get_log_dir(hyperparameters))
        os.makedirs(logdir, exist_ok=True)

        with open(os.path.join(logdir, 'hyper_parameters.pkl'), 'wb') as f:
            hp = hyperparameters.__dict__
            os.chdir(os.path.expanduser("~") + "/mproj/deeprl")
            hp.update({'git_description': subprocess.check_output(["git", "describe", "--always"]).decode('utf8').strip()})
            pkl.dump(hyperparameters.__dict__, f, pkl.HIGHEST_PROTOCOL)
        return logdir

    def crossval_dir(self, idx):
        return os.path.join(self._logdir_base, 'fold{}'.format(idx))


def get_optimizer(opt, lr):
    return   tflearn.optimizers.Adam(learning_rate=lr) if opt == 'adam' \
        else tflearn.optimizers.AdaDelta(learning_rate=lr, epsilon=1e-6, rho=0.95)


def network_default(input):
    net = tflearn.layers.conv_2d(input, nb_filter=96, filter_size=7, strides=4, activation=tf.nn.elu, padding='valid')
    net = tflearn.layers.max_pool_2d(net, kernel_size=3, strides=2, padding='valid')
    net = tflearn.layers.local_response_normalization(net)

    net = tflearn.layers.conv_2d(net, nb_filter=256, filter_size=5, activation=tf.nn.elu, padding='valid')
    net = tflearn.layers.max_pool_2d(net, kernel_size=3, strides=2, padding='valid')
    net = tflearn.layers.local_response_normalization(net)

    net = tflearn.layers.conv_2d(net, nb_filter=384, filter_size=3, activation=tf.nn.elu, padding='valid')
    net = tflearn.layers.max_pool_2d(net, kernel_size=3, strides=2, padding='valid')

    net = tflearn.layers.fully_connected(net, 512, activation=tf.nn.elu)
    net = tflearn.layers.dropout(net, keep_prob=0.5)
    net = tflearn.layers.fully_connected(net, 512, activation=tf.nn.elu)
    net = tflearn.layers.dropout(net, keep_prob=0.5)

    net = tflearn.layers.fully_connected(net, 2, activation='softmax')
    net = tflearn.layers.estimator.regression(net, optimizer=get_optimizer(args.optimizer, args.learning_rate),
                                              batch_size=args.batch_size)

    return net


def network_spatial_interpolation(input):
    net = tflearn.layers.conv_2d(input, nb_filter=96, filter_size=7, strides=4, activation=tf.nn.elu, padding='valid')
    net = tflearn.layers.max_pool_2d(net, kernel_size=3, strides=2, padding='valid')
    net = tflearn.layers.local_response_normalization(net)

    net = spatial_weight_sharing(net, n_centroids=[2, 2], n_filters=128, filter_size=5, strides=1, activation=tf.nn.elu)
    net = tflearn.layers.max_pool_2d(net, kernel_size=3, strides=2, padding='valid')
    net = tflearn.layers.local_response_normalization(net)

    net = spatial_weight_sharing(net, n_centroids=[2, 2], n_filters=192, filter_size=3, strides=1, activation=tf.nn.elu)
    net = tflearn.layers.max_pool_2d(net, kernel_size=3, strides=2, padding='valid')

    net = tflearn.layers.fully_connected(net, 512, activation=tf.nn.elu)
    net = tflearn.layers.dropout(net, keep_prob=0.5)
    net = tflearn.layers.fully_connected(net, 512, activation=tf.nn.elu)
    net = tflearn.layers.dropout(net, keep_prob=0.5)

    net = tflearn.layers.fully_connected(net, 2, activation='softmax')
    net = tflearn.layers.estimator.regression(net, optimizer=get_optimizer(args.optimizer, args.learning_rate),
                                              batch_size=args.batch_size)

    return net


def train(idx):

    h5f = h5py.File('/home/jos/datasets/aligned/fold{}.hdf5'.format(idx), 'r')
    X_train = h5f['train/images']
    y_train = h5f['train/labels']

    X_test  = h5f['test/images']
    y_test  = h5f['test/labels']

    if args.display:
        for _ in range(50):
            idx = np.random.randint(len(X_test))
            plt.imshow(X_test[idx])
            plt.title(['man', 'woman'][np.argmax(y_test[idx])])
            plt.show()

    imaug = tflearn.ImageAugmentation()
    #if args.crop:
    imaug.add_random_crop((227, 227))
    imaug.add_random_flip_leftright()

    input_layer = tflearn.layers.core.input_data(shape=[None, 227, 227, 3], data_augmentation=imaug)
    network = (network_default if args.model == 'default' else network_spatial_interpolation)(input_layer)

    model = tflearn.DNN(network, tensorboard_verbose=args.verbosity, tensorboard_dir=logdir.crossval_dir(idx))
    model.fit(X_train, y_train, batch_size=args.batch_size, validation_set=(X_test, y_test), snapshot_step=10000,
              show_metric=True, validation_batch_size=args.batch_size, n_epoch=args.n_epochs, shuffle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--model", type=str, default='default')
    parser.add_argument("--crop", dest='crop', action='store_true')
    parser.add_argument("--display", dest='display', action='store_true')
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()

    logdir = LogDir(args.model)
    train(args.idx)