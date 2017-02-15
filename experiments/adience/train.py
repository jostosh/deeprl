import h5py
import tflearn
import tensorflow as tf
import argparse
from deeprl.common.hyper_parameters import HyperParameters
import pickle as pkl
import os
from deeprl.common.logger import get_log_dir
import subprocess


def get_logdir():
    hyperparameters = HyperParameters({
        'model': args.model,
        'env': 'adience',
        'frame_prediction': False,
        'feedback': False,
        'optimality_tightening': False,
        'residual_prediction': False,
    })
    logdir = get_log_dir(hyperparameters)
    os.makedirs(logdir, exist_ok=True)

    with open(os.path.join(logdir, 'hyper_parameters.pkl'), 'wb') as f:
        hp = hyperparameters.__dict__
        os.chdir(os.path.expanduser("~") + "/mproj/deeprl")
        hp.update({'git_description': subprocess.check_output(["git", "describe", "--always"]).decode('utf8').strip()})
        pkl.dump(hyperparameters.__dict__, f, pkl.HIGHEST_PROTOCOL)

    return logdir


def network_default(input):
    net = tflearn.layers.conv_2d(input, nb_filter=96, filter_size=7, activation=tf.nn.elu, padding='valid')
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
    net = tflearn.layers.estimator.regression(net, optimizer='adam')

    return net



def network_spatial_interpolation(input):
    net = tflearn.layers.conv_2d(input, nb_filter=96, filter_size=7, activation=tf.nn.elu, padding='valid')
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
    net = tflearn.layers.estimator.regression(net, optimizer='adam', batch_size=8)

    return net


def train(idx):
    h5f = h5py.File('/home/jos/datasets/aligned/fold{}.hdf5'.format(idx), 'r')
    X_train = h5f['train/images']
    y_train = h5f['train/labels']

    X_test  = h5f['test/images']
    y_test  = h5f['test/labels']

    imaug = tflearn.ImageAugmentation()
    if args.crop:
        imaug.add_random_crop((227, 227))
    imaug.add_random_flip_leftright()

    input_layer = tflearn.layers.core.input_data(shape=[None, 224, 224, 3], data_augmentation=imaug)
    network = (network_default if args.model == 'default' else network_spatial_interpolation)(input_layer)

    print(network.placeholder)

    model = tflearn.DNN(network, tensorboard_verbose=args.verbosity, tensorboard_dir=get_logdir())
    model.fit(X_train, y_train, batch_size=16, validation_set=(X_test, y_test), snapshot_step=100, show_metric=True,
              validation_batch_size=16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--model", type=str, default='default')
    parser.add_argument("--crop", dest='crop', action='store_true')
    parser.add_argument("--verbosity", type=int, default=0)
    args = parser.parse_args()

    train(args.idx)