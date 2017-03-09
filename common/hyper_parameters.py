import argparse
from deeprl.approximators.nn import ModelNames
import os
import numpy as np
import itertools
from deeprl.common.logger import logger
import tensorflow as tf

"""
For some important parameter settings check out:
https://github.com/muupan/async-rl/wiki
"""

config1 = {
    'T_max': 1e8,
    't_max': 20,
    'gamma': 0.99,
    'learning_rate': 7e-4,
    'rms_decay': 0.99,
    'rms_epsilon': 0.1,
    'beta': 0.01,
    'frames_per_state': 4,
    'input_shape': '4,84,84',
    'env': 'Pong-v0',
    'model': ModelNames.A3C_FF,
    'n_threads': 8,
    'action_repeat': 4,
    'clip_rewards': True,
    'clip_advantage': False,
    'render': False,
    'pwv': False,
    'frame_prediction': False,
    'logdir': os.path.expanduser("~/tensorflowlogs/mpi"),
    'residual_prediction': False,
    'evaluation_interval': 1000000,
    'optimality_tightening': False,
    'param_sweep': "",
    'score_at_10m': -10.,
    'fplc': 1.0,
    'otc': 0.25,
    'feedback': False,
    'fp_decay': 0.99999,
    'activation': 'relu',
    'ss_epsilon': 0.5,
    'optimizer': 'rmsprop',
    'beta1': 0.9,
    'beta2': 0.999,
    'adam_epsilon': 0.1,
    'safe_softmax': False,
    'logprefix': '',
    'global_clipping': False,
    'global_clip_norm': 1.0,
    'mbc': False
}

param_sweep = {
    'learning_rate': [5e-5, 1e-4, 3.5e-4, 7e-4, 1e-3],
    'fplc': [1e-4, 1e-3, 1e-2],
    'otc': [0.5, 1.0, 2.0]
}

activation_fn = {
    'relu': tf.nn.relu,
    'elu': tf.nn.elu
}


def set_param_sweep(sweep_epoch, hp):
    sweeping_parameters = sorted(hp.param_sweep.split(','))
    all_combinations = itertools.product(*[param_sweep[p] for p in sweeping_parameters])

    for pidx, p in enumerate(sweeping_parameters):
        hp.__dict__[p] = all_combinations[sweep_epoch][pidx]

    logger.info("Set new hyperparameters: {}".format(hp))


def parse_cmd_args():
    """
    This function sets the command line arguments to look for. The defaults are given in config1 above.
    :return:
    """
    parser = argparse.ArgumentParser(description='This program applies an RL method to an OpenAI gym environment')
    for name, val in config1.items():
        if type(val) is bool:
            parser.add_argument('--' + name, action='store_true', dest=name)
            parser.add_argument('--not_' + name, action='store_false', dest=name)
            parser.set_defaults(**{name: val})
        else:
            parser.add_argument('--' + name, type=type(val), default=val)

    args = parser.parse_args()
    return args


class HyperParameters(object):

    def __init__(self, params):
        """
        This class can be instantiated using a dict containing the parameters. If the param object is not a dict, the
        constructor assumes that it contains the parameters listed below.
        :param params: object containing the parameters
        """
        if isinstance(params, dict):
            self.__dict__.update(**params)
            return

        self.T_max = params.T_max
        self.t_max = params.t_max
        self.gamma = params.gamma
        self.learning_rate = params.learning_rate
        self.rms_decay = params.rms_decay
        self.rms_epsilon = params.rms_epsilon
        self.beta = params.beta
        self.frames_per_state = params.frames_per_state
        input_shape_str = params.input_shape
        self.input_shape = [int(elem) for elem in input_shape_str.split(',')]
        self.env = params.env
        self.model = params.model
        self.n_threads = params.n_threads
        self.action_repeat = params.action_repeat
        self.clip_rewards = params.clip_rewards
        self.clip_advantage = params.clip_advantage
        self.render = params.render
        self.policy_weighted_val = params.pwv
        self.frame_prediction = params.frame_prediction
        self.logdir = params.logdir
        self.residual_prediction = params.residual_prediction
        self.evaluation_interval = params.evaluation_interval
        self.optimality_tightening = params.optimality_tightening
        self.otc = params.otc
        self.fplc = params.fplc
        self.param_sweep = params.param_sweep
        self.score_at_10m = params.score_at_10m
        self.feedback = params.feedback
        self.fp_decay = params.fp_decay
        self.activation = activation_fn[params.activation]
        self.ss_epsilon = params.ss_epsilon
        self.optimizer = params.optimizer
        self.beta1 = params.beta1
        self.beta2 = params.beta2
        self.safe_softmax = params.safe_softmax
        self.logprefix = params.logprefix
        self.global_clipping = params.global_clipping
        self.global_clip_norm = params.global_clip_norm
        self.mbc = params.mbc
        self.adam_epsilon = params.adam_epsilon