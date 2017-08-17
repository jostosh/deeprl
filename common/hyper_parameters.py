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
    'env': 'Pong',
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
    'ss_hierarchical': False,
    'optimizer': 'rmsprop',
    'beta1': 0.9,
    'beta2': 0.999,
    'adam_epsilon': 0.1,
    'safe_softmax': False,
    'logprefix': '',
    'global_clipping': False,
    'global_clip_norm': 1.0,
    'mbc': False,
    'trainable_temp': True,
    'softmax_only': False,
    'force_store': False,
    'ss_temp': 0.60,
    'policy_quantization': False,
    'ignore_annealing': False,
    'prototype_factor': 1.0,
    'weights_init': 'torch',
    'nwp': 16,
    'ppa': 16,
    'debug': False,
    'wpr': 0.0,
    'pq_sim_fn': 'euc_sq',
    'pq_soft_labels': False,
    'zpi': False,
    'value_quantization': False,
    'vp': 45,
    'pq_cpa': True,
    'value_loss_fac': 0.5,
    'pq_init_noise': 0.01,
    'pt_samples': 1000,
    'pt_sample_init': False,
    'glvq': False,
    'ppao': False,
    'ng_anneal': False,
    'tau0': 0,
    'tauN': 2.5,
    'lpq_temp': 1.0,
    'lpq_anneal_nbh': False,
    'lpq_p0': 0.9,
    'lpq_pN': 0.99,
    'ss_temp_global': False,
    'pi_loss_correct': False,
    'sisws_init': 'tflearn',
    'lpq_trainable_temp': False,
    'lws_npf': False,
    'lws_of': 32,
    'mahalanobis': False,
    'noiselevel': 0.0,
    'lpq_single_winner': False,
    'lpq_hot': True,
    'exp_beta': 0.5,
    'lpq_init': 'torch',
    'lpq_temp_exp': False,
    'lpq_gamma': 0.0,
    'lpq_init_fac': 3.16,
    'lpq_zero_clip': False
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
        self.trainable_temp = params.trainable_temp
        self.softmax_only = params.softmax_only
        self.force_store = params.force_store
        self.ss_temp = params.ss_temp
        self.ss_hierarchical = params.ss_hierarchical
        self.policy_quantization = params.policy_quantization
        self.ignore_annealing = params.ignore_annealing
        self.prototype_factor = params.prototype_factor
        self.weights_init = params.weights_init
        self.ppa = params.ppa
        self.nwp = params.nwp
        self.wpr = params.wpr
        self.pq_sim_fn = params.pq_sim_fn
        self.zpi = params.zpi
        self.pq_soft_labels = params.pq_soft_labels
        self.value_quantization = params.value_quantization
        self.vp = params.vp
        self.pq_cpa = params.pq_cpa
        self.value_loss_fac = params.value_loss_fac
        self.pq_init_noise = params.pq_init_noise
        self.pt_samples = params.pt_samples
        self.pt_sample_init = params.pt_sample_init
        self.glvq = params.glvq
        self.ppao = params.ppao
        self.ng_anneal = params.ng_anneal
        self.tau0 = params.tau0
        self.tauN = params.tauN
        self.lpq_temp = params.lpq_temp
        self.lpq_anneal_nbh = params.lpq_anneal_nbh
        self.lpq_p0 = params.lpq_p0
        self.lpq_pN = params.lpq_pN
        self.ss_temp_global = params.ss_temp_global
        self.pi_loss_correct = params.pi_loss_correct
        self.sisws_init = params.sisws_init
        self.lpq_trainable_temp = params.lpq_trainable_temp
        self.lws_npf = params.lws_npf
        self.lws_of = params.lws_of
        self.mahalanobis = params.mahalanobis
        self.noiselevel = params.noiselevel
        self.lpq_single_winner = params.lpq_single_winner
        self.lpq_hot = params.lpq_hot
        self.exp_beta = params.exp_beta
        self.lpq_init = params.lpq_init
        self.lpq_temp_exp = params.lpq_temp_exp
        self.lpq_gamma = params.lpq_gamma
        self.lpq_init_fac = params.lpq_init_fac
        self.lpq_zero_clip = params.lpq_zero_clip