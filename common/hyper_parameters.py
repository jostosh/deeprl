import argparse
from deeprl.approximators.nn import ModelNames

"""
For some important parameter settings checkout:
https://github.com/muupan/async-rl/wiki
"""
config1 = {
    'T_max': 80e6,
    't_max': 5,
    'gamma': 0.99,
    'learning_rate': 7e-4,
    'rms_decay': 0.99,
    'rms_epsilon': 0.1,
    'beta': 0.01,
    'frames_per_state': 4,
    'input_shape': '4,84,84',
    'env': 'Breakout-v0',
    'method': 'rlmethods',
    'model': ModelNames.A3C_FF,
    'n_threads': 8,
    'action_repeat': 4,
    'clip_rewards': False,
    'clip_advantage': False,
    'render': False
}


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

