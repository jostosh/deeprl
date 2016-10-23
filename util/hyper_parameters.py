import argparse

config1 = {
    'T_max': 10000000,
    't_max': 5,
    'gamma': 0.99,
    'learning_rate': 0.001,
    'lr_decay': 0.99,
    'rms_epsilon': 0.1,
    'beta': 0.01,
    'frames_per_state': 3,
    'input_shape': '3,84,84',
    'env': 'Breakout-v0',
    'method': 'rlmethods',
    'model': 'a3c_ff',
    'n_threads': 8
}


def parse_cmd_args():
    """
    This function sets the command line arguments to look for. The defaults are given in config1 above.
    :return:
    """
    parser = argparse.ArgumentParser(description='This program applies an RL method to an OpenAI gym environment')
    [parser.add_argument('--' + name, type=type(val), default=val) for name, val in config1.items()]
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
        self.lr_decay = params.lr_decay
        self.rms_epsilon = params.rms_epsilon
        self.beta = params.beta
        self.frames_per_state = params.frames_per_state
        input_shape_str = params.input_shape
        self.input_shape = [int(elem) for elem in input_shape_str.split(',')]
        self.env = params.env
        self.model = params.model
        self.n_threads = params.n_threads

