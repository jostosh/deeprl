import abc
import tensorflow as tf
from deeprl.approximators.dnn import DNN
from deeprl.common.logger import logger
from deeprl.common.config import Config


class Approximator(abc.ABC):

    def __init__(self, session, num_actions, optimizer, global_approximator, name, global_t=None, async=True):
        """
        Initilaizes an approximator
        :param session: TensorFlow session
        :param num_actions: The number of actions
        :param optimizer: The gradient descent optimizer
        :param global_approximator: The global approximator
        """
        self.session = session
        self.num_actions = num_actions
        self.optimizer = optimizer
        self.global_approximator = global_approximator
        self.async = async
        self.name = name
        self.layers = {}
        self.summaries = []
        self.dnn = DNN(global_t=global_t)
        self.global_t = global_t

        with tf.variable_scope(name):
            self._build_network()
        self.theta = self.dnn.theta

        if not async:
            optimizer.set_global_theta(self.theta)

        if global_approximator or not async:
            self._build_loss()
            if async:
                self.build_param_sync()
            self.build_param_update()
            self.merged_summaries = tf.summary.merge(self.summaries)

    @abc.abstractmethod
    def get_action(self, state):
        """
        This function returns a single array reflecting the stochastic policy pi for the given state.
        :param state: The input state
        :return: State's policy
        """

    @abc.abstractmethod
    def get_value(self, state):
        """
        This function returns a single value that corresponds to the critic's valuation of the given state.
        :param state: The input state
        :return: State's value
        """

    @abc.abstractmethod
    def get_value_and_action(self, state):
        """
        Returns the value and action given a state
        :param state: State input
        :return: Value and action as float and integer, respectively
        """

    @abc.abstractmethod
    def get_embedding(self, state):
        """
        Returns an embedding vector given a state
        :param state:
        """

    @abc.abstractmethod
    def update_params(self, actions, states, values, n_step_returns, lr, include_summaries, **kwargs):
        """
        Updates the global parameters
        :param actions: Actions that were chosen
        :param states: States observed
        :param values: Values estimated
        :param lr: Learning rate
        :param n_step_returns: n-step returns
        :param include_summaries: Whether to include summaries in output for TensorBoard
        :param kwargs: Other arguments
        :return:
        """

    @abc.abstractmethod
    def _build_loss(self):
        """
        Builds the loss on top of the network's output
        """

    @abc.abstractmethod
    def _build_hidden_layers(self):
        """
        Builds hidden layers
        :return: Tensor Op of last layer
        """

    @abc.abstractmethod
    def _build_network(self):
        """
        Builds the full network for approximated functions such as Q(s,a), V(s) or pi(s,a)
        """

    def reset(self):
        """
        Does everything to reset the approximator such that it is ready for a new episode
        """
        pass

    def _base_feed_dict(self, state):
        return {self.states: state}

    def build_param_sync(self):
        with tf.name_scope("ParamSynchronization"):
            self.param_sync = tf.group(*[tf.assign(local_theta, global_theta, use_locking=False)
                                         for local_theta, global_theta in zip(self.theta, self.global_approximator.theta)])

    def build_param_update(self):
        with tf.name_scope("ParamUpdate"):
            self.minimize = self.optimizer.build_update_from_vars(self.theta, self.loss)

    def synchronize_parameters(self):
        self.session.run(self.param_sync)

    def _nips_hidden_layers(self):
        with tf.variable_scope('HiddenLayers') as scope:
            self.dnn.conv_layer(32, 8, 4, tf.nn.relu, name='Conv1', incoming=self.states)
            self.dnn.conv_layer(64, 4, 2, tf.nn.relu, name='Conv2')
            net = self.dnn.fc_layer(256, tf.nn.relu, name='FC3')

        return net, scope

    def _show_layer_overview(self):
        logger.info("Layer overview:")
        for key in self.layers.keys():
            logger.info('\t' + key)

    def _input_shape(self):
        return [None, Config.im_h, Config.im_w, Config.stacked_frames]