import abc
import tensorflow as tf
from deeprl.approximators.dnn import DNN
from deeprl.common.logger import logger


class Approximator(abc.ABC):

    def __init__(self, session, num_actions, optimizer, global_approximator, name):
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
        self.name = name
        self.layers = {}
        self.embedding_layer = None
        self.states = None
        self.actions = None
        self.minimize = None
        self.summaries = []
        self.loss = None
        self.hidden_head = None
        self.loss_scope = None
        self.predicted_frame = None
        self.fp_loss_coeff = None
        self.frame_target = None
        self.param_sync = None
        self.dnn = DNN()

        with tf.variable_scope(name):
            self._build_network()
        self.theta = self.dnn.theta

        if global_approximator:
            self._build_loss()
            self.build_param_sync()
            self.build_param_update()
            self.merged_summaries = tf.summary.merge(self.summaries)

    @abc.abstractmethod
    def get_action(self, state):
        """
        This function returns a_t single array reflecting the stochastic policy pi for the given state.
        :param state: The input state
        :return: State's_t policy
        """

    @abc.abstractmethod
    def get_value(self, state):
        """
        This function returns a_t single value that corresponds to the critic's_t valuation of the given state.
        :param state: The input state
        :return: State's_t value
        """

    @abc.abstractmethod
    def get_value_and_action(self, state):
        """
        Returns the value and action given a_t state
        :param state: State input
        :return: Value and action as float and integer, respectively
        """

    @abc.abstractmethod
    def get_embedding(self, state):
        """
        Returns an embedding vector given a_t state
        :param state:
        """

    @abc.abstractmethod
    def update_params(self, actions, states, values, lr, n_step_returns, include_summaries, **kwargs):
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
        Builds the loss on top of the network's_t output
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
        Builds the full network for approximated functions such as Q(s_t,a_t), V(s_t) or pi(s_t,a_t)
        """

    def reset(self):
        """
        Does everything to reset the approximator such that it is ready for a_t new episode
        """
        pass

    def _base_feed_dict(self, state):
        return {self.states: state}

    '''
    def _add_trainable(self, layer, name=None):
        self.layers[layer.name] = layer
        if name:
            self.theta += tflearn.get_layer_variables_by_name(name)
        else:
            self.theta.append(layer.W)
            if layer.b:
                if isinstance(layer.b, list):
                    self.theta += layer.b
                else:
                    self.theta.append(layer.b)
    '''

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
            # Add first convolutional layer
            self.dnn.conv_layer(32, 8, 4, tf.nn.relu, name='Conv1', incoming=self.states)
            self.dnn.conv_layer(64, 4, 2, tf.nn.relu, name='Conv2')
            net = self.dnn.fc_layer(256, tf.nn.relu, name='FC3')
            self.embedding_layer = net

        return net, scope
