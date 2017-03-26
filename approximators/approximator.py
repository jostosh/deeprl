import abc
import tensorflow as tf
from deeprl.common.logger import logger
from deeprl.approximators.layers.core import conv_layer, fc_layer
import tflearn


class Approximator(abc.ABC):

    def __init__(self, hyperparameters, session, num_actions, optimizer, global_approximator, agent_name, **kwargs):
        """
        Initilaizes an approximator
        :param hyperparameters: Hyperparameter object
        :param session: TensorFlow session
        :param num_actions: The number of actions
        :param optimizer: The gradient descent optimizer
        :param global_approximator: The global approximator
        """
        self.hp = hyperparameters
        self.session = session
        self.num_actions = num_actions
        self.optimizer = optimizer
        if global_approximator:
            self.global_approximator = global_approximator
        else:
            self.global_approximator = \
                self if 'single_approximator' in kwargs and kwargs['single_approximator'] else None
        self.agent_name = agent_name
        self.theta = []
        self.layers = {}
        self.embedding_layer = None
        self.inputs = None
        self.actions = None
        self.minimize = None
        self.merged_summaries = None
        self.summaries = []
        self.forward_input_scope = None
        self.loss = None
        self.hidden_head = None
        self.loss_scope = None
        self._conv1 = self._conv2 = None
        self.predicted_frame = None
        self.fp_loss_coeff = None
        self.frame_target = None
        self.param_sync = None

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

    def build_param_sync(self):
        with tf.name_scope("ParamSynchronization"):
            self.param_sync = tf.group(*[tf.assign(local_theta, global_theta, use_locking=False)
                                         for local_theta, global_theta in zip(self.theta, self.global_approximator.theta)])

    def build_param_update(self):
        with tf.name_scope("ParamUpdate"):
            self.minimize = self.optimizer.build_update_from_vars(self.theta, self.loss)

    def synchronize_parameters(self):
        self.session.run(self.param_sync)

    def decoding_network(self, incoming, conv1, conv2):
        with tf.name_scope("DecodingNetwork"):
            # First we reshape the embedding into a volume with the shape of conv2
            net = tflearn.reshape(incoming, [-1] + conv2.get_shape().as_list()[1:], 'TransformationReshaped')
            if self.hp.residual_prediction:
                net += conv2
            net = self.hp.activation(net, 'TransformationAct')

            # Then we perform a conv_2d_transpose (this is sometimes referred to as a DeConvolution layer)
            net = tflearn.conv_2d_transpose(net, 32, 4, strides=2, activation='linear',
                                            output_shape=conv1.get_shape().as_list()[1:],
                                            weight_decay=0., padding='valid', name='DeConv2')
            logger.warn("First deconv shape: {}".format(net.get_shape().as_list()))
            self._add_trainable(net)
            if self.hp.residual_prediction:
                net += conv1
            net = self.hp.activation(net, name='DeConv2Act')

            # Then we do the latter again
            net = tflearn.conv_2d_transpose(net, 1, 8, strides=4, activation='linear',
                                            output_shape=[84, 84, 1], padding='valid', weight_decay=0., name='DeConv1')
            logger.warn("First deconv shape: {}".format(net.get_shape().as_list()))
            self._add_trainable(net)
            net = self.hp.activation(net, name='DeConv1Act')
        return net

    def build_frame_predictor(self):

        with tf.name_scope("EncodingHead"):
            # Embed those actions
            action_one_hot = tflearn.one_hot_encoding(self.actions, self.num_actions)
            action_embedding = tflearn.fully_connected(action_one_hot, 256, weight_decay=0.0, bias=False,
                                                       name='ActionEmbedding', activation='linear')
            self._add_trainable(action_embedding)

            # Embed the hidden layer head
            encoding = tflearn.fully_connected(self.hidden_head, 256, weight_decay=0.0, bias=False,
                                               name='EncodingEmbedding', activation='linear')
            self._add_trainable(encoding)

            # We will use the linear conv activation of conv1 and conv2 and inject those in their mirrored decoding layers
            conv1 = self._conv1
            conv2 = self._conv2

            # Now we can compute the 'transformation layer' which we will be put into the decoding stream
            transformation = tflearn.fully_connected(tf.mul(action_embedding, encoding),
                                                     np.prod(conv2.get_shape().as_list()[1:]), weight_decay=0.0,
                                                     activation='linear', name='Transformation')
            self._add_trainable(transformation)

        decoded_prediction = self.decoding_network(transformation, conv1, conv2)

        with tf.name_scope("FramePrediction"):
            self.predicted_frame = tf.transpose(decoded_prediction, [0, 3, 1, 2]) \
                                   + self.inputs[:, -1:, :, :]
            self.frame_target = tf.placeholder(tf.float32, [None, 1] + self.input_shape[1:])

        with tf.name_scope(self.loss_scope):
            frame_prediction_loss = tf.nn.l2_loss(self.frame_target[:, :, 2:-2, 2:-2] - self.predicted_frame[:, :, 2:-2, 2:-2],
                                                  name='FramePredictionLoss')
            self.summaries.append(tf.summary.scalar('{}/FramePredictionLoss'.format(self.agent_name),
                                                    frame_prediction_loss))

        self.fp_loss_coeff = tf.placeholder(tf.float32)
        self.loss += self.fp_loss_coeff * frame_prediction_loss

    def _nips_hidden_layers(self):
        with tf.name_scope(self.forward_input_scope):
            net = tf.transpose(self.inputs, [0, 2, 3, 1])

        with tf.name_scope('HiddenLayers') as scope:
            # Add first convolutional layer
            self._conv1 = conv_layer(net, 32, 8, 4, activation='linear', name='Conv1')

            self._add_trainable(self._conv1)
            net = self.hp.activation(self._conv1)

            # Add second convolutional layer
            self._conv2 = conv_layer(net, 64, 4, 2, activation='linear', name='Conv2')
            self._add_trainable(self._conv2)
            net = self.hp.activation(self._conv2)

            net = tflearn.flatten(net)
            net = fc_layer(net, 256, activation=self.hp.activation, name='FC3')
            self._add_trainable(net)
            self.embedding_layer = net

        return net, scope
