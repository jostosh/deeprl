import tensorflow as tf
import numpy as np
from deeprl.approximators.approximator import Approximator
from deeprl.common.logger import logger
from deeprl.approximators.layers.core import conv_layer, fc_layer
import abc
import tflearn


class ActorCriticApproximator(Approximator, abc.ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = None
        self.pi = None
        self.n_step_returns = None
        self.advantage = None
        self.advantage_no_grad = None

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
    def _build_hidden_layers(self):
        """
        Builds hidden layers
        :return: Tensor Op of last layer
        """

    def _build_loss(self):

        with tf.name_scope('BackwardInputs'):
            # The actions attribute is an array of length n
            self.actions = tf.placeholder(tf.int32, [None], name='Actions')
            # Rewards
            self.advantage_no_grad = tf.placeholder(tf.float32, [None], name="AdvantageNoGrad")
            self.n_step_returns = tf.placeholder(tf.float32, [None], name='NStepReturns')

        if self.hp.clip_advantage:
            # I empirically found that it might help to clip the advantage that is used for the policy loss. This might
            # improve stability and consistency of the gradients
            logger.info("Clipping advantage in graph")
            self.advantage_no_grad = tf.clip_by_value(self.advantage_no_grad, -1., 1., name="ClippedAdvantage")

        with tf.name_scope("PolicyLoss"):
            # action matrix is n x a where each row corresponds to a time step and each column to an action
            action_mask = tf.one_hot(self.actions, self.num_actions, 1.0, 0.0, name="ActionMask")
            # self.pi and log_pi are n x a matrices
            log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0), name="LogPi")
            # The entropy is added to encourage exploration
            entropy = -tf.reduce_sum(log_pi * self.pi, reduction_indices=1, name="Entropy")
            # Define the loss for the policy (minus is needed to perform *negative* gradient descent == gradient ascent)
            pi_loss = -(tf.reduce_sum(tf.mul(action_mask, log_pi), reduction_indices=1) * self.advantage_no_grad
                        + self.beta * entropy)

        with tf.name_scope("ValueLoss"):
            # A3C originally uses a factor 0.5 for the value loss. The l2_loss() method already does this
            advantage = self.n_step_returns - self.value
            value_loss = tf.square(advantage)
            if self.hp.optimality_tightening:
                self.upper_limits = tf.placeholder(tf.float32, [None], name='UpperLimits')
                self.lower_limits = tf.placeholder(tf.float32, [None], name='LowerLimits')
                value_loss += self.hp.otc * (tf.nn.relu(self.lower_limits - self.value) ** 2 +
                                             tf.nn.relu(self.value - self.upper_limits) ** 2)
                value_loss /= (self.hp.otc + 1)

        # We can combine the policy loss and the value loss in a single expression
        with tf.name_scope("CombinedLoss"):

            # Add losses and
            self.loss = tf.reduce_sum(pi_loss + 0.5 * value_loss)

            # Add TensorBoard summaries
            self.summaries.append(tf.summary.scalar('{}/Loss'.format(self.agent_name), self.loss))
            self.summaries.append(tf.summary.scalar('{}/MaxAbsValue'.format(self.agent_name),
                                                    tf.reduce_max(tf.abs(self.value))))

    def _update_feed_dict(self, actions, states, values, n_step_returns, lr, last_state):
        return {
            self.actions: actions,
            self.inputs: states,
            self.optimizer.learning_rate: lr,
            self.n_step_returns: n_step_returns,
            self.advantage_no_grad: n_step_returns - values
        }.update({
            self.frame_target: [s[-1:, :, :] for s in states[1:]] + [last_state[-1:, :, :]],
            self.fp_loss_coeff: self.hp.fplc
        } if self.hp.frame_prediction else {})

    def _build_network(self):
        logger.debug('Input shape: {}'.format(self.hp.input_shape))
        with tf.name_scope('ForwardInputs') as scope:
            self.forward_input_scope = scope
            self.inputs = tf.placeholder(tf.float32, [None] + self.hp.input_shape, name="StateInput")
        logger.info('Building network: {}'.format(self.hp.model))

        net = self._build_hidden_layers()
        with tf.name_scope("Outputs"):
            with tf.name_scope("Policy"):
                if self.hp.safe_softmax:
                    self.pi = fc_layer(net, self.num_actions, activation='linear', name='pi_sa')
                    self._add_trainable(self.pi)
                    self.pi -= tf.stop_gradient(tf.expand_dims(tf.reduce_max(self.pi, reduction_indices=[1]), 1))
                    self.pi = tf.nn.softmax(self.pi)
                else:
                    self.pi = fc_layer(net, self.num_actions, activation='softmax', name='pi_sa')
                    self._add_trainable(self.pi)
            with tf.name_scope("Value"):
                if self.hp.pwv:
                    q_val = fc_layer(net, self.num_actions, activation='linear', name='q_sa')
                    self._add_trainable(q_val)
                    self.value = tf.reduce_sum(tf.mul(q_val, tf.stop_gradient(self.pi)),
                                               reduction_indices=1, name='v_s')
                else:
                    self.value = fc_layer(net, 1, activation='linear', name='v_s')
                    self._add_trainable(self.value)
                self.value = tflearn.reshape(self.value, [-1], 'FlattenedValue')

        if self.agent_name == 'GLOBAL':
            logger.info("Layer overview:")
            for key in self.layers.keys():
                logger.info('\t' + key)
        self.hidden_head = net
