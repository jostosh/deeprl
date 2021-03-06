import tensorflow as tf
import numpy as np
from deeprl.approximators.approximator_base import Approximator
from deeprl.common.logger import logger
import abc
from deeprl.common.config import Config


class ActorCriticApproximator(Approximator, abc.ABC):

    def get_action(self, state):
        """
        This function returns a_t single array reflecting the stochastic policy pi for the given state.
        :param state: The input state
        :return: An action
        """
        pi = self.session.run(self.pi, feed_dict=self._base_feed_dict(state))
        return self._sample_actions(pi)

    def get_value(self, state):
        """
        This function returns a_t single value that corresponds to the critic's_t valuation of the given state.
        :param state: The input state
        :return: A value estimate
        """
        return self.session.run(self.value, feed_dict=self._base_feed_dict(state))[0]

    def get_value_and_action(self, state):
        """
        Returns the value and action given a_t state
        :param state: State input
        :return: Value and action as float and integer, respectively
        """
        value, pi = self.session.run([self.value, self.pi], feed_dict=self._base_feed_dict(state))
        return value, self._sample_actions(pi)

    def _sample_actions(self, pi):
        return [np.random.choice(self.num_actions, p=p) for p in pi]

    def get_embedding(self, state):
        """ Returns an embedding vector given a_t state """
        return self.session.run(self.hidden_head, feed_dict={self.states: [state]})

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
        """
        feed_dict = self._update_feed_dict(actions, states, values, n_step_returns, lr)
        # Update the parameters
        if include_summaries:
            _, summaries = self.session.run([self.minimize, self.merged_summaries], feed_dict=feed_dict)
            return summaries
        else:
            self.session.run(self.minimize, feed_dict=feed_dict)

    @abc.abstractmethod
    def _build_hidden_layers(self):
        """ Builds hidden layers """

    def _build_loss(self):

        with tf.variable_scope('BackwardInputs'):
            # The actions attribute is an array of length n
            self.actions = tf.placeholder(tf.int32, [None], name='Actions')
            # Rewards
            self.advantage_no_grad = tf.placeholder(tf.float32, [None], name="AdvantageNoGrad")
            self.n_step_returns = tf.placeholder(tf.float32, [None], name='NStepReturns')

        with tf.variable_scope("PolicyLoss"):
            # action matrix is n x a_t where each row corresponds to a_t time step and each column to an action
            action_mask = tf.one_hot(self.actions, self.num_actions, 1.0, 0.0, name="ActionMask")
            # self.pi and log_pi are n x a_t matrices
            log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0), name="LogPi")
            # The entropy is added to encourage exploration
            entropy = -tf.reduce_sum(log_pi * self.pi, reduction_indices=1, name="Entropy")
            # Define the loss for the policy (minus is needed to perform *negative* gradient descent == gradient ascent)
            pi_loss = -(tf.reduce_sum(tf.multiply(action_mask, log_pi), reduction_indices=1) * self.advantage_no_grad
                        + Config.entropy_beta * entropy)

        with tf.variable_scope("ValueLoss"):
            # A3C originally uses a_t factor 0.5 for the value loss. The l2_loss() method already does this
            advantage = self.n_step_returns - self.value
            value_loss = tf.nn.l2_loss(advantage)

        # We can combine the policy loss and the value loss in a_t single expression
        with tf.variable_scope("CombinedLoss"):

            # Add losses
            self.loss = tf.reduce_sum(pi_loss) + value_loss

            # Add TensorBoard summaries
            self.summaries.append(tf.summary.scalar('{}/Loss'.format(self.name), self.loss))
            self.summaries.append(tf.summary.scalar('{}/MaxAbsValue'.format(self.name),
                                                    tf.reduce_max(tf.abs(self.value))))

    def _update_feed_dict(self, actions, states, values, n_step_returns, lr):
        return {
            self.actions: actions,
            self.states: states,
            self.optimizer.learning_rate: lr,
            self.n_step_returns: n_step_returns,
            self.advantage_no_grad: n_step_returns - values
        }

    def _build_network(self):
        """ Builds the full network """
        logger.info('Building network: {}'.format(Config.model))
        with tf.variable_scope('ForwardInputs'):
            self.states = tf.placeholder(tf.float32, self._input_shape(), name="StateInput")

        net = self.hidden_head = self._build_hidden_layers()
        with tf.variable_scope("Outputs"):
            self._build_pi(net)
            self._build_v(net)

        if self.name == 'Global':
            self._show_layer_overview()

    def _build_v(self, net):
        with tf.variable_scope("Value"):
            self.dnn.fc_layer(1, activation=tf.identity, name='V_s', incoming=net)
            self.value = self.dnn.flatten(name='V_s_flat')

    def _build_pi(self, net):
        with tf.variable_scope("Policy"):
            self.pi = self.dnn.fc_layer(self.num_actions, activation=tf.nn.softmax, name='pi_sa', incoming=net)
