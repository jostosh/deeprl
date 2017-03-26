import tensorflow as tf
import numpy as np
from deeprl.approximators.actorcritic import ActorCriticApproximator
from deeprl.approximators.layers.core import conv_layer, fc_layer
from deeprl.approximators.layers.sisws import spatial_weight_sharing
import abc
import tflearn


class A3CFF(ActorCriticApproximator):

    def __init__(self, hyperparameters, session, num_actions, optimizer, global_approximator, agent_name):
        super().__init__(hyperparameters, session, num_actions, optimizer, global_approximator, agent_name)

    def get_action(self, state):
        """
        This function returns a single array reflecting the stochastic policy pi for the given state.
        :param state: The input state
        :return: State's policy
        """
        pi = self.session.run(self.pi, feed_dict={self.inputs: [state]})
        action = np.random.choice(self.num_actions, p=pi[0])
        return action

    def get_value(self, state):
        """
        This function returns a single value that corresponds to the critic's valuation of the given state.
        :param state: The input state
        :return: State's value
        """
        return self.session.run(self.value, feed_dict={self.inputs: [state]})[0]

    def get_value_and_action(self, state):
        """
        Returns the value and action given a state
        :param state: State input
        :return: Value and action as float and integer, respectively
        """
        value, pi = self.session.run([self.value, self.pi], feed_dict={self.inputs: [state]})
        action = np.random.choice(self.num_actions, p=pi[0])
        return value[0], action

    def get_embedding(self, state):
        assert self.embedding_layer is not None, "No embedding layer was configured for TensorBoard embeddings"
        return self.session.run(self.embedding_layer, feed_dict={self.inputs: [state]})

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
        feed_dict = self._update_feed_dict(actions, states, values, n_step_returns, lr)
        # Update the parameters
        if include_summaries:
            _, summaries = self.session.run([self.minimize, self.merged_summaries], feed_dict=feed_dict)
            return summaries
        else:
            self.session.run(self.minimize, feed_dict=feed_dict)

    def _build_hidden_layers(self):
        """
        Builds hidden layers
        :return: Tensor Op of last layer
        """
        return self._nips_hidden_layers()[0]


class A3CSISWS(A3CFF):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_hidden_layers(self):

        with tf.name_scope(self.forward_input_scope):
            net = tf.transpose(self.inputs, [0, 2, 3, 1])

        with tf.name_scope('HiddenLayers') as scope:
            # Add first convolutional layer
            net = spatial_weight_sharing(net, 3, n_filters=32, filter_size=8, strides=4, activation=self.hp.activation,
                                         name='Conv1', centroids_trainable=True, per_feature=True)
            self._add_trainable(net)

            # Add second convolutional layer
            net = spatial_weight_sharing(net, 3, n_filters=32, filter_size=4, strides=2, activation=self.hp.activation,
                                         name='Conv2', centroids_trainable=True, per_feature=True)
            self._add_trainable(net)

            net = tflearn.flatten(net)
            net = fc_layer(net, 256, activation=self.hp.activation, name='FC3')
            self._add_trainable(net)

            self.embedding_layer = net

        return net


class A3CSISWSS(A3CFF):

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def _build_hidden_layers(self):
        with tf.name_scope(self.forward_input_scope):
            net = tf.transpose(self.inputs, [0, 2, 3, 1])

        with tf.name_scope('HiddenLayers') as scope:
            # Add first convolutional layer
            net = spatial_weight_sharing(net, [2, 2], n_filters=32, filter_size=8, strides=4,
                                         activation=self.hp.activation, name='Conv1')
            self._add_trainable(net)

            # Add second convolutional layer
            net = spatial_weight_sharing(net, [2, 2], n_filters=32, filter_size=4, strides=2,
                                         activation=self.hp.activation, name='Conv2')
            self._add_trainable(net)

            net = tflearn.flatten(net)
            net = fc_layer(net, 256, activation=self.hp.activation, name='FC3')
            self._add_trainable(net)

            self.embedding_layer = net

        return net


class A3CFFSS(A3CFF):

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def _build_hidden_layers(self):
        """
        This is the feedforward model taken from "Asynchronous Methods for Reinforcement Learning" together with a
        spatial softmax layer.
        :param network_name:    Name of the network
        :return:                The feedforward model (last hidden layer) as a graph node
        """
        with tf.name_scope('Inputs'):
            net = tf.transpose(self.inputs, [0, 2, 3, 1])

        with tf.name_scope('HiddenLayers'):
            net = conv_layer(net, 32, 5, 3, activation=self.hp.activation, name='Conv1')
            self._add_trainable(net)
            net = conv_layer(net, 64, 3, 2, activation=tf.identity, name='Conv2')
            self._add_trainable(net)
            net = spatialsoftmax(net, epsilon=self.hp.ss_epsilon, trainable_temperature=self.hp.trainable_temp,
                                 safe_softmax=self.hp.trainable_temp, use_softmax_only=self.hp.softmax_only)
            if self.hp.trainable_temp:
                self.theta += net.b
            net = fc_layer(net, 256, activation=self.hp.activation, name='FC3')
            self._add_trainable(net)
            self.embedding_layer = net

        return net