import tensorflow as tf
import numpy as np
from deeprl.approximators.actorcritic import ActorCriticApproximator
from deeprl.common.config import Config
from deeprl.common.logger import logger


def get_approximator(*args, **kwargs):
    logger.info("Loading {}".format(Config.model))
    return {
        'a3cff':    A3CFF,
        'a3clws':   A3CLWS,
        'a3css':    A3CSS,
        'a3cww':    A3CWW,
        'a3clpq':   A3CLPQ,
        'a3cglpq':  A3CGLPQ,
        'a3clpqld': A3CLPQLD
    }[Config.model](*args, **kwargs)


class A3CFF(ActorCriticApproximator):
    def _build_hidden_layers(self):
        """ Builds hidden layers """
        return self._nips_hidden_layers()[0]


class A3CLWS(ActorCriticApproximator):
    def _build_hidden_layers(self):
        """ Builds hidden layers """
        with tf.variable_scope('HiddenLayers') as scope:
            # Add first convolutional layer
            self.dnn.local_weight_sharing(3, n_filters=32, filter_size=8, strides=4, activation=tf.nn.relu,
                                          name='Conv1', centroids_trainable=True, per_feature=True,
                                          incoming=self.states)
            self.dnn.local_weight_sharing(3, n_filters=64, filter_size=4, strides=2, activation=tf.nn.relu,
                                          name='Conv2', centroids_trainable=True, per_feature=True)
            net = self.dnn.fc_layer(256, tf.nn.relu, 'FC3')

        return net


class A3CSS(ActorCriticApproximator):
    def _build_hidden_layers(self):
        """
        This is the feedforward model taken from "Asynchronous Methods for Reinforcement Learning" together with a_t
        spatial softmax layer.
        :param network_name:    Name of the network
        :return:                The feedforward model (last hidden layer) as a_t graph node
        """
        with tf.variable_scope('HiddenLayers'):
            self.dnn.conv_layer(32, 5, 3, tf.nn.relu, name='Conv1', incoming=self.states)
            self.dnn.conv_layer(64, 3, 2, tf.identity, name='Conv2')
            self.dnn.spatial_softmax()
            net = self.dnn.fc_layer(256, tf.nn.relu, name='FC3')

        return net


class A3CWW(ActorCriticApproximator):
    def _build_hidden_layers(self):
        """
        This is the feedforward model taken from "Asynchronous Methods for Reinforcement Learning" together with a_t
        spatial softmax layer.
        :param network_name:    Name of the network
        :return:                The feedforward model (last hidden layer) as a_t graph node
        """

        with tf.variable_scope('HiddenLayers'):
            self.dnn.conv_layer(32, 5, 3, activation=tf.nn.relu, name='Conv1', incoming=self.states)
            net = self.dnn.conv_layer(64, 3, 2, activation=tf.nn.relu, name='Conv2')
            what, where = tf.split(3, 2, net)
            where = self.dnn.spatial_softmax(incoming=where)
            what = tf.nn.relu(what)
            what = self.dnn.flatten(incoming=what, name='Flat')
            net = tf.concat(1, [what, where])
            net = self.dnn.fc_layer(256, tf.nn.relu, name='FC3', incoming=net)

        return net


class A3CLPQ(A3CFF):

    def _build_pi(self, net):
        with tf.variable_scope("Policy"):
            temperature = 1/2 * np.log(-Config.lpq_p0 * (self.num_actions - 1) / (Config.lpq_p0 - 1))
            self.pi = self.dnn.lpq_layer(
                ppa=Config.ppa, n_classes=self.num_actions, temperature=temp,
                incoming=net, sim_fn=Config.lpq_distance_fn, glpq=False, name='LPQ'
            )


class A3CGLPQ(A3CFF):

    def _build_pi(self, net):
        with tf.variable_scope("Policy"):
            self.lpq_temp = 1/2 * np.log(-Config.lpq_p0 * (self.num_actions - 1) / (Config.lpq_p0 - 1))
            self.pi = self.dnn.lpq_layer(
                ppa=Config.ppa, n_classes=self.num_actions, temperature=self.lpq_temp,
                incoming=net, sim_fn=Config.lpq_distance_fn, glpq=True, name='GLPQ'
            )

    def _build_pi_loss(self, action_mask, entropy, log_pi):
        return -(tf.reduce_sum(tf.multiply(action_mask, log_pi), reduction_indices=1) * self.advantage_no_grad
                 + Config.entropy_beta * entropy) / self.lpq_temp


class A3CLPQLD(A3CFF):

    def _build_pi(self, net):
        with tf.variable_scope("Policy"):
            t0 = 1/2 * np.log(-Config.lpq_p0 * (self.num_actions - 1) / (Config.lpq_p0 - 1))
            tN = 1/2 * np.log(-Config.lpq_pN * (self.num_actions - 1) / (Config.lpq_pN - 1))

            self.temp_var = tf.to_float(t0 + self.global_t / Config.T_max * (tN - t0))

            ld = self.dnn.fc_layer(Config.lpq_d, activation=tf.identity, name="LD", incoming=net)
            self.pi = self.dnn.lpq_layer(
                ppa=Config.ppa, n_classes=self.num_actions, temperature=self.temp_var,
                incoming=ld, sim_fn=Config.lpq_distance_fn, glpq=False, name='LPQ'
            )
