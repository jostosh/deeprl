import tensorflow as tf
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
        'a3cglpq':  A3CGLPQ
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
            self.embedding_layer = net

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
            self.embedding_layer = net

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
            what = self.dnn.flatten(what, name='Flat')
            net = tf.concat(1, [what, where])
            net = self.dnn.fc_layer(256, tf.nn.relu, name='FC3', incoming=net)
            self.embedding_layer = net

        return net


class A3CLPQ(A3CFF):

    def _build_pi(self, net):
        with tf.variable_scope("Policy"):
            self.pi = self.dnn.lpq_layer(
                ppa=Config.ppa, n_classes=self.num_actions, init=Config.lpq_init, temperature=Config.lpq_temp,
                incoming=net, sim_fn=Config.lpq_distance_fn, glpq=False, name='LPQ'
            )


class A3CGLPQ(A3CFF):

    def _build_pi(self, net):
        with tf.variable_scope("Policy"):
            self.pi = self.dnn.lpq_layer(
                ppa=Config.ppa, n_classes=self.num_actions, init=Config.lpq_init, temperature=Config.lpq_temp,
                incoming=net, sim_fn=Config.lpq_distance_fn, glpq=True, name='GLPQ'
            )
