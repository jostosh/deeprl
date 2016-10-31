import tensorflow as tf
import tflearn
from deeprl.common.logger import logger


class ModelNames:
    a3c_ff = 'a3c_ff'
    nature = 'nature'
    small_fcn = 'small_fcn'
    a3c_lstm = 'a3c_lstm'


class ActorCriticNN(object):

    def __init__(self, num_actions, network_name, optimizer, session, hyper_parameters, global_network=None):
        self.num_actions = num_actions
        self.optimizer = optimizer
        self.global_network = global_network
        self.session = session
        self.beta = hyper_parameters.beta

        # Build computational graphs for loss, synchronization of parameters and parameter updates
        with tf.device('/cpu:0'):
            with tf.name_scope(network_name):
                self.build_network(num_actions, hyper_parameters.input_shape, network_model=hyper_parameters.model)
                with tf.name_scope('Loss'):
                    self.build_loss()

                if global_network:
                    self.build_param_sync()
                    self.build_param_update()

    def _nature_model(self):
        """
        This is the more complicated model as taken from the nature paper in 2015.
        :param network_name:   Name of the network
        :return:               The feedforward model (last hidden layer as a graph node)
        """
        with tf.name_scope('Inputs'):
            net = tf.transpose(self.inputs, [0, 2, 3, 1])

        with tf.name_scope('HiddenLayers'):
            net = tflearn.conv_2d(net, 32, 8, strides=4, activation='relu', name='Conv1')
            self._add_trainable(net)
            net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu', name='Conv2')
            self._add_trainable(net)
            net = tflearn.conv_2d(net, 64, 3, strides=1, activation='relu', name='Conv3')
            self._add_trainable(net)
            net = tflearn.flatten(net)
            net = tflearn.fully_connected(net, 512, activation='relu', name='FC4')
            self._add_trainable(net)

        return net

    def _a3c_ff(self):
        """
        This is the feedforward model taken from "Asynchronous Methods for Reinforcement Learning"
        :param network_name:    Name of the network
        :return:                The feedforward model (last hidden layer) as a graph node
        """
        with tf.name_scope('Inputs'):
            net = tf.transpose(self.inputs, [0, 2, 3, 1])

        with tf.name_scope('HiddenLayers'):
            net = tflearn.conv_2d(net, 32, 8, strides=4, activation='relu', name='Conv1')
            self._add_trainable(net)
            net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu', name='Conv2')
            self._add_trainable(net)
            net = tflearn.flatten(net)
            net = tflearn.fully_connected(net, 256, activation='relu', name='FC3')
            self._add_trainable(net)

        return net

    def _a3c_lstm(self):
        """
        This is the feedforward model taken from "Asynchronous Methods for Reinforcement Learning"
        :param network_name:    Name of the network
        :return:                The feedforward model (last hidden layer) as a graph node
        """
        with tf.name_scope('Inputs'):
            self.initial_state = tf.placeholder(tf.float32, shape=[None, 256])
            net = tf.transpose(self.inputs, [0, 2, 3, 1])
        with tf.name_scope('HiddenLayers'):
            net = tflearn.conv_2d(net, 32, 8, strides=4, activation='relu', name='Conv1')
            self._add_trainable(net)
            net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu', name='Conv2')
            self._add_trainable(net)
            net = tflearn.flatten(net)
            net = tflearn.fully_connected(net, 256, activation='relu', name='FC3')
            self._add_trainable(net)
            net = tf.expand_dims(net, 1)
            net, states = tflearn.lstm(net, 256, initial_state=self.initial_state, return_states=True, name='LSTM4')
            self._add_trainable(net)
            logger.error(type(net))
            net = tflearn.reshape(net, [-1, 256])
            self.states = states

        logger.error(type(states, states.get_shape()))
        return net

    def _add_trainable(self, layer):
        self.theta += tflearn.get_layer_variables_by_name(layer.scope[:-1])

    def _small_fcn(self):
        """
        This network works with the CartPole-v0/v1 environments (sort of)
        :param network_name:    The name of the network
        :return:                The network as a graph node
        """
        with tf.name_scope('HiddenLayers'):
            net = tflearn.fully_connected(self.inputs, 128, activation='tanh', name='FC1')
            self._add_trainable(net)
        return net

    def build_network(self, num_actions, input_shape, network_model='nature'):
        logger.debug('Input shape: {}'.format(input_shape))
        with tf.name_scope('Inputs'):
            self.inputs = tf.placeholder(tf.float32, [None] + input_shape)
        logger.info('Building network: {}'.format(network_model))

        self.theta = []

        if network_model == ModelNames.nature:
            net = self._nature_model()
        elif network_model == ModelNames.a3c_ff:
            net = self._a3c_ff()
        elif network_model == ModelNames.a3c_lstm:
            net = self._a3c_lstm()
        else:
            net = self._small_fcn()

        with tf.name_scope("Outputs"):
            with tf.name_scope("Policy"):
                self.pi = tflearn.fully_connected(net, num_actions, activation='softmax', name='pi_sa')
                self._add_trainable(self.pi)
            with tf.name_scope("Value"):
                self.value = tflearn.fully_connected(net, 1, activation='linear', name='v_s')
                self._add_trainable(self.value)

        logger.info(self.theta, [t.get_shape() for t in self.theta])


    def build_param_sync(self):
        with tf.name_scope("ParamSynchronization"):
            self.param_sync = [tf.assign(local_theta, global_theta)
                               for local_theta, global_theta in zip(self.theta, self.global_network.theta)]
            #self.param_sync = [tf.assign(global_theta, local_theta) for global_theta, local_theta in zip(self.theta, self.global_network.theta)]


    def build_param_update(self):
        with tf.name_scope("ParamUpdate"):
            gradients = [tf.clip_by_norm(grad, 40.0)
                         for grad, _ in self.optimizer.compute_gradients(self.loss, var_list=self.theta)]
            self.param_update = self.optimizer.apply_gradients(zip(gradients, self.global_network.theta))


    def build_loss(self):
        """
        This function defines the computation graph that is needed to compute the loss of the approximators.
        """

        # The n_step_return is an array of length n (where n is the batch size)
        self.n_step_returns = tf.placeholder(tf.float32, [None], name='NStepReturns')
        # The actions attribute is an array of length n
        self.actions = tf.placeholder(tf.int32, [None], name='Actions')
        # The advantage function requires a
        self.advantage_no_grad = tf.placeholder(tf.float32, [None], name='TDErrors')

        # The advantage is simply the estimated value minus the bootstrapped returns
        advantage = self.n_step_returns - self.value
        advantage_no_grad = self.n_step_returns - tf.stop_gradient(self.value)

        with tf.name_scope("PolicyLoss"):
            # action matrix is n x a where each row corresponds to a time step and each column to an action
            action_mask = tf.one_hot(self.actions, self.num_actions, 1.0, 0.0, name="ActionMask")
            # self.pi and log_pi are n x a matrices
            log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0), name="LogPi")
            # The entropy is added to encourage exploration
            entropy = -tf.reduce_sum(log_pi * self.pi, reduction_indices=1, name="Entropy")

            # Define the loss for the policy (minus is needed to perform *negative* gradient descent == gradient ascent)
            pi_loss = tf.neg(tf.reduce_sum(action_mask * log_pi, reduction_indices=1) * advantage_no_grad #self.advantage_no_grad  #advantage_no_gradient
                        + self.beta * entropy, name='PiLoss')

        with tf.name_scope("ValueLoss"):
            value_loss = tf.square(advantage)

        # We can combine the policy loss and the value loss in a single expression
        with tf.name_scope("CombinedLoss"):
            self.loss = tf.reduce_mean(pi_loss + 0.5 * value_loss)


    def get_pi(self, state):
        """
        This function returns a single array reflecting the stochastic policy pi for the given state.
        :param state: The input state
        :return: State's policy
        """
        return self.session.run(self.pi, feed_dict={self.inputs: [state]})[0]

    def get_value(self, state):
        """
        This function returns a single value that corresponds to the critic's valuation of the given state.
        :param state: The input state
        :return: State's value
        """
        return self.session.run(self.value, feed_dict={self.inputs: [state]})[0][0]

