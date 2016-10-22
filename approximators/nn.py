import tensorflow as tf
import tflearn
from deeprl.util.logger import logger


class ModelNames:
    a3c_ff = 'a3c_ff'
    nature = 'nature'
    small_fcn = 'small_fcn'


class ActorCriticNN(object):

    def __init__(self, num_actions, network_name, optimizer, session, hyper_parameters, global_network=None):
        self.num_actions = num_actions
        self.optimizer = optimizer
        self.global_network = global_network
        self.session = session
        self.beta = hyper_parameters.beta

        # Build computational graphs for loss, synchronization of parameters and parameter updates
        with tf.device('/cpu:0'):
            self.build_network(num_actions, hyper_parameters.input_shape, network_name,
                               network_model=hyper_parameters.model)
            self.build_loss()
            if global_network:
                self.build_param_sync()
                self.build_param_update()

    def _nature_model(self, network_name):
        net = tf.transpose(self.inputs, [0, 2, 3, 1])
        net = tflearn.conv_2d(net, 32, 8, strides=4, activation='relu', name=network_name + '_conv1')
        net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu', name=network_name + '_conv2')
        net = tflearn.conv_2d(net, 64, 3, strides=1, activation='relu', name=network_name + '_conv3')
        net = tflearn.flatten(net)
        net = tflearn.fully_connected(net, 512, activation='relu', name=network_name + '_fc4')

        self.theta_layer_names = ['_conv1', '_conv2', '_conv3', '_fc4', '_policy', '_value']
        return net

    def _a3c_ff(self, network_name):
        net = tf.transpose(self.inputs, [0, 2, 3, 1])
        net = tflearn.conv_2d(net, 32, 8, strides=4, activation='relu', name=network_name + '_conv1')
        net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu', name=network_name + '_conv2')
        net = tflearn.flatten(net)
        net = tflearn.fully_connected(net, 256, activation='relu', name=network_name + '_fc3')

        self.theta_layer_names = ['_conv1', '_conv2', '_fc3', '_policy', '_value']
        return net

    def _small_fcn(self, network_name):
        net = tflearn.fully_connected(self.inputs, 32, activation='relu', name=network_name + '_fc1')
        net = tflearn.fully_connected(net, 16, activation='relu', name=network_name + '_fc2')
        #net = tflearn.dropout(net, 0.5)
        self.theta_layer_names = ['_fc1', '_fc2', '_policy', '_value']
        return net

    #def _build_tensorboard_summary(self):

    def build_network(self, num_actions, input_shape, network_name, network_model='nature'):
        logger.debug('Input shape: {}'.format(input_shape))
        self.inputs = tf.placeholder(tf.float32, [None] + input_shape)
        logger.info('Building network: {}'.format(network_model))

        if network_model == ModelNames.nature:
            net = self._nature_model(network_model)
        elif network_model == ModelNames.a3c_ff:
            net = self._a3c_ff(network_model)
        else:
            net = self._small_fcn(network_model)

        self.pi = tflearn.fully_connected(net, num_actions, activation='softmax', name=network_name + '_policy')
        self.value = tflearn.fully_connected(net, 1, activation='linear', name=network_name + '_value')

        def _flatten_list(l):
            return [item for sublist in l for item in sublist]
        self.theta = _flatten_list([tflearn.get_layer_variables_by_name(network_name + layer_name)
                                    for layer_name in self.theta_layer_names])

    def build_param_sync(self):
        self.param_sync = [tf.assign(local_theta, global_theta)
                           for local_theta, global_theta in zip(self.theta, self.global_network.theta)]
        #self.param_sync = [tf.assign(global_theta, local_theta) for global_theta, local_theta in zip(self.theta, self.global_network.theta)]

    def build_param_update(self):
        gradients = [tf.clip_by_norm(grad, 40.0)
                     for grad, _ in self.optimizer.compute_gradients(self.loss, var_list=self.theta)]
        #logger.warn(type(gradients))
        self.param_update = self.optimizer.apply_gradients(zip(gradients, self.global_network.theta))

    def build_loss(self):
        """
        This function defines the computation graph that is needed to compute the loss of the approximators.
        """

        # The n_step_return is an array of length n (where n is the batch size)
        self.n_step_returns = tf.placeholder(tf.float32, [None], name='loss_n_step_return')
        # The actions attribute is an array of length n
        self.actions = tf.placeholder(tf.int32, [None], name='loss_actions')
        # The advantage function requires a
        self.advantage_no_grad = tf.placeholder(tf.float32, [None], name='td_errors')

        # action matrix is n x a where each row corresponds to a time step and each column to an action
        action_matrix = tf.one_hot(self.actions, self.num_actions, 1.0, 0.0)
        # self.pi and log_pi are n x a matrices
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))

        # The advantage is simply the estimated value minus the bootstrapped returns
        advantage = self.n_step_returns - self.value 

        # The entropy is added to encourage exploration
        entropy = -tf.reduce_sum(tf.mul(log_pi, self.pi), reduction_indices=1)

        # Define the loss for the policy (minus is needed to perform *negative* gradient descent == gradient ascent)
        pi_loss = -(tf.reduce_sum(tf.mul(action_matrix, log_pi), reduction_indices=1) * self.advantage_no_grad  #advantage_no_gradient
                    + self.beta * entropy)
        value_loss = tf.square(advantage)

        # We can combine the policy loss and the value loss in a single expression
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

