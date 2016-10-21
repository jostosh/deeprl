import threading
import tensorflow as tf
import logging
import gym
import numpy as np
import sys
import tflearn
from scipy.misc import imresize


logger = logging.getLogger('test')
logger.propagate = False
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s] (%(threadName)-10s) - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)


class AtariEnvironment(object):

    def __init__(self, env_name, frames_per_state=3):
        self.env = gym.make(env_name)
        self.last_observation = self.env.reset()
        self.frames_per_state = frames_per_state
        self.state = []

    def _preprocess_observation(self, observation):
        """
        This preprocessing step was taken from "Human-level control through deep reinforcement learning"
        (Mnih et al 2015).
        :param observation: the raw observation
        :return: a preprocessed observation
        """
        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

        # Remove Atari artifacts
        preprocessed_observation = np.maximum(self.last_observation, observation)
        # Convert to gray scale and resize
        return imresize(rgb2gray(preprocessed_observation), (84, 84))

    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        preprocessed_observation = self._preprocess_observation(observation)

        if len(self.state) != self.frames_per_state:
            self.state.append(preprocessed_observation)
        else:
            self.state = self.state[1:] + [preprocessed_observation]

        return self.state, reward, terminal, info

    def reset(self):
        self.state = []
        self.last_observation = self.env.reset()
        for _ in range(0, self.frames_per_state):
            self.step(0)

        assert len(self.state) == self.frames_per_state, 'State length: {}'.format(len(self.state))

        return self.state

    def state_shape(self):
        return (self.frames_per_state, 84, 84)

    def num_actions(self):
        return self.env.action_space.n


class DNN(object):

    def __init__(self, num_actions, network_name, optimizer, session, global_network=None):
        self.num_actions = num_actions
        self.optimizer = optimizer
        self.global_network = global_network
        self.session = session

        # Build computational graphs for loss, synchronization of parameters and parameter updates
        with tf.device('/cpu:0'):
            self.build_network(num_actions, hyper_parameters['frames_per_state'], network_name)
            self.build_loss()
            if global_network:
                self.build_param_sync()
                self.build_param_update()

    def build_network(self, num_actions, frames_per_state, network_name):
        self.inputs = tf.placeholder(tf.float32, [None, frames_per_state, 84, 84])
        net = tf.transpose(self.inputs, [0, 2, 3, 1])
        net = tflearn.conv_2d(net, 32, 8, strides=4, activation='relu', name=network_name + '_conv1')
        net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu', name=network_name + '_conv2')
        net = tflearn.conv_2d(net, 64, 3, strides=1, activation='relu', name=network_name + '_conv3')
        net = tflearn.fully_connected(net, 512, activation='relu', name=network_name + '_fc4')
        #net = tflearn.lstm(net, 256, name=network_name + '_lstm5')

        self.pi = tflearn.fully_connected(net, num_actions, activation='softmax', name=network_name + '_policy')
        self.value = tflearn.fully_connected(net, 1, activation='linear', name=network_name + '_value')

        self.theta_layer_names = ['_conv1', '_conv2', '_conv3', '_fc4', '_policy', '_value']

        def _flatten_list(l):
            return [item for sublist in l for item in sublist]
        self.theta = _flatten_list([tflearn.get_layer_variables_by_name(network_name + layer_name)
                                    for layer_name in self.theta_layer_names])

    def build_param_sync(self):
        self.param_sync = [tf.assign(global_theta, local_theta) for global_theta, local_theta in zip(self.theta, self.global_network.theta)]

    def build_param_update(self):
        gradients = tf.gradients(self.loss, self.theta)
        self.param_update = self.optimizer.apply_gradients(zip(gradients, self.global_network.theta))

    def build_loss(self):
        """
        This function defines the computation graph that is needed to compute the loss of the network.
        """

        # The n_step_return is a 1 x n tensor (where n is the batch size)
        self.n_step_returns = tf.placeholder(tf.float32, [None], name='loss_n_step_return')
        # The actions attribute is a 1 x n tensor
        self.actions = tf.placeholder(tf.int32, [None], name='loss_actions')

        # action matrix is n x a where each row corresponds to a time step and each column to an action
        action_matrix = tf.one_hot(self.actions, self.num_actions, 1.0, 0.0)
        # self.pi and log_pi are n x a matrices
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))

        # The advantage is simply the estimated value minus the bootstrapped returns
        advantage = self.value - self.n_step_returns

        # The entropy is added to encourage exploration
        entropy = -tf.reduce_sum(tf.mul(log_pi, self.pi), reduction_indices=1)

        # Define the loss for the policy (minus is needed to perform *negative* gradient descent == gradient ascent)
        pi_loss = -(tf.reduce_sum(tf.mul(action_matrix, log_pi), reduction_indices=1) * advantage
                    + hyper_parameters['beta'] * entropy)
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


hyper_parameters = {
    'T_max': 100000,
    't_max': 10,
    'gamma': 0.99,
    'learning_rate': 0.001,
    'T': 1,
    'decay': 0.99,
    'epsilon': 0.1,
    'beta': 0.01,
    'frames_per_state': 3
}

class A3CAgent(object):

    def __init__(self, env_name, global_network, agent_name, session):
        self.global_network = global_network
        self.env = AtariEnvironment(env_name) #gym.make(env_name)
        self.num_actions = self.env.num_actions()
        self.local_network = DNN(num_actions=self.num_actions,
                                 network_name=agent_name,
                                 optimizer=tf.train.RMSPropOptimizer(learning_rate=hyper_parameters['learning_rate'],
                                                                     decay=hyper_parameters['decay'],
                                                                     epsilon=hyper_parameters['epsilon']),
                                 session=session,
                                 global_network=global_network)
        self.thread = threading.Thread(target=self._run, name=agent_name)
        self.t = 1
        self.last_state = self.env.reset()
        self.session = session

        self.agent_name = agent_name

    def get_action(self):
        pi = self.local_network.get_pi(self.last_state)
        return np.random.choice(self.num_actions, 1, p=pi)[0]

    def state_value(self, observation):
        return self.local_network.get_value(observation)

    def get_value_and_action(self):
        value, pi = self.session.run([self.local_network.value, self.local_network.pi],
                                     feed_dict={self.local_network.inputs: [self.last_state]})
        action = np.random.choice(self.num_actions, 1, p=pi[0]) #[0]
        return value, action

    def update_params(self, n_step_return, actions, states):
        dnn = self.local_network
        logger.debug('n_step returns: {}'.format(n_step_return))
        logger.debug('actions: {}'.format(actions))
        self.session.run(dnn.param_update, feed_dict={dnn.n_step_returns: n_step_return,
                                                      dnn.actions: actions,
                                                      dnn.inputs: states})

    def run(self):
        logging.info('Starting training')
        self.thread.start()

    def synchronize_thread_parameters(self):
        logging.debug('Synchronizing global parameters')
        self.session.run(self.local_network.param_sync)

    def _run(self):
        """
        This is the thread function for a single A3C agent. The pseudo-code can be found in "Asynchronous Methods for
        Reinforcement Learning" by Mnih et al (2016): https://arxiv.org/abs/1602.01783

        It executes the actor-critic method with asynchronous updates and n-step returns in a forward view
        """
        # Initialize the reward, action and observation arrays #TODO maybe rename observations...
        rewards = np.zeros(hyper_parameters['t_max'], dtype='float')
        actions = np.zeros(hyper_parameters['t_max'], dtype='int')
        values  = np.zeros(hyper_parameters['t_max'], dtype='float')
        n_step_targets = np.zeros(hyper_parameters['t_max'], dtype='float')
        states = np.zeros((hyper_parameters['t_max'],) + self.env.state_shape(), dtype='float')

        # Main loop, execute this while T < T_max
        while hyper_parameters['T'] < hyper_parameters['T_max']:
            # A new batch begins, reset the gradients and synchronize thread-specific parameters
            self.synchronize_thread_parameters()
            # Set t_start to current t
            t_start = self.t

            # Boolean to denote whether the current state is terminal
            terminal_state = False

            # Now take steps following the thread-specific policy given by self.theta and self.theta_v
            while not terminal_state and self.t - t_start != hyper_parameters['t_max']:
                # Index of current step
                i = self.t - t_start
                # Set the current observation
                states[i] = self.last_state
                # Get the corresponding value and action. This is done simultaneously such that the network only has to
                # perform a single forward pass.
                values[i], actions[i] = self.get_value_and_action()
                # Perform step in environment and obtain rewards and observations
                self.last_state, rewards[i], terminal_state, info = self.env.step(actions[i])
                # Increment time counters
                self.t += 1
                hyper_parameters['T'] += 1

                if self.agent_name == 'Agent_0':
                    self.env.env.render()

            # Initialize the n-step return
            n_step_target = 0 if terminal_state else self.state_value(self.last_state)

            # Forward view of n-step returns, start from i == t_max - 1 and go to i == 0
            for i in reversed(range(hyper_parameters['t_max'])):
                # Straightforward accumulation of rewards
                n_step_target = rewards[i] + hyper_parameters['gamma'] * n_step_target
                n_step_targets[i] = n_step_target

            # Now update the global network parameters
            self.update_params(n_step_targets, actions, states)
            logger.debug('Parameters updated!')

            if terminal_state:
                logger.info('Terminal state reached: resetting state')
                self.last_state = self.env.reset()


if __name__ == "__main__":
    env_name = sys.argv[1]
    n_threads = int(sys.argv[2])

    global_env = AtariEnvironment(env_name)
    num_actions = global_env.num_actions()

    session = tf.InteractiveSession()

    global_network = DNN(num_actions=num_actions,
                         network_name='GLOBAL',
                         optimizer=tf.train.RMSPropOptimizer(learning_rate=hyper_parameters['learning_rate'],
                                                             decay=hyper_parameters['decay'],
                                                             epsilon=hyper_parameters['epsilon']),
                         session=session)

    agents = [A3CAgent(env_name, global_network, 'Agent_%d' % i, session) for i in range(n_threads)]

    session.run(tf.initialize_all_variables())
    for agent in agents:
        agent.run()








