import threading
import tensorflow as tf
import numpy as np

from deeprl.util.logger import logger
from deeprl.approximators.nn import ActorCriticNN
from deeprl.util.environments import AtariEnvironment
from deeprl.util.hyper_parameters import *


class A3CAgent(object):

    def __init__(self, env_name, global_network, agent_name, session):
        self.global_network = global_network
        self.env = AtariEnvironment(env_name) #gym.make(env_name)
        self.num_actions = self.env.num_actions()
        self.local_network = ActorCriticNN(num_actions=self.num_actions,
                                           network_name=agent_name,
                                           optimizer=tf.train.RMSPropOptimizer(
                                               learning_rate=hyper_parameters.learning_rate,
                                               decay=hyper_parameters.lr_decay,
                                               epsilon=hyper_parameters.rms_epsilon),
                                           session=session,
                                           global_network=global_network)
        self._train_thread = threading.Thread(target=self._train, name=agent_name)
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
        self.session.run(dnn.param_update, feed_dict={dnn.n_step_returns: n_step_return,
                                                      dnn.actions: actions,
                                                      dnn.inputs: states})

    def train(self):
        logger.info('Starting training')
        self._train_thread.start()

    def synchronize_thread_parameters(self):
        logger.debug('Synchronizing global parameters')
        self.session.run(self.local_network.param_sync)

    def _train(self):
        """
        This is the thread function for a single A3C agent. The pseudo-code can be found in "Asynchronous Methods for
        Reinforcement Learning" by Mnih et al (2016): https://arxiv.org/abs/1602.01783

        It executes the actor-critic method with asynchronous updates and n-step returns in a forward view
        """
        global T
        # Initialize the reward, action and observation arrays
        rewards         = np.zeros(hyper_parameters.t_max, dtype='float')
        actions         = np.zeros(hyper_parameters.t_max, dtype='int')
        values          = np.zeros(hyper_parameters.t_max, dtype='float')
        n_step_targets  = np.zeros(hyper_parameters.t_max, dtype='float')
        states          = np.zeros((hyper_parameters.t_max,) + self.env.state_shape(), dtype='float')

        # Main loop, execute this while T < T_max
        while T < hyper_parameters.T_max:
            # A new batch begins, reset the gradients and synchronize thread-specific parameters
            self.synchronize_thread_parameters()
            # Set t_start to current t
            t_start = self.t

            # Boolean to denote whether the current state is terminal
            terminal_state = False

            # Now take steps following the thread-specific policy given by self.theta and self.theta_v
            while not terminal_state and self.t - t_start != hyper_parameters.t_max: #hyper_parameters['t_max']:
                # Index of current step
                i = self.t - t_start
                # Set the current observation
                states[i] = self.last_state
                # Get the corresponding value and action. This is done simultaneously such that the approximators only
                # has to perform a single forward pass.
                values[i], actions[i] = self.get_value_and_action()
                # Perform step in environment and obtain rewards and observations
                self.last_state, rewards[i], terminal_state, info = self.env.step(actions[i])
                # Increment time counters
                self.t += 1
                T += 1

                if self.agent_name == 'Agent_0':
                    self.env.env.render()

            # Initialize the n-step return
            n_step_target = 0 if terminal_state else self.state_value(self.last_state)

            # Forward view of n-step returns, start from i == t_max - 1 and go to i == 0
            for i in reversed(range(hyper_parameters.t_max)):
                # Straightforward accumulation of rewards
                n_step_target = rewards[i] + hyper_parameters.gamma * n_step_target
                n_step_targets[i] = n_step_target

            # Now update the global approximators parameters
            self.update_params(n_step_targets, actions, states)
            logger.debug('Parameters updated!')

            if terminal_state:
                logger.info('Terminal state reached: resetting state')
                self.last_state = self.env.reset()


if __name__ == "__main__":
    hyper_parameters = HyperParameters(parse_cmd_args())
    T = 1

    env_name = hyper_parameters.env
    n_threads = hyper_parameters.n_threads

    global_env = AtariEnvironment(env_name)
    num_actions = global_env.num_actions()

    session = tf.InteractiveSession()

    global_network = ActorCriticNN(num_actions=num_actions,
                                   network_name='GLOBAL',
                                   optimizer=tf.train.RMSPropOptimizer(
                                       learning_rate=hyper_parameters.learning_rate,
                                       decay=hyper_parameters.lr_decay,
                                       epsilon=hyper_parameters.rms_epsilon),
                                   session=session)

    agents = [A3CAgent(env_name, global_network, 'Agent_%d' % i, session) for i in range(n_threads)]

    session.run(tf.initialize_all_variables())
    for agent in agents:
        agent.train()








