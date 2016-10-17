import threading
import tensorflow as tf
import logging
import time
import gym
import numpy as np
import sys


logger = logging.getLogger('test')
logger.propagate = False
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s] (%(threadName)-10s) - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

def _init_theta():
    return None

def _init_theta_v():
    return None

class ParameterSynchronizer(object):

    def __init__(self):
        self.lock = threading.Lock()
        self.theta = _init_theta()
        self.theta_v = _init_theta_v()

    def update(self, d_theta, d_theta_v):
        logging.debug('Waiting for parameter update lock')
        self.lock.acquire()
        try:
            logging.debug('Updating global parameters')
            self.theta += d_theta
            self.theta_v += d_theta_v
        finally:
            self.lock.release()

    def get(self):
        logging.debug('Waiting for parameter synchronize lock')
        self.lock.acquire()

        try:
            logging.debug('Getting global parameters')
            theta = self.theta
            theta_v = self.theta_v
        finally:
            self.lock.release()

        return theta, theta_v


global_net_params = {
    'theta': _init_theta(),
    'theta_v': _init_theta_v()
}

hyper_parameters = {
    'T_max': 100000,
    't_max': 10,
    'gamma': 0.99,
    'lr': 0.1,
    'T': 1
}


class A3CAgent(object):

    def __init__(self, env_name, parameter_synchronizer, agent_name):
        self.ps = parameter_synchronizer
        self.env = gym.make(env_name)
        self.thread = threading.Thread(target=self._run, name=agent_name)

        self.t = 1
        self.reset_gradients()
        self.last_observation = self.env.reset()

    def reset_gradients(self):
        self.d_theta = None
        self.d_theta_v = None

    def synchronize_thread_parameters(self):
        self.theta = global_net_params['theta']
        self.theta_v = global_net_params['theta_v']

    def get_action(self):
        return self.env.action_space.sample()

    def state_value(self, observation):
        return None

    def accumulate_gradients(self, state, action, R):
        return

    def run(self):
        self.thread.start()

    def _run(self):
        rewards = np.zeros(hyper_parameters['t_max'])
        actions = np.zeros(hyper_parameters['t_max'])
        observations = np.zeros((hyper_parameters['t_max'],) + self.env.observation_space.shape)

        logger.debug('Running!')

        return
        '''
        while hyper_parameters['T'] < hyper_parameters['T_max']:
            self.reset_gradients()
            self.synchronize_thread_parameters()
            t_start = self.t

            terminal = False
            while not terminal and self.t - t_start != hyper_parameters['t_max']:
                i = self.t - t_start
                observations[i] = self.last_observation

                actions[i] = self.get_action()
                self.last_observation, rewards[i], terminal, info = self.env.step(actions[i])

                self.t += 1
                hyper_parameters['T'] += 1

            R = 0 if terminal else self.state_value(self.last_observation)

            for i in range(hyper_parameters['t_max'] - 1, 0, -1):
                R = rewards[i] + hyper_parameters['gamma'] * R
                self.accumulate_gradients(observations[i], actions[i], R)

            self.ps.update(self.d_theta, self.d_theta_v)
        '''

if __name__ == "__main__":
    parameter_synchronizer = ParameterSynchronizer()
    env_name = sys.argv[1]
    n_threads = int(sys.argv[2])

    agents = [A3CAgent(env_name, parameter_synchronizer, 'Agent %d' % i) for i in range(n_threads)]
    for agent in agents:
        agent.run()








