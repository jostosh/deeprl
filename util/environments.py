from scipy.misc import imresize
import gym
import numpy as np


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
