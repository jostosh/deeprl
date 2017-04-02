from scipy.misc import imresize
import gym
import numpy as np
from copy import deepcopy
import tensorflow as tf
from deeprl.common.catch import CatchEnv
from gym.envs.atari.atari_env import ACTION_MEANING

def get_env(env, frames_per_state=4, output_shape=None, session=None):
    if env in ['Breakout-v0', 'Pong-v0', 'BeamRider-v0', 'Qbert-v0', 'SpaceInvaders-v0']:
        return AtariEnvironment(env.replace('-', 'Deterministic-'), frames_per_state, output_shape=output_shape,
                                session=session)
    elif env == 'Catch':
        return CatchEnv(frames_per_state)
    return ClassicControl(env)


class ClassicControl(object):

    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.last_observation = self.env.reset()

    def step(self, action):
        self.last_observation, reward, terminal, info = self.env.step(action)
        return self.last_observation, reward, terminal, info

    def reset(self):
        return self.env.reset()

    def state_shape(self):
        return self.env.observation_space.shape

    def num_actions(self):
        return self.env.action_space.n

    def set_train(self):
        self.is_training = True

    def set_test(self):
        self.is_training = False

    def reset_random(self):
        return self.reset()


class AtariEnvironment(object):

    def __init__(self, env_name, frames_per_state=4, action_repeat=4, output_shape=(84, 84), session=None):
        self.env = gym.make(env_name)
        self.last_observation = np.empty((210, 160, 1), dtype=np.uint8) #self.env.reset()
        self.frames_per_state = frames_per_state
        self.state = []
        self.action_repeat = action_repeat

        self.output_shape = output_shape
        #self.env.frameskip = 1

        self.is_training = True

        self.env.ale.setInt(b'frame_skip', 1)
        self.env.ale.setFloat(b'repeat_action_probability', 0.)
        #self.env.ale.setBool(b'color_averaging', True)
        self.real_actions = self.env.ale.getMinimalActionSet()
        if 'Breakout' in env_name or 'Pong' in env_name:
            self.real_actions = np.asarray([
                a for a in self.real_actions
                if np.array(self.env.get_action_meanings())[np.where(self.real_actions == a)[0]] not in
                   ['FIRE', 'LEFTFIRE', 'RIGHTFIRE']
            ])

        self._screen = np.empty((210, 160, 1), dtype=np.uint8)
        assert action_repeat > 0


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
        # preprocessed_observation = np.maximum(self.last_observation, observation)
        # Convert to gray scale and resize
        return imresize(np.reshape(observation, (210, 160)), self.output_shape) / 255. #(84, 84))

    def step(self, action):
        step_terminal = False

        reward = 0
        for i in range(self.action_repeat):
            reward += self.env.ale.act(self.real_actions[action])

        if self.env.ale.game_over():
            step_terminal = True
        self.env.ale.getScreenGrayscale(self._screen)

        preprocessed_observation = self._preprocess_observation(self._screen).astype(dtype='float32')
        [self.state.append(preprocessed_observation) for _ in range(self.frames_per_state - len(self.state) + 1)]

        self.state = self.state[-self.frames_per_state:]

        return np.copy(self.state), reward, step_terminal, None

    def reset(self):
        self.state = []
        self.env.reset()
        self.env.ale.getScreenGrayscale(self.last_observation)

        self.step(0)

        assert len(self.state) == self.frames_per_state, 'State length: {}'.format(len(self.state))

        return np.copy(self.state)

    def state_shape(self):
        return tuple([self.frames_per_state] + self.output_shape)

    def num_actions(self):
        return len(self.real_actions)

    def set_train(self):
        self.is_training = True

    def set_test(self):
        self.is_training = False

    def reset_random(self):
        state = self.reset()

        #random_steps = np.random.randint(8)
        #for i in range(random_steps):
        #    action = np.random.randint(self.num_actions()) #self.env.action_space.sample()
        #    state, _, _, _ = self.step(action)

        return state

    def get_action_meanings(self):
        return [ACTION_MEANING[a] for a in self.real_actions]