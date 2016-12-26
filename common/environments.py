from scipy.misc import imresize
import gym
import numpy as np
from copy import deepcopy
import tensorflow as tf


def get_env(env, frames_per_state=4, output_shape=None, session=None):
    if env in ['Breakout-v0', 'Pong-v0', 'BeamRider-v0', 'Qbert-v0', 'SpaceInvaders-v0']:
        return AtariEnvironment(env, frames_per_state, output_shape=output_shape, session=session)
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


class AtariEnvironment(object):

    def __init__(self, env_name, frames_per_state=4, action_repeat=4, output_shape=(84, 84), session=None):
        self.env = gym.make(env_name)
        self.last_observation = self.env.reset()
        self.frames_per_state = frames_per_state
        self.state = []
        self.action_repeat = action_repeat

        self.output_shape = output_shape
        self.env.frameskip = 1

        self.is_training = True

        assert action_repeat > 0

        self.session = session
        if session:
            with tf.name_scope("FramePreprocessing"):
                self.raw_image = tf.placeholder(tf.float32, [210, 160, 3])
                self.prev_image = tf.placeholder(tf.float32, [210, 160, 3])
                without_artifacts = tf.maximum(self.raw_image, self.prev_image, name="WithoutArtifacts")
                gray_scale = tf.image.rgb_to_grayscale(tf.reshape(self.raw_image, [1, 210, 160, 3]))
                self.preprocessed = tf.reshape(tf.image.resize_images(gray_scale, [84, 84]), [84, 84])

    def _preprocess_observation(self, observation):
        """
        This preprocessing step was taken from "Human-level control through deep reinforcement learning"
        (Mnih et al 2015).
        :param observation: the raw observation
        :return: a preprocessed observation
        """
        if self.session:
            return self.session.run(self.preprocessed, feed_dict={self.raw_image: observation,
                                                                  self.prev_image: self.last_observation})

        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

        # Remove Atari artifacts
        preprocessed_observation = np.maximum(self.last_observation, observation)
        # Convert to gray scale and resize
        return imresize(rgb2gray(preprocessed_observation), self.output_shape) / 255. #(84, 84))

    def step(self, action):
        step_reward = 0
        step_terminal = False

        lives = deepcopy(self.env.ale.lives())

        for _ in range(self.action_repeat - 1):
            if not step_terminal:
                # Only if we are not already at a terminal state we actually perform an action and preprocess the
                # observation, as well as a check whether the current state is terminal
                observation, reward, terminal, info = self.env.step(action)
                step_reward += reward

                step_terminal = max(terminal, step_terminal)
                self.last_observation = observation

                if self.is_training and self.env.ale.lives() < lives:
                    step_terminal = True

        # preprocessed_observation must be set and should be added to the buffer whether it is terminal or not
        if not step_terminal:
            observation, reward, terminal, info = self.env.step(action)
            step_reward += reward
            step_terminal = max(terminal, step_terminal)
        preprocessed_observation = self._preprocess_observation(observation)
        [self.state.append(preprocessed_observation) for _ in range(self.frames_per_state - len(self.state) + 1)]

        self.state = self.state[-self.frames_per_state:]

        return np.copy(self.state), step_reward, step_terminal, info

    def reset(self):
        self.state = []
        self.last_observation = self.env.reset()
        self.step(0)

        assert len(self.state) == self.frames_per_state, 'State length: {}'.format(len(self.state))

        return np.copy(self.state)

    def state_shape(self):
        return tuple([self.frames_per_state] + self.output_shape)

    def num_actions(self):
        return self.env.action_space.n

    def set_train(self):
        self.is_training = True

    def set_test(self):
        self.is_training = False

    def reset_random(self):
        state = self.reset()

        random_steps = np.random.randint(8)
        for i in range(random_steps):
            action = self.env.action_space.sample()
            state, _, _, _ = self.step(action)

        return state