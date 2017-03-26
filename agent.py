import threading
import tensorflow as tf
import numpy as np

from deeprl.common.logger import logger
from deeprl.approximators.nn import ActorCriticNN
from deeprl.common.environments import get_env, AtariEnvironment
from deeprl.common.hyper_parameters import *
from deeprl.common.tensorboard import writer_new_event, make_summary_from_python_var
from deeprl.approximators.optimizers import RMSPropShared, AdamShared
import time
import pickle
from random import shuffle
from tensorflow.contrib.tensorboard.plugins import projector
import scipy.misc
import subprocess
from copy import deepcopy
from deeprl.common.catch import CatchEnv
import abc


class Agent(abc.ABC):

    def __init__(self, env, approximator, session, optimizer, hyper_parameters, global_step, saver, writer):
        self.env = env
        self.local_approximator = approximator
        self.session = session
        self.optimizer = optimizer
        self.hp = hyper_parameters
        self.global_step = global_step
        self.t = 0
        self.r = np.zeros(self.hp.t_max,    dtype='float')
        self.a = np.zeros(self.hp.t_max,    dtype='float')
        self.v = np.zeros(self.hp.t_max+1,  dtype='float')
        self.G = np.zeros(self.hp.t_max,    dtype='float')
        self.s = np.zeros((self.hp.t_max,) + self.env.state_shape(), dtype='float')
        self.last_state = None
        self.show_stats = False

        self.clock0 = time.time()
        self.stat_interval = 50
        self.n_batches = 0
        self._evaluating = False
        self._storing = False
        self._saver = saver
        self._writer = writer
        self._weights_path = os.path.join(writer.get_logdir(), 'model.ckpt')

    @abc.abstractmethod
    def train(self):
        """
        Performs training algorithm
        """

    @abc.abstractmethod
    def _prepare_for_batch(self):
        """
        Prepares the agent for a new episode
        """

    @abc.abstractmethod
    def _do_batch(self):
        """
        Performs a batch of steps in the env
        :return: Bool whether the state is terminal
        """

    @abc.abstractmethod
    def _prepare_episode(self):
        """
        Does everything to end the episode internally and externally
        """

    def toggle_evaluating(self):
        self._evaluating = not self._evaluating

    def _show_statistics(self, T):
        total_duration = time.time() - self.clock0
        mean_duration = total_duration / T

        if self.n_batches % self.stat_interval == 0:
            logger.info("Steps per second: {}, steps per hour: {}".format(1 / mean_duration, 3600 / mean_duration))

    def toggle_stats(self):
        self.show_stats = not self.show_stats

    @abc.abstractmethod
    def eval(self):

        """
        This is the thread function for a single A3C agent. The pseudo-code can be found in "Asynchronous Methods for
        Reinforcement Learning" by Mnih et al (2016): https://arxiv.org/abs/1602.01783

        It executes the actor-critic method with asynchronous updates and n-step returns in a forward view
        """
        num_episodes = 50
        # Initialize the reward, action and observation arrays
        logger.info("Evaluating for {} episodes".format(num_episodes))

        episode_idx = 0
        self._prepare_episode()
        self._prepare_for_batch()
        self.env.set_test()

        returns = np.zeros(num_episodes)

        t = 0
        embeddings = []
        embedding_images = []
        # Main loop, execute this while T < T_max
        while episode_idx < num_episodes:
            # Get action
            action = self.local_approximator.get_action(self.last_state)
            self.last_state, reward, terminal, info = self.env.step(action)
            returns[episode_idx] += reward

            if terminal:
                self.last_state = self.env.reset_random()
                self.local_network.reset()
                episode_idx += 1

            if t % 100 == 0:
                embeddings.append(self.local_network.get_embedding(self.last_state, session))
                embedding_images.append(self.env.env._get_image() if isinstance(self.env, AtariEnvironment) else self.last_state[-1, :, :])

                if len(embeddings) > 100:
                    deletion_index = np.random.randint(100)
                    del embeddings[deletion_index]
                    del embedding_images[deletion_index]

            t += 1

        if isinstance(self.env, AtariEnvironment):
            self.store_embeddings(embedding_images, embeddings)

        logger.info("Mean score {}".format(np.mean(returns)))
        writer.add_summary(make_summary_from_python_var('Evaluation/Score', np.mean(returns)), self.train_episode)
        writer.flush()
        self.train_episode += 1
        self.env.set_train()

        return np.mean(returns)

    def _train(self):

        logger.info('Starting training')
        self.score = 0
        total_duration = 0.

        T = self.session.run(self.global_step)
        last_checkpoint = T - self.hp.evaluation_interval

        # Main loop, execute this while T < T_max
        while T < self.hp.T_max:
            # A new batch begins
            self._prepare_for_batch()
            terminal = self._do_batch()
            if terminal:
                self._prepare_episode()
            if self.show_stats:
                self._show_statistics(T)
            if self._evaluating and T - last_checkpoint > self.hp.evaluation_interval:
                self.eval()
                last_checkpoint = np.round(T / self.hp.evaluation_interval) * self.hp.evaluation_interval
                if self._storing:
                    logger.info("Storing weights at {}".format(weights_path))
                    self._saver.save(self.session, self.weights_path, global_step=self.global_step)
                    logger.info("Stored weights!")


