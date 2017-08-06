import threading
import tensorflow as tf
import numpy as np
from deeprl.common.logger import logger
from deeprl.common.environments import get_env
from deeprl.common.tensorboard import make_summary_from_python_var
import time
import abc
from deeprl.approximators.approximator_base import Approximator
from deeprl.approximators.optimizers.shared import RMSPropOptimizer
from deeprl.common.config import Config
import os


class Agent(abc.ABC):

    class Mode:
        Train = 'Train'
        Eval = 'Eval'
        Display = 'Display'

    def __init__(self, approximator: Approximator, session: tf.Session, optimizer: RMSPropOptimizer,
                 global_step, saver: tf.train.Saver, writer: tf.summary.FileWriter, global_time, name='Agent',
                 threaded=True):
        self.env = get_env()
        self.approximator = approximator
        self.session = session
        self.optimizer = optimizer
        self.global_step = global_step
        self.t = 0
        self.r_t = np.zeros(Config.t_max, dtype='float')
        self.a_t = np.zeros(Config.t_max, dtype='int')
        self.v_t = np.zeros(Config.t_max + 1, dtype='float')
        self.G_t = np.zeros(Config.t_max, dtype='float')
        self.s_t = np.zeros((Config.t_max,) + (Config.im_h, Config.im_w, Config.stacked_frames), dtype='float')
        self.last_state = None
        self.show_stats = True
        self.name = name
        self.threaded = threaded
        self.episode_score = 0
        self.train_episode = 0
        self.episode = 0

        self.global_time = global_time
        self.clock0 = time.time()
        self.n_batches = 0
        self._evaluating = False
        self._storing = False
        self._saver = saver
        self._writer = writer
        self._weights_path = os.path.join(Config.log_dir, 'model.ckpt')
        self.T = 0

    def train(self):
        """ Performs training algorithm """
        if self.threaded:
            self.thread = threading.Thread(target=self._train, name=self.name)
            self.thread.start()
        else:
            self._train()

    @abc.abstractmethod
    def _prepare_for_batch(self):
        """ Prepares the agent for a_t new episode """

    @abc.abstractmethod
    def _do_batch(self):
        """ Performs a_t batch of steps in the env :return: Bool whether the state is terminal """

    def _prepare_episode(self):
        """ Does everything to end the episode internally and externally """
        self.last_state = self.env.reset()
        self.episode_score = 0
        self.episode += 1

    @abc.abstractmethod
    def _update_approximator(self, batch_len):
        """ Updates approximator's parameters """

    def toggle_evaluating(self):
        self._evaluating = not self._evaluating

    def _show_statistics(self):
        total_duration = time.time() - self.clock0
        steps_per_second = self.T / total_duration

        if self.n_batches % Config.stat_interval == 0:
            logger.info("Steps per second: {}, steps per hour: {}".format(steps_per_second, 3600 * steps_per_second))

    def toggle_stats(self):
        self.show_stats = not self.show_stats

    def eval(self):

        """
        This is the thread function for a_t single A3C agent. The pseudo-code can be found in "Asynchronous Methods for
        Reinforcement Learning" by Mnih et al (2016): https://arxiv.org/abs/1602.01783

        It executes the actor-critic method with asynchronous updates and n-step returns in a_t forward view
        """
        num_episodes = 25
        # Initialize the reward, action and observation arrays
        logger.info("Evaluating for {} episodes".format(num_episodes))

        episode_idx = 0
        self._prepare_episode()
        self._prepare_for_batch()
        self.env.set_test()

        returns = np.zeros(num_episodes)

        t = 0
        # Main loop, execute this while T < T_max
        while episode_idx < num_episodes:
            # Get action
            action = self.approximator.get_action(self.last_state)
            self.last_state, reward, terminal, info = self.env.step(action)
            returns[episode_idx] += reward

            if terminal:
                self.last_state = self.env.reset()
                self.approximator.reset()
                episode_idx += 1

            t += 1

        logger.info("Mean score {}".format(np.mean(returns)))
        self._writer.add_summary(make_summary_from_python_var('Evaluation/Score', np.mean(returns)), self.train_episode)
        self.train_episode += 1
        self.env.set_train()

        return np.mean(returns)

    def _train(self):

        logger.info('Starting training')
        self.score = 0

        T = self.session.run(self.global_time)
        last_checkpoint = T - Config.eval_interval

        # Main loop, execute this while T < T_max
        self._prepare_episode()
        while self.T < Config.T_max:
            # A new batch begins
            self._prepare_for_batch()
            terminal = self._do_batch()
            if terminal:
                self._prepare_episode()
            if self.show_stats:
                self._show_statistics()
            if self._evaluating and self.T - last_checkpoint > Config.eval_interval:
                self.eval()
                last_checkpoint = np.round(self.T / Config.eval_interval) * Config.eval_interval
                self._store_parameters()

    def _store_parameters(self):
        if self._storing:
            logger.info("Storing weights at {}".format(self._weights_path))
            self._saver.save(self.session, self._weights_path, global_step=self.global_step)
            logger.info("Stored weights!")


class A3CAgent(Agent):

    def _prepare_for_batch(self):
        pass

    def _do_batch(self):

        # Set t_start to current t
        t_start = self.t

        # Boolean to denote whether the current state is terminal
        terminal_state = False

        # Now take steps following the thread-specific policy given by self.theta and self.theta_v
        while not terminal_state and self.t - t_start != Config.t_max:
            # Index of current step
            i = self.t - t_start
            # Set the current observation
            self.s_t[i] = self.last_state
            # Get the corresponding value and action. This is done simultaneously such that the approximators only
            # has to perform a single forward pass.
            self.v_t[i], self.a_t[i] = self.approximator.get_value_and_action([self.last_state])
            # Perform step in environment and obtain rewards and observations
            self.last_state, self.r_t[i], terminal_state = self.env.step(self.a_t[i])
            # Increment the relevant counters and sums
            self.increment_t()
            self.episode_score += self.r_t[i]

        # Reward clipping helps to find a robust hyperparameter setting
        self.r_t = np.clip(self.r_t, -1.0, 1.0)

        # Initialize the n-step return
        n_step_target = 0 if terminal_state else self.approximator.get_value([self.last_state])
        batch_len = self.t - t_start

        # Forward view of n-step returns, start from i == t_max - 1 and go to i == 0
        for i in reversed(range(batch_len)):
            # Straightforward accumulation of rewards
            n_step_target = self.r_t[i] + Config.gamma * n_step_target
            self.G_t[i] = n_step_target

        if terminal_state:
            logger.info(
                'Terminal state reached (episode {}, reward {}, T {}): resetting state'.
                    format(self.episode, self.episode_score, self.session.run(self.global_time))
            )

        self._update_approximator(batch_len)
        self.n_batches += 1

        return terminal_state

    def increment_t(self):
        self.t += 1
        self.T = self.session.run(self.global_step)

    def _update_approximator(self, batch_len):
        t = self.session.run(self.global_time)
        current_lr = Config.lr - Config.lr / Config.T_max * t
        summaries = self.approximator.update_params(
            self.a_t[:batch_len],
            self.s_t[:batch_len],
            self.v_t[:batch_len],
            current_lr,
            self.G_t[:batch_len],
            include_summaries=(self.name == "Agent1" and self.n_batches % 50 == 0)
        )
        if summaries:
            self._writer.add_summary(summaries, t)

    def _prepare_episode(self):
        super()._prepare_episode()
        self.approximator.synchronize_parameters()

