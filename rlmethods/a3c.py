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
from tensorflow.python import debug as tf_debug
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean("debug", False,
                     "Use debugger to track down bad values during training")


class A3CAgent(object):

    def __init__(self, env_name, global_network, agent_name, session, optimizer, hp):
        """
        Initializes an Asynchronous Advantage Actor-Critic agent (A3C).
        :param env_name:        Name of the environment
        :param global_network:  Global network to use for updates and synchronization
        :param agent_name:      Name of this agent
        :param session:         TensorFlow session
        """
        self.env = get_env(env_name)
        self.num_actions = self.env.num_actions()

        self.local_network = ActorCriticNN(num_actions=self.num_actions,
                                           agent_name=agent_name,
                                           optimizer=optimizer,
                                           hyper_parameters=hp,
                                           global_network=global_network)
        self.global_network = global_network

        self._train_thread = threading.Thread(target=self._train, name=agent_name)
        self.t = 1
        self.last_state = self.env.reset()
        self.session = session
        self.agent_name = agent_name

        self.n_episodes = 1
        self.train_episode = 0
        self.hp = hp

    def train(self):
        """
        Starts a training thread
        """
        self._train_thread.start()

    def synchronize_thread_parameters(self):
        """
        Synhronizes the thread network parameters with the global network
        """
        logger.debug('Synchronizing global parameters')
        self.session.run(self.local_network.param_sync)

    def _train(self):
        """
        This is the thread function for a single A3C agent. The pseudo-code can be found in "Asynchronous Methods for
        Reinforcement Learning" by Mnih et al (2016): https://arxiv.org/abs/1602.01783

        It executes the actor-critic method with asynchronous updates and n-step returns in a forward view
        """
        logger.info('Starting training')
        # Initialize the reward, action and observation arrays
        rewards = np.zeros(self.hp.t_max, dtype='float')
        values = np.zeros(self.hp.t_max + 1, dtype='float')
        actions = np.zeros(self.hp.t_max, dtype='int')
        n_step_targets = np.zeros(self.hp.t_max, dtype='float')
        states = np.zeros((self.hp.t_max,) + self.env.state_shape(), dtype='float')

        epr = 0
        total_duration = 0.

        T = session.run(T_var)
        last_checkpoint = T - self.hp.evaluation_interval
        n_updates = 0

        # Main loop, execute this while T < T_max
        while T < self.hp.T_max:
            #[arr.fill(0) for arr in [rewards, actions, values, n_step_targets, states]]
            t0 = time.time()

            # A new batch begins, reset the gradients and synchronize thread-specific parameters
            self.synchronize_thread_parameters()

            # Set t_start to current t
            t_start = self.t

            # Boolean to denote whether the current state is terminal
            terminal_state = False

            # Now take steps following the thread-specific policy given by self.theta and self.theta_v
            while not terminal_state and self.t - t_start != self.hp.t_max:

                # Index of current step
                i = self.t - t_start
                # Set the current observation
                states[i] = self.last_state
                # Get the corresponding value and action. This is done simultaneously such that the approximators only
                # has to perform a single forward pass.
                values[i], actions[i] = self.local_network.get_value_and_action(self.last_state, session)
                # Perform step in environment and obtain rewards and observations
                a = actions[i] if not self.hp.lpq_single_winner else actions[i] // self.hp.ppa
                self.last_state, rewards[i], terminal_state = self.env.step(a)
                # Increment time counters
                self.t += 1
                T = session.run(global_step)
                current_lr = self.hp.learning_rate - ((self.hp.learning_rate / self.hp.T_max * T)
                                                      if not self.hp.ignore_annealing else 0)
                epr += rewards[i]

            if self.hp.clip_rewards:
                # Reward clipping helps to stabilize training
                rewards = np.clip(rewards, -1.0, 1.0)

            # Initialize the n-step return
            n_step_target = 0 if terminal_state else self.local_network.get_value(self.last_state, session)
            batch_len = self.t - t_start

            # Forward view of n-step returns, start from i == t_max - 1 and go to i == 0
            for i in reversed(range(batch_len)):
                # Straightforward accumulation of rewards
                n_step_target = rewards[i] + self.hp.gamma * n_step_target
                n_step_targets[i] = n_step_target

            # Count the number of updates
            n_updates += 1

            if self.hp.optimality_tightening:
                values[batch_len] = n_step_target
                lower_limits = []
                for j in range(batch_len):
                    current_max = -np.inf
                    for k in range(batch_len - j):
                        current_max = max(current_max,
                                          np.sum(rewards[j:j + k + 1] * (self.hp.gamma ** np.arange(k + 1))) +
                                          self.hp.gamma ** (k + 1) * values[j + k + 1])
                    lower_limits.append(current_max)

                upper_limits = []
                for j in range(batch_len):
                    current_min = np.inf
                    for k in range(j):
                        current_min = min(current_min, np.sum(rewards[j - k - 1:j] *
                                                              (self.hp.gamma ** np.arange(-k-1, 0))) +
                                          self.hp.gamma ** (-k - 1) * values[j - k - 1])
                    upper_limits.append(current_min)
            else:
                lower_limits = upper_limits = None

            self.hp.fplc *= self.hp.fp_decay

            # Now update the global approximator's parameters
            summaries = self.local_network.update_params(actions=actions[:batch_len],
                                                         states=states[:batch_len],
                                                         lr=current_lr,
                                                         last_state=self.last_state,
                                                         session=session,
                                                         n_step_returns=n_step_targets[:batch_len],
                                                         values=values[:batch_len],
                                                         upper_limits=upper_limits,
                                                         lower_limits=lower_limits,
                                                         include_summaries=(self.agent_name == "Agent_1" and
                                                                            n_updates % 50 == 0))

            if summaries:
                writer.add_summary(summaries, self.t)

            if terminal_state:
                if n_updates % 20 == 0:
                    logger.info('Terminal state reached (episode {}, reward {}, lr {:.5f}, T {}): resetting state'.format(
                        self.n_episodes, epr, current_lr, T))

                self.n_episodes += 1
                self.last_state = self.env.reset()
                epr = 0
                self.local_network.reset()

            total_duration += time.time() - t0
            mean_duration = total_duration / T

            if self.agent_name == "Agent_1" and n_updates % 50 == 0:
                logger.info("Steps per second: {}, steps per hour: {}".format(1 / mean_duration,
                                                                           3600 / mean_duration))

            if T - last_checkpoint > self.hp.evaluation_interval and self.agent_name == 'Agent_0':
                if isinstance(self.env, AtariEnvironment) or isinstance(self.env, CatchEnv):
                    mean_score = self.evaluate(50)

                last_checkpoint = T // (self.hp.evaluation_interval / 10) * (self.hp.evaluation_interval / 10) # round to the nearest 1e6
                if isinstance(self.env, AtariEnvironment) or self.hp.force_store:
                    logger.info("Storing weights at {}".format(weights_path))
                    saver.save(session, weights_path, global_step=T_var)
                    logger.info("Stored weights!")

    def evaluate(self, num_episodes):
        """
        This is the thread function for a single A3C agent. The pseudo-code can be found in "Asynchronous Methods for
        Reinforcement Learning" by Mnih et al (2016): https://arxiv.org/abs/1602.01783

        It executes the actor-critic method with asynchronous updates and n-step returns in a forward view
        """
        # Initialize the reward, action and observation arrays
        logger.info("Evaluating for {} episodes".format(num_episodes))

        episode_idx = 0
        self.synchronize_thread_parameters()
        self.last_state = self.env.reset()
        self.local_network.reset()
        returns = np.zeros(num_episodes)

        t = 0
        embeddings = []
        embedding_images = []
        # Main loop, execute this while T < T_max
        while episode_idx < num_episodes:
            # Get action
            action = self.local_network.get_action(self.last_state, session)
            a = action if not self.hp.lpq_single_winner else action // self.hp.ppa

            self.last_state, reward, terminal = self.env.step(a)
            returns[episode_idx] += reward

            if terminal:
                self.last_state = self.env.reset()
                self.local_network.reset()
                episode_idx += 1

            t += 1

        logger.info("Mean score {}".format(np.mean(returns)))
        writer.add_summary(make_summary_from_python_var('Evaluation/Score', np.mean(returns)), self.train_episode)
        writer.flush()
        self.train_episode += 1
        self.env.set_train()

        return np.mean(returns)

    def sample_head_space(self):
        self.synchronize_thread_parameters()
        self.last_state = self.env.reset()
        self.local_network.reset()

        heads = []

        print("Sampling head space!\n\n\n")
        for t in range(self.hp.pt_samples):

            heads.append(self.local_network.get_embedding(self.last_state, session)[0])

            _ = self.local_network.get_action(self.last_state, session)
            action = np.random.randint(self.num_actions)
            self.last_state, reward, terminal = self.env.step(action)

            if terminal:
                self.last_state = self.env.reset()
                self.local_network.reset()

        heads_stacked = np.stack(heads)
        prototypes = self.global_network.prototypes

        candidate_indices = np.random.permutation(
            np.arange(0, len(heads_stacked)))[:prototypes.get_shape().as_list()[0]]

        prototype_inits = heads_stacked[candidate_indices] + \
                          self.hp.pq_init_noise * np.random.standard_normal(prototypes.get_shape().as_list())

        prototype_ph = tf.placeholder(tf.float32, prototypes.get_shape().as_list())
        assign_prototypes = tf.assign(prototypes, prototype_ph)

        self.session.run(assign_prototypes, feed_dict={prototype_ph: prototype_inits})


if __name__ == "__main__":

    hyperparameters = HyperParameters(parse_cmd_args())
    T_var = tf.Variable(0, name='T')
    global_step = tf.assign_add(T_var, 1)

    env_name = hyperparameters.env
    n_threads = hyperparameters.n_threads

    session = tf.Session()
    #
    # session = tf_debug.LocalCLIDebugWrapperSession(session)
    #session.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

    if FLAGS.debug:
        session = tf_debug.LocalCLIDebugWrapperSession(session)
        session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    global_env = get_env(env_name)
    num_actions = global_env.num_actions()

    learning_rate_ph = tf.placeholder(tf.float32)

    if hyperparameters.optimizer == "rmsprop":
        shared_optimizer = RMSPropShared(session,
                                         learning_rate_ph,
                                         decay=hyperparameters.rms_decay,
                                         epsilon=hyperparameters.rms_epsilon,
                                         feedback=hyperparameters.feedback,
                                         global_clipping=hyperparameters.global_clipping,
                                         global_clip_norm=hyperparameters.global_clip_norm,
                                         ms_bias_correction=hyperparameters.mbc,
                                         prototype_factor=hyperparameters.prototype_factor)
    else:
        shared_optimizer = AdamShared(session, learning_rate_ph,
                                      beta1=hyperparameters.beta1,
                                      beta2=hyperparameters.beta2,
                                      feedback=hyperparameters.feedback,
                                      global_clipping=hyperparameters.global_clipping,
                                      global_clip_norm=hyperparameters.global_clip_norm,
                                      epsilon=hyperparameters.adam_epsilon)

    global_network = ActorCriticNN(num_actions=num_actions,
                                   agent_name='GLOBAL',
                                   hyper_parameters=hyperparameters,
                                   optimizer=shared_optimizer)
    shared_optimizer.set_global_theta(global_network.theta) #.build_update(global_network.theta)

    agents = [A3CAgent(env_name, global_network, 'Agent_%d' % i, session, optimizer=shared_optimizer, hp=hyperparameters)
              for i in range(n_threads)]

    writer = writer_new_event(hyperparameters, session)

    with open(os.path.join(writer.get_logdir(), 'hyper_parameters.pkl'), 'wb') as f:
        hp = hyperparameters.__dict__
        os.chdir(os.path.expanduser("~") + "/mproj/deeprl")
        hp.update({'git_description': subprocess.check_output(["git", "describe", "--always"]).decode('utf8').strip()})
        pickle.dump(hyperparameters.__dict__, f, pickle.HIGHEST_PROTOCOL)

    embedding_length = global_network.embedding_layer.get_shape().as_list()[-1]

    embedding_var = tf.Variable(np.zeros([100, embedding_length]), trainable=False,
                                name='TensorBoardEmbeddings', dtype=tf.float32)
    embedding_placeholder = tf.placeholder(tf.float32, [None, global_network.embedding_layer.get_shape().as_list()[1]])
    embedding_assign = tf.assign(embedding_var, embedding_placeholder)
    embedding_config = projector.ProjectorConfig()
    embedding = embedding_config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.sprite.image_path = os.path.join(writer.get_logdir(), 'embedding_sprite.png')
    embedding.sprite.single_image_dim.extend([160, 210])
    projector.visualize_embeddings(writer, embedding_config)

    saver = tf.train.Saver({var.name: var for var in
                            global_network.theta + shared_optimizer.ms + [T_var, embedding_var]})
    weights_path = os.path.join(writer.get_logdir(), 'model.ckpt')
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    init = tf.global_variables_initializer()
    session.run(init)

    if hyperparameters.pt_sample_init:
        agents[0].sample_head_space()

    for agent in agents:
        agent.train()

    if hyperparameters.render:
        while session.run(T_var) < hyperparameters.T_max:
            for a in agents:
                a.env.env.render()
                time.sleep(0.02 / hyperparameters.n_threads)

    for agent in agents:
        agent._train_thread.join()








