import threading
import tensorflow as tf
import numpy as np

from deeprl.common.logger import logger
from deeprl.approximators.nn import ActorCriticNN
from deeprl.common.environments import get_env
from deeprl.common.hyper_parameters import *
from deeprl.common.tensorboard import writer_new_event, make_summary_from_python_var
from deeprl.approximators.optimizers import RMSPropCustom
import time
import pickle
from random import shuffle
from tensorflow.contrib.tensorboard.plugins import projector
import scipy.misc


class A3CAgent(object):

    def __init__(self, env_name, global_network, agent_name, session, optimizer):
        """
        Initializes an Asynchronous Advantage Actor-Critic agent (A3C).
        :param env_name:        Name of the environment
        :param global_network:  Global network to use for updates and synchronization
        :param agent_name:      Name of this agent
        :param session:         TensorFlow session
        """
        self.env = get_env(env_name,
                           frames_per_state=hyperparameters.frames_per_state,
                           output_shape=hyperparameters.input_shape[1:])
        self.num_actions = self.env.num_actions()

        self.local_network = ActorCriticNN(num_actions=self.num_actions,
                                           agent_name=agent_name,
                                           optimizer=optimizer,
                                           hyper_parameters=hyperparameters,
                                           global_network=global_network)

        self._train_thread = threading.Thread(target=self._train, name=agent_name)
        self.t = 1
        self.last_state = self.env.reset()
        self.session = session
        self.agent_name = agent_name

        self.n_episodes = 1
        self.train_episode = 0


    def train(self):
        """
        Starts a training thread
        """
        logger.info('Starting training')
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
        global current_lr, lr_step
        # Initialize the reward, action and observation arrays
        rewards = np.zeros(hyperparameters.t_max, dtype='float')
        actions = np.zeros(hyperparameters.t_max, dtype='int')
        values = np.zeros(hyperparameters.t_max, dtype='float')
        n_step_targets = np.zeros(hyperparameters.t_max, dtype='float')
        states = np.zeros((hyperparameters.t_max,) + self.env.state_shape(), dtype='float')

        epr = 0

        all_rewards = []
        all_values = []

        nloops = 0
        mean_duration = 0

        total_duration = 0.

        T = session.run(T_var)
        last_checkpoint = T

        # Main loop, execute this while T < T_max
        while T < hyperparameters.T_max:
            #[arr.fill(0) for arr in [rewards, actions, values, n_step_targets, states]]
            t0 = time.time()

            # A new batch begins, reset the gradients and synchronize thread-specific parameters
            self.synchronize_thread_parameters()

            # Set t_start to current t
            t_start = self.t

            # Boolean to denote whether the current state is terminal
            terminal_state = False

            # Now take steps following the thread-specific policy given by self.theta and self.theta_v
            while not terminal_state and self.t - t_start != hyperparameters.t_max:

                # Index of current step
                i = self.t - t_start
                # Set the current observation
                states[i] = self.last_state
                # Get the corresponding value and action. This is done simultaneously such that the approximators only
                # has to perform a single forward pass.
                values[i], actions[i] = self.local_network.get_value_and_action(self.last_state, session)
                # Perform step in environment and obtain rewards and observations
                self.last_state, rewards[i], terminal_state, info = self.env.step(actions[i])
                # Increment time counters
                self.t += 1
                T = session.run(global_step)
                current_lr -= lr_step

                epr += rewards[i]

                all_rewards.append(rewards[i])
                all_values.append(values[i])

            if hyperparameters.clip_rewards:
                # Reward clipping helps to stabilize training
                rewards = np.clip(rewards, -1.0, 1.0)

            # Initialize the n-step return
            n_step_target = 0 if terminal_state else self.local_network.get_value(self.last_state, session)

            batch_len = self.t - t_start

            # Forward view of n-step returns, start from i == t_max - 1 and go to i == 0
            for i in reversed(range(batch_len)):
                # Straightforward accumulation of rewards
                n_step_target = rewards[i] + hyperparameters.gamma * n_step_target
                n_step_targets[i] = n_step_target

            # Now update the global approximator's parameters
            summaries = self.local_network.update_params(n_step_targets[:batch_len],
                                                         actions[:batch_len],
                                                         states[:batch_len],
                                                         values[:batch_len],
                                                         learning_rate_ph,
                                                         current_lr,
                                                         self.last_state,
                                                         session)
            if summaries:
                writer.add_summary(summaries, self.t)

            if terminal_state:
                logger.info('Terminal state reached (episode {}, reward {}, T {}): resetting state'.format(
                    self.n_episodes, epr, T))

                writer.add_summary(make_summary_from_python_var('{}/Score'.format(self.agent_name), epr), T)
                self.n_episodes += 1
                self.last_state = self.env.reset_random()
                epr = 0
                self.local_network.reset()

                all_values = []
                all_rewards = []

            duration = (time.time() - t0) / batch_len
            total_duration += time.time() - t0

            nloops += 1
            mean_duration = (nloops - 1) / float(nloops) * mean_duration + duration / float(nloops)
            logger.info("Mean duration {}, or {} per hour".format(mean_duration,
                                                                   3600 / mean_duration * n_threads))

            if T - last_checkpoint > hyperparameters.evaluation_interval \
                    and self.agent_name == 'Agent_0':
                self.evaluate(50)
                last_checkpoint = T / int(hyperparameters.evaluation_interval / 10) * \
                                  int(hyperparameters.evaluation_interval / 10) # round to the nearest 1e6
                logger.info("Storing weights at {}".format(weights_path))
                saver.save(session, weights_path, global_step=global_step)
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
        self.last_state = self.env.reset_random()
        self.local_network.reset()
        self.env.set_test()
        returns = np.zeros(num_episodes)

        t = 0
        embeddings = []
        embedding_images = []
        # Main loop, execute this while T < T_max
        while episode_idx < num_episodes:
            # Get action
            _, action = self.local_network.get_value_and_action(self.last_state, session)
            self.last_state, reward, terminal, info = self.env.step(action)
            returns[episode_idx] += reward

            if terminal:
                self.last_state = self.env.reset_random()
                self.local_network.reset()
                episode_idx += 1

            if t % 1 == 0:
                embeddings.append(self.local_network.get_embedding(self.last_state, session))
                embedding_images.append(self.env.env._get_image())

                if len(embeddings) > 100:
                    deletion_index = np.random.randint(100)
                    del embeddings[deletion_index]
                    del embedding_images[deletion_index]

            t += 1

        zipped_embeddings = list(zip(embeddings, embedding_images))
        shuffle(zipped_embeddings)
        embeddings, embedding_images = zip(*zipped_embeddings[:100])

        frame_height = 210
        frame_width = 160
        sprite_image = np.empty((10 * frame_height, 10 * frame_width, 3))

        image_index = 0
        for i in range(0, sprite_image.shape[0], frame_height):
            for j in range(0, sprite_image.shape[1], frame_width):
                sprite_image[i:i+frame_height, j:j+frame_width, :] = embedding_images[image_index]
                image_index += 1

        scipy.misc.imsave(os.path.join(writer.get_logdir(), 'embedding_sprite.png'), sprite_image)
        session.run(embedding_assign, feed_dict={embedding_placeholder: np.concatenate(embeddings, axis=0)})
        logger.info("Mean score {}".format(np.mean(returns)))
        writer.add_summary(make_summary_from_python_var('Evaluation/Score', np.mean(returns)),
                           self.train_episode)
        self.train_episode += 1
        self.env.set_train()

def upper_bounds(v_t, r_t, v_end):
    T = len(r_t)

    R_t = np.array(r_t)

    g = hyperparameters.gamma

    R_t[-1] += g * v_end
    for i in reversed(range(T - 1)):
        R_t[i] += g * R_t[i+1]

    return [g ** (-t2) * min([g * v_t[t1] + R_t[-t2] - R_t[t1] for t1 in range(T - t2)])
            for t2 in range(hyperparameters.t_max)]


if __name__ == "__main__":

    hyperparameters = HyperParameters(parse_cmd_args())
    T_var = tf.Variable(0, name='T')
    global_step = tf.assign_add(T_var, 1)

    lr_step = hyperparameters.learning_rate / hyperparameters.T_max
    current_lr = hyperparameters.learning_rate

    env_name = hyperparameters.env
    n_threads = hyperparameters.n_threads

    session = tf.Session()
    global_env = get_env(env_name)
    num_actions = global_env.num_actions()

    learning_rate_ph = tf.placeholder(tf.float32)

    shared_optimizer = RMSPropCustom(session,
                                     learning_rate_ph,
                                     decay=hyperparameters.rms_decay,
                                     epsilon=hyperparameters.rms_epsilon)

    global_network = ActorCriticNN(num_actions=num_actions,
                                   agent_name='GLOBAL',
                                   hyper_parameters=hyperparameters,
                                   optimizer=shared_optimizer)
    shared_optimizer.set_global_theta(global_network.theta) #.build_update(global_network.theta)

    agents = [A3CAgent(env_name, global_network, 'Agent_%d' % i, session, optimizer=shared_optimizer)
              for i in range(n_threads)]

    writer = writer_new_event(hyperparameters, session)

    with open(os.path.join(writer.get_logdir(), 'hyper_parameters.pkl'), 'wb') as f:
        pickle.dump(hyperparameters.__dict__, f, pickle.HIGHEST_PROTOCOL)

    embedding_var = tf.Variable(np.zeros([100, 256]), trainable=False,
                                name='TensorBoardEmbeddings', dtype=tf.float32)
    embedding_placeholder = tf.placeholder(tf.float32, [None, global_network.embedding_layer.get_shape().as_list()[1]])
    embedding_assign = tf.assign(embedding_var, embedding_placeholder)
    embedding_config = projector.ProjectorConfig()
    embedding = embedding_config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.sprite.image_path = os.path.join(writer.get_logdir(), 'embedding_sprite.png')
    embedding.sprite.single_image_dim.extend([160, 210])
    projector.visualize_embeddings(writer, embedding_config)

    saver = tf.train.Saver(global_network.theta + shared_optimizer.g_moving_average + [T_var, embedding_var])
    weights_path = os.path.join(writer.get_logdir(), 'model.ckpt')
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    session.run(tf.global_variables_initializer())
    for agent in agents:
        agent.train()

    if hyperparameters.render:
        while session.run(T_var) < hyperparameters.T_max:
            for a in agents:
                a.env.env.render()
                time.sleep(0.02 / hyperparameters.n_threads)








