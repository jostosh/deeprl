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
                           frames_per_state=hyper_parameters.frames_per_state,
                           output_shape=hyper_parameters.input_shape[1:])
        self.num_actions = self.env.num_actions()

        self.local_network = ActorCriticNN(num_actions=self.num_actions,
                                           agent_name=agent_name,
                                           optimizer=optimizer,
                                           hyper_parameters=hyper_parameters,
                                           global_network=global_network)

        self._train_thread = threading.Thread(target=self._train, name=agent_name)
        self.t = 1
        self.last_state = self.env.reset()
        self.session = session
        self.agent_name = agent_name

        self.n_episodes = 1


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
        global T, current_lr, lr_step
        # Initialize the reward, action and observation arrays
        rewards = np.zeros(hyper_parameters.t_max, dtype='float')
        actions = np.zeros(hyper_parameters.t_max, dtype='int')
        values = np.zeros(hyper_parameters.t_max, dtype='float')
        n_step_targets = np.zeros(hyper_parameters.t_max, dtype='float')
        states = np.zeros((hyper_parameters.t_max,) + self.env.state_shape(), dtype='float')

        epr = 0

        all_rewards = []
        all_values = []

        nloops = 0
        mean_duration = 0

        total_duration = 0.

        # Main loop, execute this while T < T_max
        while T < hyper_parameters.T_max:
            #[arr.fill(0) for arr in [rewards, actions, values, n_step_targets, states]]
            t0 = time.time()

            # A new batch begins, reset the gradients and synchronize thread-specific parameters
            self.synchronize_thread_parameters()

            # Set t_start to current t
            t_start = self.t

            # Boolean to denote whether the current state is terminal
            terminal_state = False

            # Now take steps following the thread-specific policy given by self.theta and self.theta_v
            while not terminal_state and self.t - t_start != hyper_parameters.t_max:

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
                T += 1
                current_lr -= lr_step

                epr += rewards[i]

                all_rewards.append(rewards[i])
                all_values.append(values[i])

            if hyper_parameters.clip_rewards:
                # Reward clipping helps to stabilize training
                rewards = np.clip(rewards, -1.0, 1.0)

            # Initialize the n-step return
            n_step_target = 0 if terminal_state else self.local_network.get_value(self.last_state, session)

            batch_len = self.t - t_start

            # Forward view of n-step returns, start from i == t_max - 1 and go to i == 0
            for i in reversed(range(batch_len)):
                # Straightforward accumulation of rewards
                n_step_target = rewards[i] + hyper_parameters.gamma * n_step_target
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
            writer.add_summary(summaries, self.t)

            if terminal_state:
                logger.info('Terminal state reached (episode {}, reward {}): resetting state'.format(self.n_episodes, epr))

                writer.add_summary(make_summary_from_python_var('{}/EpisodeReward'.format(self.agent_name), epr), T)
                self.n_episodes += 1
                self.last_state = self.env.reset()
                epr = 0
                self.local_network.reset()

                all_values = []
                all_rewards = []

            duration = (time.time() - t0) / batch_len
            total_duration += time.time() - t0

            nloops += 1
            mean_duration = (nloops - 1) / float(nloops) * mean_duration + duration / float(nloops)
            #logger.info("Mean duration {}, or {} per hour".format(mean_duration,
            #                                                     3600 / mean_duration * n_threads))


def upper_bounds(v_t, r_t, v_end):
    T = len(r_t)

    R_t = np.array(r_t)

    g = hyper_parameters.gamma

    R_t[-1] += g * v_end
    for i in reversed(range(T - 1)):
        R_t[i] += g * R_t[i+1]

    return [g ** (-t2) * min([g * v_t[t1] + R_t[-t2] - R_t[t1] for t1 in range(T - t2)])
            for t2 in range(hyper_parameters.t_max)]


if __name__ == "__main__":

    hyper_parameters = HyperParameters(parse_cmd_args())
    T = 1
    lr_step = hyper_parameters.learning_rate / hyper_parameters.T_max
    current_lr = hyper_parameters.learning_rate

    env_name = hyper_parameters.env
    n_threads = hyper_parameters.n_threads

    global_env = get_env(env_name)
    num_actions = global_env.num_actions()

    session = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=hyper_parameters.n_threads,
        intra_op_parallelism_threads=hyper_parameters.n_threads))
    learning_rate_ph = tf.placeholder(tf.float32)

    shared_optimizer = RMSPropCustom(session,
                                     learning_rate_ph,
                                     decay=hyper_parameters.rms_decay,
                                     epsilon=hyper_parameters.rms_epsilon)

    global_network = ActorCriticNN(num_actions=num_actions,
                                   agent_name='GLOBAL',
                                   hyper_parameters=hyper_parameters,
                                   optimizer=shared_optimizer)
    shared_optimizer.set_global_theta(global_network.theta) #.build_update(global_network.theta)

    agents = [A3CAgent(env_name, global_network, 'Agent_%d' % i, session, optimizer=shared_optimizer)
              for i in range(n_threads)]

    writer = writer_new_event(hyper_parameters, session)
    
    session.run(tf.initialize_all_variables())
    for agent in agents:
        agent.train()

    if hyper_parameters.render:
        while T < hyper_parameters.T_max:
            for a in agents:
                a.env.env.render()
                time.sleep(0.02 / hyper_parameters.n_threads)








