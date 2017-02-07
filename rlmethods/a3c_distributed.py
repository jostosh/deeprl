import threading
import tensorflow as tf
import numpy as np

from deeprl.common.logger import logger
from deeprl.approximators.nn import ActorCriticNN
from deeprl.common.environments import get_env
from deeprl.common.hyper_parameters_distributed import *
from deeprl.common.tensorboard import writer_new_event, make_summary_from_python_var
from deeprl.approximators.optimizers import RMSPropCustom
import time
from deeprl.common.logger import get_log_dir

import os

# cluster specification
#tf.train.Server.create_local_server()

class A3CAgent(object):
    def __init__(self, env, global_network, agent_name, optimizer):
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

        #self._train_thread = threading.Thread(target=self._train, name=agent_name)
        self.t = 1
        self.last_state = self.env.reset()
        #self.session = session
        self.agent_name = agent_name

        self.n_episodes = 1

    def train(self):
        """
        Starts a training thread
        """
        logger.info('Starting training')
        #self._train_thread.start()

    def synchronize_thread_parameters(self, session):
        """
        Synhronizes the thread network parameters with the global network
        """
        session.run(self.local_network.param_sync)


    def _train(self, session):
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

        nloops = 0
        mean_duration = 0
        n_updates = 0

        total_duration = 0.
        # Main loop, execute this while T < T_max
        t0 = time.time()
        while T < hyper_parameters.T_max:
            # [arr.fill(0) for arr in [rewards, actions, values, n_step_targets, states]]

            # A new batch begins, reset the gradients and synchronize thread-specific parameters
            #self.synchronize_thread_parameters(session)
            #logger.debug("Time for param synchronization: {}".format((end - start)))


            # Set t_start to current t
            t_start = self.t

            # Boolean to denote whether the current state is terminal
            terminal_state = False

            start = time.time()
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
                current_lr = hyper_parameters.learning_rate - lr_step * session.run(global_step)

                epr += rewards[i]
                step = session.run(increment_step)

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

            # Count the number of updates
            n_updates += 1

            if hyper_parameters.optimality_tightening:
                values[batch_len] = n_step_target
                lower_limits = []
                for j in range(batch_len):
                    current_max = -np.inf
                    for k in range(batch_len - j):
                        current_max = max(current_max,
                                          np.sum(rewards[j:j + k + 1] * (hyper_parameters.gamma ** np.arange(k + 1))) +
                                          hyper_parameters.gamma ** (k + 1) * values[j + k + 1])
                    lower_limits.append(current_max)

                upper_limits = []
                for j in range(batch_len):
                    current_min = np.inf
                    for k in range(j):
                        current_min = min(current_min, np.sum(rewards[j - k - 1:j] *
                                                              (hyper_parameters.gamma ** np.arange(-k-1, 0))) +
                                          hyper_parameters.gamma ** (-k - 1) * values[j - k - 1])
                    upper_limits.append(current_min)
            else:
                lower_limits = upper_limits = None

            # Now update the global approximator's parameters
            summaries = self.local_network.update_params(actions[:batch_len],
                                                         states[:batch_len],
                                                         current_lr,
                                                         self.last_state,
                                                         session,
                                                         n_step_targets[:batch_len],
                                                         values[:batch_len],
                                                         upper_limits=upper_limits,
                                                         lower_limits=lower_limits)

            if summaries:
                writer.add_summary(summaries, self.t)

            if terminal_state:
                if n_updates % 5 == 0:
                    logger.info('Terminal state reached (episode {}, reward {}, lr {:.5f}, T {}): resetting state'.format(
                        self.n_episodes, epr, current_lr, T))

                self.n_episodes += 1
                self.last_state = self.env.reset_random()
                epr = 0
                self.local_network.reset()

            total_duration += time.time() - t0
            mean_duration = total_duration / T

            if self.agent_name == "Agent_1" and n_updates % 50 == 0:
                logger.info("Steps per second: {}, steps per hour: {}".format(T / (time.time() - t0),
                                                                              T / (time.time() - t0) * 3600))


if __name__ == "__main__":
    hyper_parameters = HyperParameters(parse_cmd_args())
    parameter_servers = ["localhost:{}".format(hyper_parameters.port0)]
    workers = ["localhost:{}".format(str(hyper_parameters.port0 + len(parameter_servers) + i))
               for i in range(int(os.environ['SLURM_JOB_CPUS_PER_NODE']) - len(parameter_servers))]
    logger.info("We should have {} instances running".format(int(os.environ['SLURM_JOB_CPUS_PER_NODE'])))
    cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5 / float(len(workers + parameter_servers)))
    #server_def = tf.train.ServerDef(cluster=cluster.as_cluster_def(), default_session_)
    server = tf.train.Server(cluster, job_name=hyper_parameters.job_name, task_index=hyper_parameters.task_index,
                             config=tf.ConfigProto(gpu_options=gpu_options,inter_op_parallelism_threads=1,
                                                   intra_op_parallelism_threads=1))
    if hyper_parameters.job_name == "ps":
        server.join()
    else:
        with tf.Graph().as_default() as graph:
            is_chief = (hyper_parameters.task_index == 0)
            logger.info("This task is {}chief.".format("" if is_chief else 'NOT '))
            logger.info("Cluster: {}".format(cluster.as_cluster_def()))
            T = 1
            lr_step = hyper_parameters.learning_rate / hyper_parameters.T_max
            current_lr = hyper_parameters.learning_rate

            env_name = hyper_parameters.env

            global_env = get_env(env_name)
            num_actions = global_env.num_actions()

            with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % hyper_parameters.task_index,
                    cluster=cluster,
            )):
            #with tf.device("/job:ps/task:0"):

                global_step = tf.Variable(0)
                increment_step = global_step.assign_add(1, use_locking=False)
                learning_rate_ph = tf.placeholder(tf.float32)

                shared_optimizer = RMSPropCustom(None,
                                                 learning_rate_ph,
                                                 decay=hyper_parameters.rms_decay,
                                                 epsilon=hyper_parameters.rms_epsilon)
                #with tf.device('/job:ps/task:0'):
                global_network = ActorCriticNN(num_actions=num_actions,
                                               agent_name='GLOBAL',
                                               hyper_parameters=hyper_parameters,
                                               optimizer=shared_optimizer)
            #with tf.device("/job:ps/task:1"):
                shared_optimizer.set_global_theta(global_network.theta)  # .build_update(global_network.theta)

            with tf.device("/job:worker/task:{}".format(hyper_parameters.task_index)):
                #env = get_env(env_name, frames_per_state=hyper_parameters.frames_per_state, output_shape=hyper_parameters.input_shape[1:])
                agents = [] #[A3CAgent(env, global_network, 'Agent_%d' % hyper_parameters.task_index, optimizer=shared_optimizer)]
                #agents = []
                for i in range(hyper_parameters.n_threads):
                    agents.append(A3CAgent(None, global_network, 'Agent_{}'.format(str(i + hyper_parameters.n_threads *
                                                                                      hyper_parameters.task_index)),
                                           optimizer=shared_optimizer))

                init_op = tf.global_variables_initializer()
                    #writer = tf.summary.FileWriter(hyper_parameters.log_dir, graph=graph) #.train.SummaryWriter(hyper_parameters.log_dir)
                    #summary_op = tf.summary.merge_all()
                saver = tf.train.Saver()

            '''
            sv = tf.train.Supervisor(is_chief=is_chief,
                                     global_step=global_step,
                                     summary_op=None, #summary_op,
                                     #summary_writer=writer,
                                     logdir=hyper_parameters.log_dir,
                                     saver=saver,
                                     init_op=init_op,
                                     )
            '''
            with tf.Session(server.target) as sess:
                #writer_new_event(hyper_parameters, sess)
                sess.run(init_op)
                #agents[hyper_parameters.task_index]._train(sess)
                agents[0]._train(sess)
                #tf.OptimizerOptions


        #sv.stop()

        #session.run(tf.initialize_all_variables())
        #for agent in agents:
        #    agent.train()

        #if hyper_parameters.render:
        #    while T < hyper_parameters.T_max:
        #       for a in agents:
        #            a.env.env.render()
        #            time.sleep(0.02 / hyper_parameters.n_threads)







