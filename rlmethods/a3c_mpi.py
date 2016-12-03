import tensorflow as tf
import numpy as np

from deeprl.common.logger import logger
from deeprl.approximators.nn import ActorCriticNN
from deeprl.common.environments import get_env
from deeprl.common.hyper_parameters import *
from deeprl.common.tensorboard import writer_new_event, make_summary_from_python_var
from deeprl.approximators.optimizers import RMSPropCustom
import time
from mpi4py import MPI



def params_to_1d(params):
    return np.concatenate([p.reshape((p.size,)) for p in params])


def one_d_to_params(params):
    return [p.reshape(shape) for p, shape in zip(np.split(params, param_offsets), param_shapes)]


def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

tags = enum('DELTA', 'SYNCPARAM', 'START')


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

        #self._train_thread = threading.Thread(target=self._train, name=agent_name)
        self.t = 1
        self.last_state = self.env.reset()
        self.session = session
        self.agent_name = agent_name

        self.n_episodes = 1

        self.new_theta = [tf.placeholder(tf.float32, th.get_shape().as_list()) for th in self.local_network.theta]
        self.assign_theta = [tf.assign(th, new_th) for th, new_th in zip(self.local_network.theta, self.new_theta)]

        self.req = None

    def synchronize_thread_parameters(self):
        """
        Synhronizes the thread network parameters with the global network
        """
        #comm.Send([None, MPI.FLOAT], dest=0, tag=tags.SYNCPARAM)
        #comm.Recv([self.buffer, MPI.FLOAT], source=0, tag=MPI.ANY_TAG, status=status)

        comm.Sendrecv([None, MPI.FLOAT], dest=0, sendtag=tags.SYNCPARAM, source=0, recvbuf=[self.buffer, MPI.FLOAT],
                      status=status, recvtag=MPI.ANY_TAG)
        theta = one_d_to_params(self.buffer)

        session.run(self.assign_theta,
                    feed_dict={
                        new_th_ph: new_th_num for new_th_ph, new_th_num in zip(self.new_theta, theta)
                    })


    def _train(self):
        """
        This is the thread function for a single A3C agent. The pseudo-code can be found in "Asynchronous Methods for
        Reinforcement Learning" by Mnih et al (2016): https://arxiv.org/abs/1602.01783

        It executes the actor-critic method with asynchronous updates and n-step returns in a forward view
        """
        self.buffer = np.empty(bufferlen, dtype='float32')
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

        delta_req = None

        nloops = 0
        mean_duration = 0

        # Main loop, execute this while T < T_max
        while T < hyper_parameters.T_max:
            #[arr.fill(0) for arr in [rewards, actions, values, n_step_targets, states]]

            # A new batch begins, reset the gradients and synchronize thread-specific parameters
            self.synchronize_thread_parameters()

            # Set t_start to current t
            t_start = self.t

            # Boolean to denote whether the current state is terminal
            terminal_state = False

            t0 = time.time()

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
            summaries, delta = self.local_network.compute_delta(n_step_targets[:batch_len],
                                                         actions[:batch_len],
                                                         states[:batch_len],
                                                         values[:batch_len],
                                                         learning_rate_ph,
                                                         current_lr,
                                                         self.last_state,
                                                         session)

            #if self.req:
            #    self.req.wait()
            self.req = comm.Isend([params_to_1d(delta), MPI.FLOAT], dest=0, tag=tags.DELTA)
            self.req.Free()

            #writer.add_summary(summaries, self.t)

            if terminal_state:
                logger.info('Terminal state reached (episode {}, reward {}): resetting state'.format(self.n_episodes, epr))

                #writer.add_summary(make_summary_from_python_var('{}/EpisodeReward'.format(self.agent_name), epr), T)
                self.n_episodes += 1
                self.last_state = self.env.reset()
                epr = 0
                self.local_network.reset()

                all_values = []
                all_rewards = []

            duration = time.time() - t0

            nloops += 1
            mean_duration = (nloops - 1) / float(nloops) * mean_duration + duration / float(nloops)
            logger.info("Mean duration {}".format(mean_duration / batch_len))


def upper_bounds(v_t, r_t, v_end):
    T = len(r_t)

    R_t = np.array(r_t)

    g = hyper_parameters.gamma

    R_t[-1] += g * v_end
    for i in reversed(range(T - 1)):
        R_t[i] += g * R_t[i+1]

    return [g ** (-t2) * min([g * v_t[t1] + R_t[-t2] - R_t[t1] for t1 in range(T - t2)])
            for t2 in range(hyper_parameters.t_max)]


def parameter_server():
    import time
    send_req = None

    buffers = [np.empty(bufferlen, 'float32') for _ in range(1, comm.size)]
    recv_reqs = [comm.Irecv([buffers[i - 1], MPI.FLOAT], source=i, tag=MPI.ANY_TAG) for i in range(1, comm.size)]
    #logger.info("Recieve request sent")

    statuses = [MPI.Status() for _ in range(1, comm.size)]

    send_buffers = [None] * (comm.size - 1)

    while True:
        MPI.Request.waitany(recv_reqs, status=status)
        #ret = MPI.Request.Waitsome(recv_reqs, statuses=statuses)

        #synch_indices = [i for i in ret if statuses[i].tag == tags.SYNCPARAM]
        #delta_indices = [i for i in ret if statuses[i].tag == tags.DELTA]

        #for i in synch_indices:
        #    send_buffers[source - 1] = params_to_1d(session.run(global_network.theta))

        #recv_req.wait(status=status)
        source = status.Get_source()
        data = np.copy(buffers[source - 1])
        tag = status.Get_tag()
        recv_reqs[source - 1] = comm.Irecv([buffers[source - 1], MPI.FLOAT], source=source, tag=MPI.ANY_TAG)

        #data = np.empty(bufferlen, 'float32')
        #comm.Recv([data, MPI.FLOAT], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

        if tag == tags.SYNCPARAM:
            send_buffers[source - 1] = params_to_1d(session.run(global_network.theta))
            send_req = comm.Isend([send_buffers[source - 1], MPI.FLOAT], dest=source, tag=tags.SYNCPARAM)
            send_req.Free()
        elif tag == tags.DELTA:
            theta_delta = one_d_to_params(data)

            fdict = {
                grad_ph: grad_num for grad_ph, grad_num in zip(shared_optimizer.gradients, theta_delta)
            }
            fdict[learning_rate_ph] = current_lr
            session.run(shared_optimizer.minimize, feed_dict=fdict)



if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.rank

    hyper_parameters = HyperParameters(parse_cmd_args())
    T = 1
    lr_step = hyper_parameters.learning_rate / hyper_parameters.T_max
    current_lr = hyper_parameters.learning_rate

    learning_rate_ph = tf.placeholder(tf.float32)

    env_name = hyper_parameters.env
    n_threads = hyper_parameters.n_threads

    status = MPI.Status()

    if rank == 0:

        global_env = get_env(env_name)
        num_actions = global_env.num_actions()
        session = tf.Session()
        shared_optimizer = RMSPropCustom(session,
                                         learning_rate_ph,
                                         decay=hyper_parameters.rms_decay,
                                         epsilon=hyper_parameters.rms_epsilon)

        global_network = ActorCriticNN(num_actions=num_actions,
                                       agent_name='GLOBAL',
                                       hyper_parameters=hyper_parameters,
                                       optimizer=shared_optimizer)
        shared_optimizer.build_update(global_network.theta)

        param_offsets = np.cumsum([np.prod(p.get_shape().as_list()) for p in global_network.theta[:-1]])
        param_shapes = [p.get_shape().as_list() for p in global_network.theta]


        bufferlen = sum([np.prod(s) for s in param_shapes])

        session.run(tf.initialize_all_variables())

        parameter_server()

        #parameter_threads = [threading.Thread(target=parameter_server, args=(r)) for r in range(1, comm.size)]

    else:
        session = tf.Session()
        agent = A3CAgent(env_name, 'mpi', 'Agent_%d' % rank, session, optimizer=None)

        writer = tf.summary.FileWriter(os.path.join(hyper_parameters.logdir, 'Agent_%d' % rank), session.graph) #writer_new_event(hyper_parameters, session)

        session.run(tf.global_variables_initializer())

        param_offsets = np.cumsum([np.prod(p.get_shape().as_list()) for p in agent.local_network.theta[:-1]])
        param_shapes = [p.get_shape().as_list() for p in agent.local_network.theta]
        bufferlen = sum([np.prod(s) for s in param_shapes])


        agent._train()





