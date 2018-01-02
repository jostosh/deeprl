import tensorflow as tf
from deeprl.common.environments import get_env
import time
import subprocess
from deeprl.common.config import load_config, Config
from deeprl.approximators.approximators import get_approximator
from deeprl.approximators.optimizers.shared import RMSPropOptimizer
from deeprl.agent import A3CAgent
import os
import pprint


def store_config():
    with open(os.path.join(writer.get_logdir(), 'hyper_parameters.txt'), 'w') as f:
        os.chdir(os.path.expanduser("~") + "/mproj/deeprl")
        setattr(
            Config, 'git_description', subprocess.check_output(["git", "describe", "--always"]).decode('utf8').strip()
        )
        pprint.pprint(Config.__dict__, f)


if __name__ == "__main__":
    load_config()
    pprint.pprint(Config.__dict__)
    T_var = tf.Variable(1, name='T')
    global_step = tf.assign_add(T_var, 1)
    session = tf.Session()
    global_env = get_env()
    num_actions = global_env.get_num_actions()

    learning_rate_ph = tf.placeholder(tf.float32)

    shared_optimizer = RMSPropOptimizer(
        session, learning_rate_ph, decay=Config.rms_decay, epsilon=Config.rms_epsilon
    )
    global_network = get_approximator(session, global_env.get_num_actions(), shared_optimizer, global_approximator=None,
                                      name='Global', global_t=T_var)
    shared_optimizer.set_global_theta(global_network.theta)

    agents = []
    saver = tf.train.Saver({var.name: var for var in global_network.theta + shared_optimizer.ms + [T_var]})
    writer = tf.summary.FileWriter(Config.log_dir, session.graph)

    for i in range(Config.n_threads):
        approximator = get_approximator(
            session, global_env.get_num_actions(), shared_optimizer, global_approximator=global_network,
            name='Agent{}'.format(i), global_t=T_var
        )
        agents.append(A3CAgent(
            approximator=approximator, session=session, global_step=global_step,
            saver=saver, writer=writer, name='Agent{}'.format(i), global_time=T_var
        ))

    store_config()
    weights_path = os.path.join(Config.log_dir, 'model.ckpt')
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    init = tf.global_variables_initializer()
    session.run(init)

    for agent in agents:
        agent.train()

    if Config.render:
        while session.run(T_var) < Config.T_max:
            for a in agents:
                a.env.env.render()
                time.sleep(0.02 / Config.n_threads)

    for agent in agents:
        agent.thread.join()

