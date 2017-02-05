import tensorflow as tf
from argparse import ArgumentParser
from deeprl.rlmethods.a3c import A3CAgent
import pickle
import os
from deeprl.common.hyper_parameters import HyperParameters
from deeprl.common.environments import get_env
import pprint
from tensorflow.contrib.framework.python.framework.checkpoint_utils import list_variables
import time

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--model_dir")
    parser.add_argument("-n", "--n_episodes", default=10)
    args = parser.parse_args()

    sess = tf.Session()

    with open(os.path.join(args.model_dir, 'hyper_parameters.pkl'), 'rb') as f:
        hp_raw = pickle.load(f)
        hp = HyperParameters(hp_raw) if not isinstance(hp_raw, HyperParameters) else hp_raw

    agent = A3CAgent(env_name=hp.env, global_network=None, agent_name='GLOBAL', session=sess, optimizer=None, hp=hp)
    checkpoint = tf.train.get_checkpoint_state(args.model_dir)
    model_checkpoint_path = os.path.join(args.model_dir, os.path.basename(checkpoint.model_checkpoint_path))

    vars = {v.name: v for v in agent.local_network.theta}
    saver = tf.train.Saver(var_list=agent.local_network.theta)
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_checkpoint_path)

    env = get_env(hp.env, hp.frames_per_state, hp.input_shape[1:])

    for _ in range(args.n_episodes):
        state = env.reset_random()
        agent.local_network.reset()
        terminal = False
        while not terminal:
            action = agent.local_network.get_action(state, sess)
            state, r, terminal, _ = env.step(action)
            env.env.render()
            time.sleep(1/60.)


