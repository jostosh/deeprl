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
import numpy as np
import cv2


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

    pprint.pprint([op.name for op in sess.graph.get_operations()])

    font = cv2.FONT_HERSHEY_PLAIN

    title_image = np.zeros((20, (1 + 20*2+2 + 9*2+2)*4, 3), dtype='float32')
    title_image[:, :, 2] = 204/255.
    title_image[:, :, 1] = 255/255.
    title_image[:, :, 0] = 204/255.

    cv2.putText(title_image, "Conv1", (5, 15), font, 1, (0., 0., 0.), bottomLeftOrigin=False, thickness=2)
    cv2.putText(title_image, "Conv2", (5 + (20*2 + 2) * 4, 15), font, 1, (0., 0., 0.), bottomLeftOrigin=False,
                thickness=2)

    conv1_image = np.zeros((20 * 8 + 8, 1 + 20 * 2 + 2, 3), dtype='float32')
    conv2_image = np.zeros((9 * 16 + 16 + 8, 9 * 2 + 2, 3), dtype='float32')

    conv1_image[:, :, 2] = 204/255.
    conv2_image[:, :, 2] = 204/255.
    conv1_image[:, :, 1] = 255/255.
    conv2_image[:, :, 1] = 255/255.
    conv1_image[:, :, 0] = 204/255.
    conv2_image[:, :, 0] = 204/255.

    slice_ms1 = np.ones((8, 2))
    slice_ms2 = np.ones((16, 2))

    iter = 1
    for _ in range(args.n_episodes):
        state = env.reset_random()
        agent.local_network.reset()
        terminal = False
        while not terminal:
            value, action, conv1_out, conv2_out = agent.local_network.get_value_and_action_and_visualize(state, sess)
            state, r, terminal, _ = env.step(action)

            for i in range(8):
                for j in range(2):
                    idx = i*2 + j
                    slice = conv1_out[0, :, :, idx]
                    slice_ms1[i, j] = (np.max(slice) + slice_ms1[i, j] * (iter-1)) / iter
                    mslice = slice_ms1[i, j]
                    im = cv2.cvtColor(slice / (mslice if mslice != 0 else 1.), cv2.COLOR_GRAY2RGB)
                    conv1_image[i*20+i:(i+1)*20+i, 1+j*20+j:1+(j+1)*20+j, :] = im

            for i in range(16):
                for j in range(2):
                    idx = i*2 + j
                    slice = conv2_out[0, :, :, idx]
                    slice_ms2[i, j] = (np.max(slice) + slice_ms2[i, j] * (iter-1)) / iter
                    mslice = slice_ms2[i, j]
                    im = cv2.cvtColor(slice / (mslice if mslice != 0 else 1.), cv2.COLOR_GRAY2RGB)
                    conv2_image[i*9+i:(i+1)*9+i, j*9+j:(j+1)*9+j, :] = im

            all_convs = cv2.resize(np.concatenate((conv1_image, conv2_image), axis=1), None, fx=4, fy=4)
            all_convs = np.concatenate((title_image, all_convs), axis=0)
            cv2.imshow('all_convs', all_convs)
            cv2.waitKey(1)

            iter += 1
            env.env.render()
            time.sleep(1/60/4)


