import os
import pickle
import pprint
from argparse import ArgumentParser

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from deeprl.train import A3CAgent
from deeprl.common.environments import get_env
from deeprl.common.hyper_parameters import HyperParameters

plt.style.use('ggplot')
import matplotlib as mpl
import signal
import sys
from copy import deepcopy
import matplotlib.gridspec as gridspec



mpl.rc('text', usetex=True)
mpl.rc('text', color='white')
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
mpl.rc('xtick', labelsize=14, color='white')
mpl.rc('ytick', labelsize=14, color='white')
mpl.rc('figure', facecolor="#000000")
mpl.rc('axes', facecolor='#111111', labelcolor='white', labelsize=16, titlesize=20, prop_cycle=mpl.cycler(color=['ffff00']))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--model_dir")
    parser.add_argument("-n", "--n_episodes", default=5)
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

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_out = os.path.join(args.model_dir, 'video.mp4')
    out = cv2.VideoWriter(video_out, fourcc, 15., (1920, 1080), isColor=True)


    def signal_handler(signal, frame):
        print("Writing video")
        out.release()
        print("Closing...")
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    env = get_env(hp.env, hp.frames_per_state, hp.input_shape[1:])

    pprint.pprint([op.name for op in sess.graph.get_operations()])

    fig, axes = plt.subplots(ncols=1, nrows=2)
    canvas = fig.canvas
    ax1, ax2 = axes.ravel()
    my_dpi = 96
    fig2 = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi)
    gs = gridspec.GridSpec(3, 6)
    ax_conv1 = plt.subplot(gs[:, 0])
    ax_conv2 = plt.subplot(gs[:, 1])
    ax_fc3 = plt.subplot(gs[:, 2])
    ax_lstm = plt.subplot(gs[:, 3])
    ax_game = plt.subplot(gs[0, 4])
    ax_val = plt.subplot(gs[1, 4])
    ax_pol = plt.subplot(gs[2, 4])

    ax_input = plt.subplot(gs[0, 5])
    ax_prediction = plt.subplot(gs[1, 5])

    display_lstm = hp.model == "a3c_lstm"
    background_color = (0.1, 0.1, 0.1)

    def set_color(im, c):
        im[:, :, 2] = c[2]
        im[:, :, 1] = c[1]
        im[:, :, 0] = c[0]

    conv1_image = np.zeros((20 * 8 + 8, 1 + 20 * 2 + 2, 3), dtype='float32')
    conv2_image = np.zeros((9 * 16 + 16, 9 * 2 + 2, 3), dtype='float32')
    fc1_image = np.zeros((34, 9, 3), dtype='float32')
    lstm_image = np.zeros((34, 9, 3), dtype='float32')
    frame_prediction_image = np.zeros((84*2+3, 86, 3), dtype='float32')
    set_color(frame_prediction_image, background_color)
    set_color(conv1_image, background_color)
    set_color(conv2_image, background_color)
    set_color(fc1_image, background_color)
    set_color(lstm_image, background_color)

    slice_ms1 = np.ones((8, 2))
    slice_ms2 = np.ones((16, 2))

    iter = 1

    for episode_idx in range(args.n_episodes):
        state = env.reset_random()
        agent.local_network.reset()
        terminal = False
        value_buffer = []
        max_val = -1000
        min_val = 1000
        episode_step = 0
        print("Now at episode {}".format(episode_idx))

        while not terminal:
            rnn_state = None
            if hp.frame_prediction and agent.local_network.recurrent:
                rnn_state = deepcopy(agent.local_network.lstm_state_numeric)

            value, action, conv1_out, conv2_out, pi, fc1, lstm = agent.local_network\
                .get_value_and_action_and_visualize(state, sess)

            if hp.frame_prediction:
                predicted_frame = agent.local_network.get_frame_prediction(state, action, sess, rnn_state)

            max_val = max(value + abs(value) * 0.1, max_val - abs(max_val) * 0.001)
            min_val = min(value - abs(value) * 0.1, min_val + abs(min_val) * 0.001)

            state, r, terminal, _ = env.step(action)

            value_buffer.append(value)
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

            env_image = env.env._get_image()[:, :, ::-1]
            fc1_image[1:-1, 0:8, :] = cv2.cvtColor(np.reshape(fc1, (32, 8)), cv2.COLOR_GRAY2RGB)
            if display_lstm:
                lstm_image[1:-1, 0:8, :] = cv2.cvtColor(np.reshape((lstm + 1.) / 2., (32, 8)), cv2.COLOR_GRAY2RGB)
            if episode_step > 20:
                [ax.cla() for ax in [ax_conv1, ax_conv2, ax_fc3, ax_lstm, ax_game, ax_input, ax_prediction,
                                     ax_pol, ax_val]]
                [ax.axis('off') for ax in [ax_conv1, ax_conv2, ax_fc3, ax_lstm, ax_game, ax_input, ax_prediction]]

                ax_conv1.imshow(conv1_image)
                ax_conv1.set_title('Conv1')
                ax_conv2.imshow(conv2_image)
                ax_conv2.set_title('Conv2')
                ax_fc3.imshow(np.reshape(fc1, (32, 8)), cmap='gray', interpolation='nearest')
                ax_fc3.set_title('FC3')
                ax_lstm.imshow(np.reshape((lstm + 1.) / 2., (32, 8)), cmap='gray', interpolation='nearest')
                ax_lstm.set_title('LSTM4')

                ax_game.imshow(env_image)
                ax_game.set_title("Game env")

                ax_input.imshow(state[-1, :, :], cmap='gray')
                ax_input.set_title("Preprocessed frame")
                if hp.frame_prediction:
                    ax_prediction.imshow(predicted_frame[0, 0, :, :], cmap='gray')
                    ax_prediction.set_title("Predicted frame")
                bar = ax_pol.bar(range(env.num_actions()), pi)
                ax_pol.set_title("Policy")  # r"\pi(s,a)")
                ax_pol.set_ylabel(r"$\pi(s,a)$")
                ax_pol.set_ylim([0., 1.])
                ax_pol.set_xticklabels(env.env.get_action_meanings(), rotation=90)
                xticks_pos = [0.65 * patch.get_width() + patch.get_xy()[0] for patch in bar]
                ax_pol.set_xticks(xticks_pos)
                line, = ax_val.plot(range(episode_step - 19, episode_step + 1), value_buffer[-20:], linewidth=2)
                ax_val.set_title("Value estimate")
                ax_val.set_ylabel(r"$v(s)$")
                ax_val.set_ylim([min_val, max_val])
                ax_val.set_xlim([episode_step - 19, episode_step])
                ax_val.set_xlabel(r"Step")
                plt.tight_layout()

                canvas2 = fig2.canvas
                canvas2.draw()
                plot_image = np.fromstring(canvas2.tostring_rgb(), dtype='uint8')
                plot_image = plot_image.reshape(canvas2.get_width_height()[::-1] + (3,))
                out.write(plot_image.astype('u1'))

            iter += 1
            episode_step += 1


