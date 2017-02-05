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
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import matplotlib as mpl
import colorlover as cl

#mpl.rc('text', usetex=True)
#mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
mpl.rc('xtick', labelsize=20)
mpl.rc('ytick', labelsize=20)
mpl.rc('figure', facecolor="#ccffcc")
mpl.rc('axes', facecolor="#f2f2f2")
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


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

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = os.path.join(args.model_dir, 'video.avi')
    out = cv2.VideoWriter(video_out, fourcc, 20.0, (692, 947))

    env = get_env(hp.env, hp.frames_per_state, hp.input_shape[1:])

    pprint.pprint([op.name for op in sess.graph.get_operations()])

    font = cv2.FONT_HERSHEY_SIMPLEX

    fig, axes = plt.subplots(ncols=1, nrows=2)
    canvas = fig.canvas
    ax1, ax2 = axes.ravel()

    display_lstm = hp.model == "a3c_lstm"

    title_image = np.zeros((20, (1 + 20*2+2 + 9*2+2)*4 + 180 * (2 if display_lstm else 1), 3), dtype='float32')

    background_color = (204/255., 255/255., 204/255.)

    def set_color(im, c):
        im[:, :, 2] = c[2]
        im[:, :, 1] = c[1]
        im[:, :, 0] = c[0]
    set_color(title_image, background_color)

    cv2.putText(title_image, "Conv1", (5, 15), font, .5, (0., 0., 0.), bottomLeftOrigin=False, thickness=2)
    cv2.putText(title_image, "Conv2", (5 + (20*2 + 2) * 4, 15), font, .5, (0., 0., 0.), bottomLeftOrigin=False,
                thickness=2)
    cv2.putText(title_image, "FC3", (5 + (20*2 + 2) * 4 + (9 * 2 + 2)*4, 15), font, .5, (0., 0., 0.),
                bottomLeftOrigin=False, thickness=2)
    if display_lstm:
        cv2.putText(title_image, "LSTM4", (5 + (20*2 + 2) * 4 + (9 * 2 + 2)*4 + 180, 15), font, .5, (0., 0., 0.),
                    bottomLeftOrigin=False, thickness=2)

    conv1_image = np.zeros((20 * 8 + 8, 1 + 20 * 2 + 2, 3), dtype='float32')
    conv2_image = np.zeros((9 * 16 + 16 + 8, 9 * 2 + 2, 3), dtype='float32')
    fc1_image = np.zeros((34, 9, 3), dtype='float32')
    lstm_image = np.zeros((34, 9, 3), dtype='float32')
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
            value, action, conv1_out, conv2_out, pi, fc1, lstm = agent.local_network\
                .get_value_and_action_and_visualize(state, sess)

            max_val = max(value, max_val - abs(max_val) * 0.001)
            min_val = min(value, min_val + abs(min_val) * 0.001)

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

            env_image = env.env._get_image()

            fc1_image[1:-1, 0:8, :] = cv2.cvtColor(np.reshape(fc1, (32, 8)), cv2.COLOR_GRAY2RGB)
            if display_lstm:
                lstm_image[1:-1, 0:8, :] = cv2.cvtColor(np.reshape(lstm, (32, 8)), cv2.COLOR_GRAY2RGB)
            #ax.text(0.0, 0.0, "Conv1", fontsize=45)
            #ax.axis('off')
            if episode_step > 10:
                ax1.cla()
                ax2.cla()
                bar = ax2.bar(range(env.num_actions()), pi)
                ax2.set_title("Policy", fontsize=24)#r"\pi(s,a)")
                ax2.set_ylabel(r"$\pi(s,a)$", fontsize=24)
                ax2.set_ylim([0., 1.])
                ax2.set_xticklabels(env.env.get_action_meanings(), rotation=45)
                xticks_pos = [0.65 * patch.get_width() + patch.get_xy()[0] for patch in bar]
                ax2.set_xticks(xticks_pos)
                ax1.plot(range(episode_step-9, episode_step+1), value_buffer[-10:], linewidth=2)
                ax1.set_title("Value estimate", fontsize=24)
                ax1.set_ylabel(r"$v(s)$", fontsize=24)
                ax1.set_ylim([min_val, max_val])
                ax1.set_xlabel(r"Step", fontsize=20)
                plt.tight_layout()
                canvas.draw()
                plot_image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
                plot_image = plot_image.reshape(canvas.get_width_height()[::-1] + (3,))

                factor = plot_image.shape[1] / env_image.shape[1]
                new_height = int(factor * env_image.shape[0])
                resized_env_image = cv2.resize(env_image, (plot_image.shape[1], new_height))
                plot_and_env = np.concatenate((resized_env_image, plot_image), axis=0) / 255.


                all_convs = cv2.resize(np.concatenate((conv1_image, conv2_image), axis=1), None, fx=4, fy=4)
                fc_image_resized = cv2.resize(fc1_image, (180, all_convs.shape[0]), interpolation=cv2.INTER_NEAREST)
                if display_lstm:
                    lstm_image_resized = cv2.resize(lstm_image, (180, all_convs.shape[0]), interpolation=cv2.INTER_NEAREST)

                all_convs = np.concatenate((all_convs, fc_image_resized) + (lstm_image_resized,)
                                           if display_lstm else tuple([]), axis=1)
                all_convs = np.concatenate((title_image, all_convs), axis=0)
                factor = all_convs.shape[0] / plot_and_env.shape[0]
                new_width = int(factor * plot_and_env.shape[1])
                resized_env_and_plot = cv2.resize(plot_and_env, (new_width, all_convs.shape[0]))


                #cv2.imshow('test', resized_env_and_plot)
                #cv2.waitKey(1)

                result = np.concatenate((all_convs.copy(), resized_env_and_plot.copy()), axis=1)

                out.write((result * 255.0).astype('u1'))


                #cv2.imshow('all_convs', result)
                #cv2.waitKey(1)

                #env.env.render()
                #time.sleep(1/60/4)

            iter += 1
            episode_step += 1


