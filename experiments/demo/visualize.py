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
import signal
import sys
from copy import deepcopy
import matplotlib.gridspec as gridspec
from deeprl.common.hyper_parameters import config1
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from deeprl.approximators.optimizers import RMSPropShared
import matplotlib.animation as manimation

import uuid


mpl.rc('text', usetex=True)
mpl.rc('text', color='white')
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
mpl.rc('xtick', labelsize=14, color='white')
mpl.rc('ytick', labelsize=14, color='white')
mpl.rc('figure', facecolor="#000000")
mpl.rc('axes', facecolor='#111111', labelcolor='white', labelsize=16, titlesize=20, prop_cycle=mpl.cycler(color=['ffff00']))
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


class Tiler:

    def __init__(self, dims, im_shape, ax, title):
        self.dims = dims
        self.ms = np.ones(dims)
        self.im_shape = im_shape
        self.out = np.zeros((dims[0] * im_shape[0] + dims[0], 1 + im_shape[1] * dims[1] + 2, 3), dtype='float32')
        self.ax = ax
        self.title = title
    #conv1_image = np.zeros((20 * 8 + 8, 1 + 20 * 2 + 2, 3), dtype='float32'

    def generate(self, conv, iter):

        h, w = self.im_shape
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                idx = i * self.dims[1] + j
                slice = conv[0, :, :, idx]
                self.ms[i, j] = (np.max(slice) + self.ms[i, j] * (iter - 1)) / iter
                mslice = self.ms[i, j]
                im = cv2.cvtColor(slice / (mslice if mslice != 0 else 1.), cv2.COLOR_GRAY2RGB)
                self.out[i*h+i:  (i+1)*h+i, \
                     1 + j*w+j:1+(j+1)*w+j, :] = im
        return self.out

    def render(self):
        self.ax.cla()
        self.ax.axis('off')
        self.ax.imshow(self.out)
        self.ax.set_title(self.title)


class Flatter:
    def __init__(self, dims, ax, title):
        self.dims = dims
        self.out = np.zeros((dims[0] + 2, dims[1] + 1, 3), dtype='float32')
        self.ax = ax
        self.title = title

    def generate(self, array):
        self.out = array
        return self.out

    def render(self):
        self.ax.cla()
        self.ax.axis('off')
        self.ax.imshow(np.reshape(self.out, self.dims), cmap='gray', interpolation='nearest')
        self.ax.set_title(self.title)


class Saliency:
    def __init__(self, dims, ax, title):
        self.dims = dims
        self.title = title
        self.ax = ax

    def generate(self, im, saliency, saliency2=None):
        val_sal = np.abs(saliency[0][-1])
        val_sal = np.exp(val_sal)
        val_sal /= val_sal.sum()
        #val_sal -= val_sal.min()
        #val_sal /= val_sal.max()

        val_sal = np.stack([val_sal, np.zeros_like(val_sal), np.zeros_like(val_sal)], axis=2)
        val_sal = cv2.resize(val_sal, self.dims).astype('float')
        env_value = np.copy(im).astype('float') / 255

        #print(val_sal)

        alpha = 0.5
        self.out = cv2.addWeighted(val_sal, alpha, env_value, 1 - alpha, 0)

    def render(self):
        self.ax.cla()
        self.ax.axis('off')
        self.ax.imshow(self.out)
        self.ax.set_title(self.title)


class SpatialSoftmax:

    def __init__(self, factor, dim, ax, title):
        self.factor = factor
        self.dim = dim
        self.ax = ax
        self.title = title

    def generate(self, im, cartesian_x, cartesian_y):
        cartesian_x *= self.factor
        cartesian_y *= self.factor

        for x, y in zip(cartesian_x[0], cartesian_y[0]):
            im_x = int((x + 0.5) * self.dim[1])
            im_y = int((y + 0.5) * self.dim[0])

            im = cv2.circle(im, (im_x, im_y), 20, (0, 255, 0), -1)

        self.out = im

    def render(self):
        self.ax.cla()
        self.ax.axis('off')
        self.ax.imshow(self.out)
        self.ax.set_title(self.title)

def execute():
    unique_filename = uuid.uuid4()
    parser = ArgumentParser()
    parser.add_argument("-d", "--model_dir")
    parser.add_argument("-n", "--n_episodes", default=5)
    args = parser.parse_args()

    agent, hp, sess, vars = load_agent(args.model_dir)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_out = os.path.join(args.model_dir, 'video{}.mp4'.format(unique_filename))

    out = cv2.VideoWriter(video_out, fourcc, 15., (1920, 1080), isColor=True)

    from skvideo.io import VideoWriter
    writer = VideoWriter(video_out, fourcc='XVID', frameSize=(1920, 1080), fps=15.0, isColor=True)
    writer.open()

    value_saliency = tf.gradients(agent.local_network.value, [agent.local_network.inputs])[0]
    policy_saliency = tf.gradients(tf.reduce_max(agent.local_network.pi, axis=1), [agent.local_network.inputs])[0]


    def signal_handler(signal, frame):
        print("Writing video to {}".format(video_out))
        out.release()
        print("Closing...")
        cv2.destroyAllWindows()
        writer.release()

        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    env = get_env(hp.env, hp.frames_per_state, hp.input_shape[1:])

    pprint.pprint([op.name for op in sess.graph.get_operations()])

    #fig, axes = plt.subplots(ncols=1, nrows=2)
    #canvas = fig.canvas
    #ax1, ax2 = axes.ravel()
    my_dpi = 96
    fig2 = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi)
    gs = gridspec.GridSpec(3, 6)
    ax_conv1 = plt.subplot(gs[:, 0])
    ax_conv2 = plt.subplot(gs[:, 1])
    ax_fc3 = plt.subplot(gs[0, 2])
    #ax_lstm = plt.subplot(gs[:, 3])
    ax_ss = plt.subplot(gs[0, 3])
    ax_input = plt.subplot(gs[0, 4])
    ax_val = plt.subplot(gs[1, 2])
    ax_pol = plt.subplot(gs[2, 2])

    ax_game = plt.subplot(gs[1:, 3:5])
    ax_val_overlay = plt.subplot(gs[1, 5])
    ax_pi_overlay = plt.subplot(gs[2, 5])

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

    slice_ms2 = np.ones((16, 2))

    iter = 1

    ww_tensors = [sess.graph.get_tensor_by_name(t) for t in [
        "GLOBAL/HiddenLayers/Conv1/Relu:0",
        "GLOBAL/HiddenLayers/SpatialSoftmax/SoftmaxPerChannel:0",
        "GLOBAL/HiddenLayers/Relu:0",
        "GLOBAL/HiddenLayers/FC3/Relu:0",
        "GLOBAL/HiddenLayers/SpatialSoftmax/xOut:0",
        "GLOBAL/HiddenLayers/SpatialSoftmax/yOut:0"
    ]] + [value_saliency, policy_saliency]

    dims = ww_tensors[0].get_shape().as_list()[1:3]
    tiler_conv1 = Tiler((16, 2), ww_tensors[0].get_shape().as_list()[1:3], ax_conv1, "Conv1")
    tiler_conv2 = Tiler((16, 2), ww_tensors[2].get_shape().as_list()[1:3], ax_conv2, "Conv2")
    flatter = Flatter((16, 16), ax_fc3, "FC3")
    val_sal = Saliency((160, 210), ax_val_overlay, "Value Saliency")
    pi_sal = Saliency((160, 210), ax_pi_overlay, "Policy Saliency")
    ss = SpatialSoftmax(80/84 * 25/27, (210, 160), ax_ss, "Spatial Softmax")


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
            if episode_step % 1 == 0:
                print("Now at step {}".format(episode_step))

            rnn_state = None
            if hp.frame_prediction and agent.local_network.recurrent:
                rnn_state = deepcopy(agent.local_network.lstm_state_numeric)

            value, action, pi, tensors_out = agent.local_network\
                .get_value_and_action_and_visualize(state, sess, ww_tensors)

            if hp.frame_prediction:
                predicted_frame = agent.local_network.get_frame_prediction(state, action, sess, rnn_state)

            max_val = max(value + abs(value) * 0.1, max_val - abs(max_val) * 0.001)
            min_val = min(value - abs(value) * 0.1, min_val + abs(min_val) * 0.001)

            state, r, terminal, _ = env.step(action)

            value_buffer.append(value)
            slice_ms1 = np.ones((8, 2))

            ims = [
                tiler_conv1.generate(tensors_out[0], iter),
                tiler_conv2.generate(tensors_out[2], iter),
                flatter.generate(tensors_out[3]),
            ]

            '''
            

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
            '''

            env_image = env.env.ale.getScreenRGB2() #[:, :, ::-1]
            #print("OAISNDOIASNF {}".format(env.env._get_image().shape))
            #fc1_image[1:-1, 0:8, :] = cv2.cvtColor(np.reshape(fc1, (32, 8)), cv2.COLOR_GRAY2RGB)
            val_sal.generate(env_image, tensors_out[-2])
            pi_sal.generate(env_image, tensors_out[-1])
            ss.generate(env_image, tensors_out[4], tensors_out[5])

            if display_lstm:
                lstm_image[1:-1, 0:8, :] = cv2.cvtColor(np.reshape((lstm + 1.) / 2., (32, 8)), cv2.COLOR_GRAY2RGB)
            if episode_step > 20:
                [ax.cla() for ax in [ax_conv1, ax_conv2, ax_fc3, ax_game, ax_input, ax_val_overlay, ax_pi_overlay,
                                     ax_pol, ax_val]]
                [ax.axis('off') for ax in [ax_conv1, ax_conv2, ax_fc3, ax_game, ax_input, ax_val_overlay, ax_pi_overlay]]
                tiler_conv1.render()
                tiler_conv2.render()
                flatter.render()
                val_sal.render()
                pi_sal.render()
                ss.render()

                '''
                
                [ax.cla() for ax in [ax_conv1, ax_conv2, ax_fc3, ax_lstm, ax_game, ax_input, ax_prediction,
                                     ax_pol, ax_val]]
                [ax.axis('off') for ax in [ax_conv1, ax_conv2, ax_fc3, ax_lstm, ax_game, ax_input, ax_prediction]]

                ax_conv1.imshow(conv1_image)
                ax_conv1.set_title('Conv1')
                ax_conv2.imshow(conv2_image)
                ax_conv2.set_title('Conv2')
                ax_fc3.imshow(np.reshape(fc1, (32, 8)), cmap='gray', interpolation='nearest')
                ax_fc3.set_title('FC3')
                '''
                #ax_lstm.imshow(np.reshape((lstm + 1.) / 2., (32, 8)), cmap='gray', interpolation='nearest')
                #ax_lstm.set_title('LSTM4')

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
                #plt.show()

                #writer.grab_frame()

                canvas2 = fig2.canvas
                canvas2.draw()
                plot_image = np.fromstring(canvas2.tostring_rgb(), dtype='uint8')
                plot_image = plot_image.reshape(canvas2.get_width_height()[::-1] + (3,))
                #out.write(plot_image.astype('u1'))
                writer.write(plot_image.astype('u1'))
                #cv2.imshow('Game', env_image)
                #cv2.waitKey()


            iter += 1
            episode_step += 1


def load_agent(model_dir):
    sess = tf.Session()
    with open(os.path.join(model_dir, 'hyper_parameters.pkl'), 'rb') as f:
        hp_raw = pickle.load(f)
        hp = HyperParameters(hp_raw) if not isinstance(hp_raw, HyperParameters) else hp_raw

    for param in config1:
        if param not in hp.__dict__:
            hp.__dict__[param] = config1[param]

    agent = A3CAgent(env_name=hp.env, global_network=None, agent_name='GLOBAL', session=sess, optimizer=None, hp=hp)
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    model_checkpoint_path = os.path.join(model_dir, os.path.basename(checkpoint.model_checkpoint_path))

    print_tensors_in_checkpoint_file(model_checkpoint_path, '')

    reader = tf.train.NewCheckpointReader(model_checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    vars = {v.name: v for v in agent.local_network.theta}
    #saver = tf.train.Saver(var_list=agent.local_network.theta)
    sess.run(tf.global_variables_initializer())

    def optimistic_restore(session, save_file, restore_vars):
        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                            if var.name.split(':')[0] in saved_shapes])
        saver = tf.train.Saver(restore_vars)
        saver.restore(session, save_file)

    #saver.restore(sess, model_checkpoint_path)
    optimistic_restore(sess, model_checkpoint_path, vars)
    return agent, hp, sess, vars


if __name__ == "__main__":
    execute()