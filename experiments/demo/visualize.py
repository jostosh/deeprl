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
import matplotlib.cm as cm

from sklearn.decomposition import PCA
import uuid


mpl.rc('text', usetex=True)
mpl.rc('text', color='white')
mpl.rc('font', **{'family': 'sans-serif'}) , #'serif': ['Computer Modern Roman']})
mpl.rc('xtick', labelsize=14, color='white')
mpl.rc('ytick', labelsize=14, color='white')
mpl.rc('figure', facecolor="#000000")
mpl.rc('axes', facecolor='#111111', labelcolor='white', labelsize=16, titlesize=20, prop_cycle=mpl.cycler(color=['ffff00']))
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
#import seaborn as sns

import colorlover as cl

class Tiler:

    def __init__(self, dims, im_shape, ax, title):
        self.dims = dims
        self.ms = np.ones(dims)
        self.im_shape = im_shape
        self.out = np.zeros((dims[0] * im_shape[0] + dims[0], 1 + im_shape[1] * dims[1] + dims[1], 3), dtype='float32')
        self.ax = ax
        self.title = title
    #conv1_image = np.zeros((20 * 8 + 8, 1 + 20 * 2 + 2, 3), dtype='float32'

    def generate(self, conv, iter, raw=False):

        h, w = self.im_shape
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                idx = i * self.dims[1] + j
                slice = conv[0, :, :, idx]
                if not raw:
                    self.ms[i, j] = (np.max(slice) + self.ms[i, j] * (iter - 1)) / iter
                    mslice = self.ms[i, j]
                    im = cv2.cvtColor(slice / (mslice if mslice != 0 else 1.), cv2.COLOR_GRAY2RGB)
                else:
                    slice -= slice.min()
                    slice /= slice.max()
                    im = cv2.cvtColor(slice, cv2.COLOR_GRAY2RGB)
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
        self.ax = ax
        self.title = title
        self.average = np.zeros(dims)

    def generate(self, array):
        array = np.reshape(array, self.dims)
        self.average = 0.1 * array + 0.9 * self.average
        self.out = array / (self.average + 1e-10)

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
        val_sal = np.sum(np.abs(saliency[0]), axis=0)
        #val_sal = np.exp(val_sal * 10)
        #val_sal /= val_sal.sum()
        val_sal -= val_sal.min()
        val_sal /= val_sal.max()

        val_sal = np.stack([val_sal, np.zeros_like(val_sal), np.zeros_like(val_sal)], axis=2)
        val_sal = cv2.resize(val_sal, self.dims).astype('float')
        env_value = np.copy(im).astype('float') / 255

        #print(val_sal)

        alpha = 0.5
        self.out = np.clip(env_value + val_sal, 0.0, 1.0) #cv2.addWeighted(val_sal, alpha, env_value, 1 - alpha, 0)

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

        im = np.copy(im)

        for x, y in zip(cartesian_x[0], cartesian_y[0]):
            im_x = int((x + 0.5) * self.dim[1])
            im_y = int((y + 0.5) * self.dim[0])

            #print(im_x, im_y)

            im = cv2.circle(im, (im_x, im_y), 3, (0, 255, 0), -1)

        self.out = im

    def render(self):
        self.ax.cla()
        self.ax.axis('off')
        self.ax.imshow(self.out)
        self.ax.set_title(self.title)

class Tsne:

    def __init__(self, prototypes, num_actions, meanings, ax, title):
        self.ax = ax
        self.title = title
        self.num_actions = num_actions

        self.projector = PCA(n_components=2, whiten=True)
        self.embedded = np.split(
            self.projector.fit_transform(prototypes),
            indices_or_sections=num_actions, axis=0
        )
        self.meanings = meanings

    def generate(self, head):
        self.head = self.projector.transform(head)[0]


    def render(self):
        def to_01(c):
            return np.asarray(c) / 255.

        self.ax.cla()
        for dat, label, color in zip(self.embedded, self.meanings, cl.to_numeric(cl.scales['11']['qual']['Paired'])):
            self.ax.scatter(dat[:, 0], dat[:, 1], lw=0, s=40, c=to_01(color))
            xtext, ytext = np.median(dat, axis=0)
            txt = self.ax.text(xtext, ytext, label, fontsize=12)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=1, foreground=to_01(color)),
                PathEffects.Normal()])

        self.ax.scatter(self.head[0], self.head[1], lw=0, s=80, c='w')

        self.ax.axis('off')
        self.ax.set_title(self.title)


def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def execute():
    unique_filename = uuid.uuid4()
    parser = ArgumentParser()
    parser.add_argument("-d", "--model_dir")
    parser.add_argument("-n", "--n_episodes", default=5)
    args = parser.parse_args()

    T = tf.Variable(83500472, name="T")
    #print([v.name for v in tf.global_variables()])

    agent, hp, sess, vars = load_agent(args.model_dir)

    pprint.pprint(hp.__dict__)

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

    pprint.pprint([op.name for op in sess.graph.get_operations() if 'Prototype' in op.name])

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

    iter = 1



    ww_tensors = [sess.graph.get_tensor_by_name(t) for t in [
        "GLOBAL/HiddenLayers/Relu:0",
        "GLOBAL/HiddenLayers/Relu_1:0",
        "GLOBAL/HiddenLayers/GLOBAL/FC3/Relu:0",
        #"GLOBAL/HiddenLayers/SpatialSoftmax/xOut:0",
        #"GLOBAL/HiddenLayers/SpatialSoftmax/yOut:0",
        #"GLOBAL/HiddenLayers/SpatialSoftmax/SoftmaxPerChannel:0",
    ]] + [agent.local_network.head, value_saliency, policy_saliency]

    prototypes_tensor = sess.graph.get_tensor_by_name("GLOBAL/Outputs/Policy/Prototypes:0")
    prototypes = sess.run(prototypes_tensor)

    dims = ww_tensors[0].get_shape().as_list()[1:3]
    tiler_conv1 = Tiler((16, 2), ww_tensors[0].get_shape().as_list()[1:3], ax_conv1, "Conv1")
    tiler_conv2 = Tiler((16, 2), ww_tensors[1].get_shape().as_list()[1:3], ax_conv2, "Conv2")
    flatter = Flatter((16, 16), ax_fc3, "FC3")
    val_sal = Saliency((160, 210), ax_val_overlay, "Value Saliency")
    pi_sal = Saliency((160, 210), ax_pi_overlay, "Policy Saliency")
    pt = Tsne(prototypes, agent.num_actions, env.env.get_action_meanings(), ax_ss, "Prototypes")


    #ss = SpatialSoftmax(80/84 * 25/27, (210, 160), ax_ss, "Spatial Softmax") #Tiler((8, 4), ww_tensors[1].get_shape().as_list()[1:3], ax_ss, "Spatial Softmax")


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

            ims = [
                tiler_conv1.generate(tensors_out[0], iter),
                tiler_conv2.generate(tensors_out[1], iter),
                flatter.generate(tensors_out[2]),
                #ss.generate(tensors_out[1], iter)
            ]

            env_image = env.env._get_image()
            val_sal.generate(env_image, tensors_out[-2])
            pi_sal.generate(env_image, tensors_out[-1])
            pt.generate(tensors_out[2 ])
            #ss.generate(env_image, tensors_out[4], tensors_out[5])

            if episode_step > 20:
                [ax.cla() for ax in [ax_conv1, ax_conv2, ax_fc3, ax_game, ax_input, ax_val_overlay, ax_pi_overlay,
                                     ax_pol, ax_val]]
                [ax.axis('off') for ax in [ax_conv1, ax_conv2, ax_fc3, ax_game, ax_input, ax_val_overlay, ax_pi_overlay]]
                tiler_conv1.render()
                tiler_conv2.render()
                flatter.render()
                val_sal.render()
                pi_sal.render()
                pt.render()

                ax_game.imshow(env_image)
                ax_game.set_title("Game env")

                ax_input.imshow(state[-1, :, :], cmap='gray')
                ax_input.set_title("Preprocessed frame")
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
                writer.write(plot_image.astype('u1'))

                #cv2.imshow('Plot', plot_image)
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

    #print_tensors_in_checkpoint_file(model_checkpoint_path, '')

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