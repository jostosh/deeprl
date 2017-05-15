import tensorflow as tf
from deeprl.approximators.layers import conv_layer, conv_transpose, fc_layer
import tflearn
import argparse
import numpy as np
from deeprl.common.environments import get_env
from tqdm import trange
import os
import subprocess
import pickle as pkl

class LogDir(object):

    def __init__(self, args):
        self._logdir_base = self._get_logdir(args)

    def _get_logdir(self, args):
        hyperparameters = args.__dict__

        path = os.path.join(os.path.expanduser('~'), 'tensorflowlogs', 'v0.9.7', 'frame_prediction')
        path = os.path.join(path, *['{}={}'.format(param, val) for param, val in hyperparameters.items()])
        os.makedirs(path, exist_ok=True)
        current_dirs = sorted([o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))])
        # We're assuming at this point that we do not exceed 1M runs per version
        if not current_dirs:
            # If there are no dirs yet, start at 0
            rundir = 'run000000'
        else:
            # Otherwise make a new one by incrementing the count
            lastdir = current_dirs[-1]
            lastrun = int(lastdir[3:])
            rundir = "run%06d" % (lastrun + 1,)
        logdir = os.path.join(path, rundir)

        os.makedirs(logdir, exist_ok=True)

        with open(os.path.join(logdir, 'hyper_parameters.pkl'), 'wb') as f:
            hp = hyperparameters
            os.chdir(os.path.expanduser("~") + "/mproj/deeprl")
            hp.update({'git_description': subprocess.check_output(["git", "describe", "--always"]).decode('utf8').strip()})
            pkl.dump(hyperparameters, f, pkl.HIGHEST_PROTOCOL)
        return logdir

    def crossval_dir(self, idx):
        return os.path.join(self._logdir_base, 'fold{}'.format(idx))


class FramePredictor(object):

    def __init__(self, hp, num_actions):
        self.hp = hp
        self.num_actions = num_actions
        self.actions = None
        self.inputs = None
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.loss = None
        self.frame_target = None
        self._build_network()

    def _build_network(self, input_shape=[4, 84, 84]):
        with tf.name_scope("ForwardInputs") as scope:
            self.inputs = tf.placeholder(tf.float32, [None] + input_shape)
            self.actions = tf.placeholder(tf.float32, [None])
            net = tf.transpose(self.inputs, [0, 2, 3, 1])

        net, _  = self._encoder(net)

        with tf.name_scope("EncodingHead"):
            # Embed those actions
            action_one_hot = tflearn.one_hot_encoding(self.actions, self.num_actions)
            action_embedding = tflearn.fully_connected(action_one_hot, 256, weight_decay=0.0, bias=False,
                                                       name='ActionEmbedding', activation='linear')
            # Embed the hidden layer head
            encoding = tflearn.fully_connected(net, 256, weight_decay=0.0, bias=False,
                                               name='EncodingEmbedding', activation='linear')

            # Now we can compute the 'transformation layer' which we will be put into the decoding stream
            transformation = tflearn.fully_connected(tf.mul(action_embedding, encoding),
                                                     np.prod(self.conv3.get_shape().as_list()[1:]), weight_decay=0.0,
                                                     activation='linear', name='Transformation')

        decoded_prediction = self._decoder(transformation)

        with tf.name_scope("FramePrediction"):
            self.predicted_frame = tf.transpose(decoded_prediction, [0, 3, 1, 2]) \
                                   + (self.inputs[:, -1:, :, :])
            self.frame_target = tf.placeholder(tf.float32, [None, 1] + input_shape[1:])

        with tf.name_scope("Loss"):
            t = self.hp.trim
            frame_prediction_loss = tf.nn.l2_loss(
                self.frame_target[:, :, t:-t, t:-t] - self.predicted_frame[:, :, t:-t, t:-t],
                name='FramePredictionLoss') / ((84 - t) ** 2)

            baseline = tf.nn.l2_loss(
                self.frame_target[:, :, t:-t, t:-t] - (self.inputs[:, -1:, t:-t, t:-t]),
                name='BaselineLoss') / ((84 - t) ** 2)
            tf.summary.scalar('FramePredictionLoss', frame_prediction_loss)

        self.loss = frame_prediction_loss
        self.baseline = baseline

        tf.summary.scalar('Loss', frame_prediction_loss)

    def _encoder(self, net):

        with tf.name_scope('HiddenLayers') as scope:
            # Add first convolutional layer
            net = conv_layer(net, 32, 8, 2, activation='linear', name='Conv1', init=self.hp.weights_init)
            self.conv1 = net

            net = tf.nn.relu(net)

            # Add second convolutional layer
            net = conv_layer(net, 64, 4, 2, activation='linear', name='Conv2', init=self.hp.weights_init)
            self.conv2 = net
            net = tf.nn.relu(net)

            net = conv_layer(net, 64, 4, 2, activation='linear', name='Conv2', init=self.hp.weights_init)
            self.conv3 = net
            net = tf.nn.relu(net)

            net = tflearn.flatten(net)
            net = fc_layer(net, 256, activation=tf.nn.relu, name='FC3', init=self.hp.weights_init)
            self.embedding_layer = net

        return net, scope

    def _decoder(self, incoming):
        with tf.name_scope("DecodingNetwork"):
            # First we reshape the embedding into a volume with the shape of conv2
            net = tflearn.reshape(incoming, [-1] + self.conv3.get_shape().as_list()[1:], 'TransformationReshaped')
            if self.hp.residual_mode == 'add':
                net += self.conv3
            elif self.hp.residual_mode == 'concat':
                net = tf.concat(3, [net, self.conv3])
            net = tf.nn.relu(net, 'TransformationAct')

            # Then we perform a conv_2d_transpose (this is sometimes referred to as a DeConvolution layer)
            net = tflearn.conv_2d_transpose(net, 64, 4, strides=2, activation='linear',
                                            output_shape=self.conv2.get_shape().as_list()[1:],
                                            weight_decay=0., padding='valid', name='DeConv3')
            if self.hp.residual_mode == 'add':
                net += self.conv2
            if self.hp.residual_mode == 'concat':
                net = tf.concat(3, [net, self.conv2])
            net = tf.nn.relu(net, name='DeConv2Act')

            # Then we do the latter again
            net = tflearn.conv_2d_transpose(net, 32, 4, strides=2, activation='linear',
                                            output_shape=self.conv1.get_shape().as_list()[1:], padding='valid', weight_decay=0., name='DeConv2')
            net = tf.nn.relu(net, name='DeConv1Act')


            if self.hp.residual_mode == 'add':
                net += self.conv1
            if self.hp.residual_mode == 'concat':
                net = tf.concat(3, [net, self.conv1])
            net = tf.nn.relu(net, name='DeConv2Act')

            # Then we do the latter again
            net = tflearn.conv_2d_transpose(net, 32, 8, strides=2, activation='linear',
                                            output_shape=[84, 84, 32], padding='valid', weight_decay=0., name='DeConv1')
            net = tf.nn.relu(net, name='DeConv1Act')
            net = tf.concat(3, [net, tf.transpose(self.inputs, [0, 2, 3, 1])])
            net = conv_layer(net, n_filters=1, filter_size=5, stride=1, activation=tf.identity, name='FinalConv',
                             padding='same')

        return net


def train(args):

    env = get_env(args.env, output_shape=(84, 84), remove_artifacts=args.ra)

    action_buffer, state_buffer, target_buffer = gather_experience(args.steps, env)
    action_buffer_test, state_buffer_test, target_buffer_test = gather_experience(args.test_steps, env)

    mean_pixel_val = state_buffer.mean() if args.mpv else 0.0

    state_buffer -= mean_pixel_val
    target_buffer -= mean_pixel_val
    state_buffer_test -= mean_pixel_val
    target_buffer_test -= mean_pixel_val

    predictor = FramePredictor(args, env.num_actions())
    train_step = tf.train.RMSPropOptimizer(args.lr, epsilon=0.01, momentum=0.9, decay=0.95).minimize(predictor.loss)

    sess = tf.Session()
    summaries = tf.summary.merge_all()

    logdir = LogDir(args)

    def run_step(s, mode):
        start = s * args.batch_size
        end = min((s + 1) * args.batch_size, state_buffer.shape[0])
        if mode == 'train':
            sess.run(
                train_step,
                feed_dict={
                    predictor.inputs: state_buffer[start:end],
                    predictor.actions: action_buffer[start:end],
                    predictor.frame_target: target_buffer[start:end]
                }
            )
            return None
        elif mode == 'test':
            ret = sess.run(
                predictor.loss,
                feed_dict={
                    predictor.inputs: state_buffer_test[start:end],
                    predictor.actions: action_buffer_test[start:end],
                    predictor.frame_target: target_buffer_test[start:end]
                }
            )
            return ret
        else:
            ret = sess.run(
                predictor.baseline,
                feed_dict={
                    predictor.inputs: state_buffer_test[start:end],
                    predictor.actions: action_buffer_test[start:end],
                    predictor.frame_target: target_buffer_test[start:end]
                }
            )
            return ret

    errors = []
    steps = np.ceil(state_buffer_test.shape[0] / args.batch_size)
    for s in trange(int(steps), desc='Testing'):
        errors.append(run_step(s, 'baseline'))

    print("Baseline: {}".format(np.mean(errors)))

    for mode in ['add', 'concat', 'none']:
        sess.run(tf.global_variables_initializer())

        for epoch in range(args.train_epochs):
            permutation = np.random.permutation(action_buffer.shape[0])
            action_buffer = action_buffer[permutation]
            state_buffer = state_buffer[permutation]
            target_buffer = target_buffer[permutation]


            steps = np.ceil(state_buffer_test.shape[0] / args.batch_size)
            errors = []
            for s in trange(int(steps), desc='Testing'):
                errors.append(run_step(s, 'test'))
            print("Epoch {}, MSE {}".format(epoch, np.mean(errors)))

            steps = np.ceil(state_buffer.shape[0] / args.batch_size)
            for s in trange(int(steps), desc='Training'):
                run_step(s, 'train')




def gather_experience(steps, env):
    state_buffer = []
    action_buffer = []
    target_buffer = []
    state = env.reset()
    for _ in trange(steps, desc='Gathering experience'):
        action = np.random.randint(env.num_actions())
        new_state, _, terminal_state, _ = env.step(action)

        state_buffer.append(state)
        action_buffer.append(action)
        target_buffer.append(new_state[-1:, :, :])
        if terminal_state:
            env.reset()
    state_buffer = np.asarray(state_buffer)
    action_buffer = np.asarray(action_buffer)
    target_buffer = np.asarray(target_buffer)
    return action_buffer, state_buffer, target_buffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual_mode", choices=['add', 'concat', 'none'], default='none')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--trim', type=int, default=2)
    parser.add_argument('--env', default='Pong-v0')
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--test_steps", type=int, default=10000)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--weights_init", default='torch', choices=['default', 'torch'])
    parser.add_argument("--ra", action='store_true', dest='ra')
    parser.add_argument("--mpv", action='store_true', dest='mpv')
    parser.set_defaults(ra=False, mpv=False)
    args = parser.parse_args()
    train(args)
