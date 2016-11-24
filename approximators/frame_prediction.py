import tensorflow as tf
import tflearn

from deeprl.common.environments import get_env
from deeprl.common.logger import logger
import numpy as np
import pprint
import os

input_index = 0
action_index = 1
target_index = 2

class FramePrediction(object):

    def __init__(self, input_shape, num_actions, input_tensor, reuse=False, scope=None):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.input = input_tensor
        self.reuse = reuse

        if not scope:
            with tf.variable_scope("Network") as scope:
                self.scope = scope
                self.build_network()
        else:
            with tf.variable_scope(scope) as scope:
                tf.get_variable_scope().reuse_variables()
                self.scope = scope
                self.build_network()


    def encoding_network(self):
        #self.input = tf.placeholder(tf.float32, self.input_shape)
        net = tf.transpose(self.input, [0, 2, 3, 1])
        conv1 = tflearn.conv_2d(net, 32, 8, 4, activation='relu', padding='valid', weight_decay=0.,
                                reuse=self.reuse, scope='Conv1', restore=False)
        conv2 = tflearn.conv_2d(conv1, 64, 4, 2, activation='relu', padding='valid', weight_decay=0.,
                                reuse=self.reuse, scope='Conv2', restore=False)
        net = tflearn.flatten(conv2)
        net = tflearn.fully_connected(net, 256, weight_decay=0., reuse=self.reuse, scope='FC1', restore=False)

        return net, conv1, conv2

    def decoding_network(self, incoming, conv1, conv2):
        net = tflearn.reshape(incoming, [-1] + conv2.get_shape().as_list()[1:])
        logger.info('Decoding input shape: {}'.format(net.get_shape()))
        logger.info("Shape conv1 {}".format(conv1.get_shape().as_list()[1:]))
        logger.info("Shape conv2 {}".format(conv2.get_shape().as_list()[1:]))
        net = tflearn.conv_2d_transpose(net, 32, 4, strides=2, activation='relu', output_shape=conv1.get_shape().as_list()[1:],
                                        weight_decay=0., reuse=self.reuse, scope='DeConv1', restore=False, padding='valid')
        logger.info("input shape {}".format(self.input_shape))
        net = tflearn.conv_2d_transpose(net, self.input_shape[0], 8, strides=4, activation='linear',
                                        output_shape=[84, 84, 4], padding='valid',
                                        weight_decay=0., reuse=self.reuse, scope='DeConv2', restore=False)
        return net

    def build_network(self):
        actor_critic_head, conv1, conv2 = self.encoding_network()

        self.action_input = tf.placeholder(tf.float32, [None])
        action_one_hot = tflearn.one_hot_encoding(self.action_input, self.num_actions)
        action_embedding = tflearn.fully_connected(action_one_hot, 256, weight_decay=0.0, bias=False,
                                                   reuse=self.reuse, scope='ActionEmbedding', restore=False)
        encoding = tflearn.fully_connected(actor_critic_head, 256, weight_decay=0.0, bias=False,
                                           reuse=self.reuse, scope='EncodingEmbedding', restore=False)

        transformation = tflearn.fully_connected(tf.mul(action_embedding, encoding),
                                                 np.prod(conv2.get_shape().as_list()[1:]), weight_decay=0.0,
                                                 activation='relu', reuse=self.reuse, scope='Transformation', restore=False)
        self.output = tf.transpose(self.decoding_network(transformation, conv1, conv2), [0, 3, 1, 2]) + \
                      tf.stop_gradient(self.input)

        self.target = tf.placeholder(tf.float32, [None] + self.input_shape)
        self.loss = tf.reduce_mean((self.target - self.output) ** 2) #tf.nn.l2_loss(self.target - self.output)


def generate_data(env, size):
    D = []
    E = []
    N = 0

    state = env.reset()

    while N < size:
        if N % 10000 == 0:
            logger.info("Generated {} samples".format(N))
        #logger.info("State shape: {}".format(np.asarray(state).shape))
        action = env.env.action_space.sample()
        next_state, reward, terminal, info = env.step(action)
        E.append([state, action, next_state])
        N += 1
        state = next_state

        if terminal:
            D += [E]
            E = []
            state = env.reset()
    if E:
        D.append(E)

    return D

def chain_network(net, n, shape, num_actions):
    all_nets = [net]
    for i in range(n - 1):
        all_nets.append(FramePrediction(shape, num_actions, all_nets[i].output, reuse=True, scope=net.scope))
    return all_nets


def sample_from_dataset(D, steps, batch_size):
    batch = []

    input_batch = []
    action_batch = []
    target_batch = []


    for _ in range(batch_size):
        i = np.random.randint(len(D))
        t = np.random.randint(len(D[i]))
        while t + steps >= len(D[i]):
            i = np.random.randint(len(D))
            t = np.random.randint(len(D[i]))

        input_batch.append(D[i][t + 0][input_index])
        action_batch.append([D[i][t + k][action_index] for k in range(steps)])
        target_batch.append([D[i][t + k][target_index] for k in range(steps)])

        batch.append((
            D[i][t + 0][input_index],
            [D[i][t + k][action_index] for k in range(steps)],
            [D[i][t + k][target_index] for k in range(steps)]
        ))
    return input_batch, action_batch, target_batch


def transpose_list(l):
    return list(map(list, zip(*l)))


def train(sess):
    env = get_env('Breakout-v0', output_shape=[84, 84], frames_per_state=4)

    input_shape = [None, 4, 84, 84]
    input_tensor = tf.placeholder(tf.float32, input_shape)
    frame_predictor = FramePrediction(list(env.state_shape()), env.num_actions(), input_tensor)

    phases = [
        {"num_iter": 1500000, "n_steps": 1, "lr": 0.001, 'batch_size': 32},
        {"num_iter": 1000000, "n_steps": 3, "lr": 0.0001, 'batch_size': 8},
        {"num_iter": 1000000, "n_steps": 5, "lr": 0.0001, 'batch_size': 8},
    ]

    chained_nets = chain_network(frame_predictor, max([p['n_steps'] for p in phases]), input_shape[1:], env.num_actions())

    D_train = generate_data(env, 1000000)
    #D_test  = generate_data(env, 100)

    writer = tf.train.SummaryWriter(logdir='{}/tensorflowlogs/frame_prediction'.format(os.path.expanduser('~')))
    optimizers = [tf.train.RMSPropOptimizer(phase['lr'], decay=0.95, momentum=0.9) for phase in phases]
    all_variables = set(tf.all_variables())

    tf.merge_all_summaries()

    combined_losses = [tf.div(tf.add_n([net.loss for net in chained_nets[:phase['n_steps']]]), phase['n_steps'])
                       for phase in phases]
    train_steps = [o.minimize(combined_loss) for o, combined_loss in zip(optimizers, combined_losses)]
    sess.run(tf.initialize_all_variables())

    for i, phase, train_step in zip(range(len(phases)), phases, train_steps):
        iter = 0

        iter_placeholder = tf.placeholder(tf.int32, iter)
        combined_loss = combined_losses[i]
        loss_summary = tf.scalar_summary('CombinedLoss{}'.format(i), combined_loss)
        output = chained_nets[phase['n_steps'] - 1].output

        #train_step = optimizer.minimize(combined_loss)

        while iter < phase["num_iter"]:
            inputs, actions, targets = sample_from_dataset(D_train, phase['n_steps'], phase['batch_size'])

            actions = np.asarray(actions).transpose((1, 0))
            targets = np.asarray(targets).transpose((1, 0, 2, 3, 4))
            inputs = np.asarray(inputs)

            fdict = {net.target: target_num for net, target_num in zip(chained_nets, targets)}
            fdict.update({chained_nets[0].input: inputs})
            fdict.update({net.action_input: action for net, action in zip(chained_nets, actions)})

            loss_num, loss_sum, _ = sess.run(
                [combined_loss, loss_summary, train_step],
                feed_dict=fdict
            )
            writer.add_summary(loss_sum)
            logger.info("Phase {}, Iter {}, loss {}".format(i, iter, loss_num))

            iter += 1


if __name__ == "__main__":
    with tf.Session() as sess:
        train(sess)


