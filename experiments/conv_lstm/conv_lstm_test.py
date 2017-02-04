import os.path
import time

import numpy as np
import tensorflow as tf
import cv2

from deeprl.approximators.layers import conv_layer, convolutional_lstm, conv_transpose

import deeprl.experiments.conv_lstm.bouncing_balls as b

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './checkpoints/train_store_conv_lstm',
                           """dir to store trained net""")
tf.app.flags.DEFINE_integer('seq_length', 10,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('seq_start', 5,
                            """ start of seq generation""")
tf.app.flags.DEFINE_integer('max_step', 200000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', .8,
                          """for dropout""")
tf.app.flags.DEFINE_float('lr', .001,
                          """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                          """weight init for fully connected layers""")

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')



def generate_bouncing_ball_sample(batch_size, seq_length, shape, num_balls):
    dat = np.zeros((seq_length, batch_size, shape, shape, 3))
    for i in range(batch_size):
        dat[:, i, :, :, :] = b.bounce_vec(32, num_balls, seq_length)
    return dat


def train():
    """Train ring_net for a number of steps."""
    with tf.Graph().as_default():
        # make inputs
        x = tf.placeholder(tf.float32, [FLAGS.seq_length, FLAGS.batch_size, 32, 32, 3], name='Input')
        target = tf.placeholder(tf.float32, [FLAGS.seq_length, FLAGS.batch_size, 32, 32, 3], name='Target')

        # possible dropout inside
        #keep_prob = tf.placeholder("float")
        #x_dropout = tf.nn.dropout(x, keep_prob)

        net = conv_layer(tf.reshape(x, [FLAGS.seq_length * FLAGS.batch_size, 32, 32 ,3]), 8, 3, 2, tf.nn.elu,
                         'encode_1', 'same')
        net = conv_layer(net, 8, 3, 1, tf.nn.elu, 'encode_2', 'same')
        net = conv_layer(net, 8, 3, 2, tf.nn.elu, 'encode_3', 'same')
        net = conv_layer(net, 4, 1, 1, tf.nn.elu, 'encode_4', 'same')
        print("HAAAAI", net.get_shape().as_list())
        net, _, initial_state = convolutional_lstm(tf.reshape(net, [FLAGS.seq_length, FLAGS.batch_size, 8, 8, 4]),
                                                   outer_filter_size=3,
                                                   num_features=4,
                                                   stride=1,
                                                   #n_steps=np.asarray([FLAGS.seq_length], dtype='int32'),
                                                   inner_filter_size=3,
                                                   padding='SAME')
        print("HAAAAI", net.get_shape().as_list())
        net = conv_transpose(net, 8, 1, 1)
        net = conv_transpose(net, 8, 3, 2)
        net = conv_transpose(net, 8, 3, 1)
        out = conv_transpose(net, 3, 3, 2)
        print(out.get_shape().as_list())
        out = tf.reshape(out, [FLAGS.seq_length, FLAGS.batch_size, 32, 32, 3])

        # calc total loss (compare x_t to x_t+1)
        loss = tf.nn.l2_loss(out - target)
        tf.scalar_summary('loss', loss)

        # training
        train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

        # List of all Variables
        variables = tf.all_variables()

        # Build a saver
        saver = tf.train.Saver(tf.all_variables())

        # Summary op
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session()

        # init if this is the very time training
        print("init network from scratch")
        sess.run(init)

        # Summary op
        graph_def = sess.graph.as_graph_def(add_shapes=True)
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=graph_def)

        state_shape = initial_state.get_shape().as_list()
        #print(state_shape)
        start_state = np.zeros([FLAGS.batch_size] + state_shape[1:])
        print("START STATE SHAPE: ", start_state.shape)

        for step in range(FLAGS.max_step):
            dat = generate_bouncing_ball_sample(FLAGS.batch_size, FLAGS.seq_length + 1, 32, FLAGS.num_balls)
            t = time.time()
            _, loss_r = sess.run([train_op, loss], feed_dict={x: dat[:-1], target: dat[1:], initial_state: start_state})

            assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

            if step % 1000 == 0 and step != 0:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                print("saved to " + FLAGS.train_dir)

                # make video
                print("now generating video!")
                video = cv2.VideoWriter()
                success = video.open("generated_conv_lstm_video.mov", fourcc, 4, (180, 180), True)
                dat_gif = dat
                ims = sess.run(out, feed_dict={x: dat_gif[:-1],
                                               initial_state: start_state}).reshape((FLAGS.seq_length, FLAGS.batch_size, 32, 32, 3))
                ims = ims[:, 0, :, :, :]
                for i in range(FLAGS.seq_length):
                    x_1_r = np.uint8(np.maximum(ims[i, :, :, :], 0) * 255)
                    new_im = cv2.resize(x_1_r, (180, 180))
                    video.write(new_im)
                video.release()


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()


