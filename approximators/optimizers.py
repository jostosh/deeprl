import tensorflow as tf
from tensorflow.python.framework import constant_op


class RMSPropCustom(object):

    def __init__(self, session, learning_rate, decay=0.99, epsilon=1e-8, theta=None):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        if theta:
            self.build_update(theta)
        self.session = session

    def build_update(self, theta):

        with tf.name_scope("GradientInput"):
            self.gradients = [tf.placeholder(tf.float32, shape=var.get_shape().as_list()) for var in theta]

        with tf.name_scope("MovingAverageGradient"):
            self.g_moving_average = [tf.Variable(tf.zeros(shape=var.get_shape().as_list(), dtype=tf.float32)) for var in theta]
            self.g_update = [tf.assign(g, self.decay * g + (1 - self.decay) * tf.square(d_t))
                             for g, d_t in zip(self.g_moving_average, self.gradients)]
        with tf.name_scope("RMSPropMinimize"):
            self.minimize = [tf.assign_add(t, -self.learning_rate * tf.div(grad, tf.sqrt(g_mov_avg + self.epsilon))) for
                             t, grad, g_mov_avg in zip(theta, self.gradients, self.g_moving_average)]


