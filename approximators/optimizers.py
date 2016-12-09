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

        self.g_moving_average = None
        self.global_theta = None

    def build_update(self, theta):

        with tf.name_scope("GradientInput"):
            self.gradients = [tf.placeholder(tf.float32, shape=var.get_shape().as_list()) for var in theta]

        with tf.name_scope("MovingAverageGradient"):
            self.g_moving_average = [tf.Variable(tf.zeros(shape=var.get_shape().as_list(), dtype=tf.float32)) for var in theta]
            self.g_update = [tf.assign(g, self.decay * g + (1 - self.decay) * tf.square(d_t), use_locking=False)
                             for g, d_t in zip(self.g_moving_average, self.gradients)]
        with tf.name_scope("RMSPropMinimize"):
            self.minimize = tf.group(*[tf.assign_add(t, -self.learning_rate * tf.div(grad, tf.sqrt(g_mov_avg + self.epsilon)),
                                                     use_locking=False) for
                                       t, grad, g_mov_avg in zip(theta, self.gradients, self.g_update)])

    def _init_from_prototype(self, theta):
        with tf.name_scope("MovingAverageGradient"):
            self.g_moving_average = [tf.Variable(tf.zeros(shape=var.get_shape().as_list(), dtype=tf.float32)) for var in theta]

    def set_global_theta(self, theta):
        self.global_theta = theta
        self._init_from_prototype(theta)


    def build_update_from_vars(self, theta, loss):
        assert self.global_theta and self.g_moving_average

        with tf.name_scope("GradientInput"):
            grads = tf.gradients(loss, theta) #[tf.placeholder(tf.float32, shape=var.get_shape().as_list()) for var in theta]

        with tf.name_scope("MovingAverageGradientUpdate"):
            g_update = [tf.assign(g, self.decay * g + (1 - self.decay) * tf.square(d_t), use_locking=False)
                        for g, d_t in zip(self.g_moving_average, grads)]
        with tf.name_scope("RMSPropMinimize"):
            minimize = tf.group(*[tf.assign_add(t, -self.learning_rate * tf.div(grad, tf.sqrt(g_mov_avg + self.epsilon)),
                                                use_locking=False) for
                                  t, grad, g_mov_avg in zip(self.global_theta, grads, g_update)], name='minimize')
        return minimize
