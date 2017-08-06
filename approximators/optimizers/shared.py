import tensorflow as tf


class RMSPropOptimizer(object):

    def __init__(self, session, learning_rate, decay, epsilon, theta=None, global_clipping=False, global_clip_norm=40):
        self.learning_rate = learning_rate
        if theta:
            self.build_update(theta)
        self.decay = decay
        self.session = session
        self.global_theta = None
        self.epsilon = epsilon

        self.global_clipping = global_clipping
        self.global_clip_norm = global_clip_norm

    def _init_from_prototype(self, theta):
        with tf.name_scope("MovingAverageGradient"):
            self.ms = [tf.Variable(tf.zeros(shape=var.get_shape().as_list(), dtype=tf.float32))
                       for var in theta]
            self.mom = [tf.Variable(tf.zeros(shape=var.get_shape().as_list(), dtype=tf.float32)) for var in theta]

    def set_global_theta(self, theta):
        self.global_theta = theta
        self._init_from_prototype(theta)

    def build_update_from_vars(self, theta, loss):
        with tf.name_scope("GradientInput"):
            if self.global_clipping:
                grads, _ = tf.clip_by_global_norm(tf.gradients(loss, theta), self.global_clip_norm)
            else:
                grads = [tf.clip_by_average_norm(grad, 40.0) for grad in tf.gradients(loss, theta)]

        return self.build_update_from_grads(grads)

    def build_update_from_grads(self, grads):
        with tf.name_scope("MovingAverageGradientUpdate"):
            g_update = [tf.assign(ms, self.decay * ms + (1 - self.decay) * tf.square(grad))
                        for ms, grad in zip(self.ms, grads)]
        with tf.name_scope("RMSPropMinimize"):
            lr_t = self.learning_rate
            minimize = tf.group(
                *([tf.assign_add(t, -lr_t * tf.div(grad, tf.sqrt(ms + self.epsilon)))
                   for t, grad, ms in zip(self.global_theta, grads, g_update)]),
                name='minimize'
            )
        return minimize
