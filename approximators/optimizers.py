import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.training import training_ops


class RMSPropCustom(object):

    def __init__(self, session, learning_rate, decay=0.99, epsilon=1e-8, theta=None, momentum=0., feedback=False,
                 thl=0.1, thu=10., feedback_decay=0.99):
        self.learning_rate = learning_rate
        self.decay = decay

        self.decay_tensor = tf.convert_to_tensor(decay, name="DecayRate")
        self.momentum_tensor = tf.convert_to_tensor(momentum, name='Momentum')
        self.epsilon_tensor = tf.convert_to_tensor(epsilon, name="Epsilon")

        self.epsilon = epsilon
        if theta:
            self.build_update(theta)
        self.session = session

        self.mean_square = None
        self.global_theta = None
        self.mom = None

        self.thl = tf.Variable(thl, dtype=tf.float32, trainable=False)
        self.thu = tf.Variable(thu, dtype=tf.float32, trainable=False)
        self.d = tf.Variable(1., dtype=tf.float32, trainable=False)

        self.loss_prev = tf.Variable(0., dtype=tf.float32, trainable=False)
        self.feedback = feedback
        self.t = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.feedback_decay = feedback_decay
        self.updates = []


    '''
    def build_update(self, theta):

        with tf.name_scope("GradientInput"):
            self.gradients = [tf.placeholder(tf.float32, shape=var.get_shape().as_list()) for var in theta]

        with tf.name_scope("MovingAverageGradient"):
            self.mean_square = [tf.Variable(tf.zeros(shape=var.get_shape().as_list(), dtype=tf.float32))
                                for var in theta]
            self.g_update = [tf.assign(g, self.decay * g + (1 - self.decay) * tf.square(d_t), use_locking=False)
                             for g, d_t in zip(self.mean_square, self.gradients)]
        with tf.name_scope("RMSPropMinimize"):
            self.minimize = tf.group(*[tf.assign_add(t, -self.learning_rate * tf.div(grad, tf.sqrt(g_mov_avg + self.epsilon)),
                                                     use_locking=False) for
                                       t, grad, g_mov_avg in zip(theta, self.gradients, self.g_update)])
    '''

    def _init_from_prototype(self, theta):
        with tf.name_scope("MovingAverageGradient"):
            self.mean_square = [tf.Variable(tf.zeros(shape=var.get_shape().as_list(), dtype=tf.float32))
                                for var in theta]
            self.mom = [tf.Variable(tf.zeros(shape=var.get_shape().as_list(), dtype=tf.float32)) for var in theta]

    def set_global_theta(self, theta):
        self.global_theta = theta
        self._init_from_prototype(theta)

    def build_update_from_vars(self, theta, loss):
        assert self.global_theta and self.mean_square and self.mom

        with tf.name_scope("GradientInput"):
            grads = [tf.clip_by_norm(grad, 40.0) for grad in tf.gradients(loss, theta)]

        #return tf.group(*[training_ops.apply_rms_prop(
        #    var=t, ms=ms, mom=mom, lr=self.learning_rate, rho=self.decay_tensor, momentum=self.momentum_tensor,
        #    epsilon=self.epsilon_tensor, grad=g, use_locking=False
        #) for t, ms, mom, g in zip(self.global_theta, self.g_moving_average, self.mom, grads)], name="RMSPropUpdate")
        other_updates = []
        d_t = None
        if self.feedback:
            not_first_iter = tf.greater(self.t, 1)

            ch_fact_lbound = tf.cond(tf.greater(loss, self.loss_prev), lambda: 1 + self.thl, lambda: 1/(1 + self.thu))
            ch_fact_ubound = tf.cond(tf.less(loss, self.loss_prev), lambda: 1 + self.thu, lambda: 1/(1 + self.thl))

            loss_ch_fact = loss / self.loss_prev
            loss_ch_fact = tf.maximum(ch_fact_lbound, loss_ch_fact)#tf.cond(tf.less(loss_ch_fact, ch_fact_lbound), lambda: ch_fact_lbound, lambda: loss_ch_fact)
            loss_ch_fact = tf.minimum(ch_fact_ubound, loss_ch_fact) #tf.cond(tf.greater(loss_ch_fact, ch_fact_ubound), lambda: ch_fact_ubound, lambda: loss_ch_fact)
            loss_hat = tf.cond(not_first_iter, lambda: self.loss_prev * loss_ch_fact, lambda: loss)

            d_den = tf.minimum(loss_hat, loss_prev) # tf.cond(tf.greater(loss_hat, self.loss_prev), lambda: self.loss_prev, lambda: loss_hat)
            d_t = self.feedback_decay * self.d + (1.-self.feedback_decay) * tf.abs((loss_hat - self.loss_prev) / d_den)
            d_t = tf.cond(not_first_iter, lambda: d_t, lambda: tf.constant(1.))

            other_updates = [tf.assign(self.d, d_t, use_locking=False),
                             tf.assign_add(self.t, 1, use_locking=False, name="Increment_t"),
                             tf.assign(self.loss_prev, loss_hat, use_locking=False)]

        return self.build_update_from_grads(grads, other_updates=other_updates, d_t=d_t)

    def build_update_from_grads(self, grads, other_updates, d_t):

        with tf.name_scope("MovingAverageGradientUpdate"):
            g_update = [tf.assign(ms, self.decay * ms + (1 - self.decay) * tf.square(grad), use_locking=False)
                        for ms, grad in zip(self.mean_square, grads)]
        with tf.name_scope("RMSPropMinimize"):
            minimize = tf.group(*([tf.assign_add(t, -self.learning_rate *
                                      tf.div(grad, tf.sqrt(ms + self.epsilon) * (d_t if self.feedback else 1.)),
                                      use_locking=False) for t, grad, ms in zip(self.global_theta, grads, g_update)] + \
                       other_updates), name='minimize')
        return minimize
