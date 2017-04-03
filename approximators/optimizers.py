import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.training import training_ops
import numpy as np


class RMSPropShared(object):

    def __init__(self, session, learning_rate, decay=0.99, epsilon=1e-8, theta=None, momentum=0., feedback=False,
                 thl=0.1, thu=10., feedback_decay=0.99, global_clipping=False, global_clip_norm=1.0, d_clip_lo=0.1,
                 d_clip_hi=10, ms_bias_correction=False):
        self.learning_rate = learning_rate
        self.decay = decay

        self.decay_tensor = tf.convert_to_tensor(decay, name="DecayRate")
        self.momentum_tensor = tf.convert_to_tensor(momentum, name='Momentum')
        self.epsilon_tensor = tf.convert_to_tensor(epsilon, name="Epsilon")

        self.epsilon = epsilon
        if theta:
            self.build_update(theta)
        self.session = session

        self.ms = None
        self.global_theta = None
        self.mom = None

        self.thl = tf.convert_to_tensor(thl) #tf.Variable(thl, dtype=tf.float32, trainable=False)
        self.thu = tf.convert_to_tensor(thu) # tf.Variable(thu, dtype=tf.float32, trainable=False)
        self.d = tf.Variable(1., dtype=tf.float32, trainable=False)

        self.loss_prev = tf.Variable(0., dtype=tf.float32, trainable=False)
        self.feedback = feedback
        self.t = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.feedback_decay = feedback_decay
        self.updates = []
        self.global_clipping = global_clipping
        self.global_clip_norm = global_clip_norm
        self.d_clip_lo = d_clip_lo
        self.d_clip_hi = d_clip_hi
        self.ms_bias_correction = ms_bias_correction


    def _init_from_prototype(self, theta):
        with tf.name_scope("MovingAverageGradient"):
            self.ms = [tf.Variable(tf.zeros(shape=var.get_shape().as_list(), dtype=tf.float32))
                       for var in theta]
            self.mom = [tf.Variable(tf.zeros(shape=var.get_shape().as_list(), dtype=tf.float32)) for var in theta]

    def set_global_theta(self, theta):
        self.global_theta = theta
        self._init_from_prototype(theta)

    def build_update_from_vars(self, theta, loss):
        assert self.global_theta and self.ms and self.mom

        with tf.name_scope("GradientInput"):
            if self.global_clipping:
                grads, _ = tf.clip_by_global_norm(tf.gradients(loss, theta), self.global_clip_norm)
            else:
                grads = [tf.clip_by_norm(grad, 40.0) for grad in tf.gradients(loss, theta)]

        other_updates = [tf.assign_add(self.t, 1, use_locking=False, name="Increment_t")]
        d_t = None
        if self.feedback:
            not_first_iter = tf.greater(self.t, 1)
            loss_prev = tf.Variable(0., dtype=tf.float32, trainable=False)
            loss_hat_prev = tf.Variable(0., dtype=tf.float32, trainable=False)

            cond = tf.greater(loss, loss_prev)
            ch_fact_lbound = tf.cond(cond, lambda: 1 + self.thl, lambda: 1 / (1 + self.thu))
            ch_fact_ubound = tf.cond(cond, lambda: 1 + self.thu, lambda: 1 / (1 + self.thl))
            loss_ch_fact = loss / loss_prev
            loss_ch_fact = tf.maximum(loss_ch_fact, ch_fact_lbound)
            loss_ch_fact = tf.minimum(loss_ch_fact, ch_fact_ubound)
            loss_hat = tf.cond(not_first_iter, lambda: loss_hat_prev * loss_ch_fact, lambda: loss)

            d_den = tf.minimum(loss_hat, loss_hat_prev) + tf.constant(1e-8)  # tf.cond(tf.greater(loss_hat, loss_prev), )
            d_t = (self.feedback_decay * self.d) + (1. - self.feedback_decay) * tf.abs((loss_hat - loss_hat_prev) / d_den)
            d_t = tf.clip_by_value(
                tf.cond(not_first_iter, lambda: d_t, lambda: tf.constant(1.)),
                self.d_clip_lo,
                self.d_clip_hi
            )

            other_updates = [tf.assign(self.d, d_t, use_locking=False),
                             tf.assign(self.loss_prev, loss, use_locking=False),
                             tf.assign(loss_hat_prev, loss_hat)]

        return self.build_update_from_grads(grads, other_updates=other_updates, d_t=d_t)

    def build_update_from_grads(self, grads, other_updates, d_t):

        with tf.name_scope("MovingAverageGradientUpdate"):
            g_update = [tf.assign(ms, self.decay * ms + (1 - self.decay) * tf.square(grad), use_locking=False)
                        for ms, grad in zip(self.ms, grads)]
        with tf.name_scope("RMSPropMinimize"):
            lr_t = self.learning_rate if not self.ms_bias_correction \
                else self.learning_rate * (1 - tf.pow(self.decay, self.t+1))
            def get_lr(var):
                if 'Prototypes' in var.name:
                    return 0.05
                return lr_t

            minimize = tf.group(*([tf.assign_add(t, -get_lr(t) *
                                      tf.div(grad, tf.sqrt(ms * (d_t if self.feedback else 1.) + self.epsilon)),
                                      use_locking=False)
                                   for t, grad, ms in zip(self.global_theta, grads, g_update)] + other_updates),
                                name='minimize')
        return minimize


class AdamShared(object):

    def __init__(self, session, learning_rate, epsilon=1e-8, theta=None, feedback=False, global_clipping=False,
                 thl=0.1, thu=10., feedback_decay=0.99, beta1=0.9, beta2=0.999, d_clip_lo=0.1, d_clip_hi=10,
                 global_clip_norm=1.0):
        self.beta1 = tf.convert_to_tensor(beta1)
        self.beta2 = tf.convert_to_tensor(beta2)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        if theta:
            self.build_update(theta)
        self.session = session

        self.ms = []
        self.vs = []
        self.global_theta = None

        self.thl = tf.Variable(thl, dtype=tf.float32, trainable=False)
        self.thu = tf.Variable(thu, dtype=tf.float32, trainable=False)
        self.d = tf.Variable(1., dtype=tf.float32, trainable=False)

        self.loss_prev = tf.Variable(0., dtype=tf.float32, trainable=False)
        self.feedback = feedback
        self.t = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.feedback_decay = feedback_decay
        self.updates = []
        self.d_clip_lo = d_clip_lo
        self.d_clip_hi = d_clip_hi
        self.global_clipping = global_clipping
        self.global_clip_norm = global_clip_norm

    def _init_from_prototype(self, theta):
        with tf.name_scope("MovingAverageGradient"):
            self.ms = [tf.Variable(tf.zeros(var.get_shape().as_list()), dtype=tf.float32) for var in theta]
            self.vs = [tf.Variable(tf.zeros(var.get_shape().as_list()), dtype=tf.float32) for var in theta]

    def set_global_theta(self, theta):
        self.global_theta = theta
        self._init_from_prototype(theta)

    def _clip_grad(self, grad):
        shape = grad.get_shape().as_list()
        if len(shape) == 4:
            return tf.clip_by_norm(grad, 40.0/np.prod(shape[:-1]), axes=-1)
        if len(shape) == 2:
            return tf.clip_by_norm(grad, 40.0/shape[0], axes=-1)
        if len(shape) == 1:
            return tf.clip_by_norm(grad, 10.0)

    def build_update_from_vars(self, theta, loss):
        assert self.global_theta and self.ms and self.vs

        with tf.name_scope("GradientInput"):
            if self.global_clipping:
                grads, _ = tf.clip_by_global_norm(tf.gradients(loss, theta), self.global_clip_norm)
            else:
                grads = [tf.clip_by_norm(grad, 40.0) for grad in tf.gradients(loss, theta)]

        other_updates = [tf.assign_add(self.t, 1, use_locking=False, name="Increment_t")]
        d_t = None
        if self.feedback:
            not_first_iter = tf.greater(self.t, 1.0)
            loss_prev = tf.Variable(0., dtype=tf.float32, trainable=False)
            loss_hat_prev = tf.Variable(0., dtype=tf.float32, trainable=False)

            cond = tf.greater(loss, loss_prev)
            ch_fact_lbound = tf.cond(cond, lambda: 1 + self.thl, lambda: 1 / (1 + self.thu))
            ch_fact_ubound = tf.cond(cond, lambda: 1 + self.thu, lambda: 1 / (1 + self.thl))
            loss_ch_fact = loss / loss_prev
            loss_ch_fact = tf.maximum(loss_ch_fact, ch_fact_lbound)
            loss_ch_fact = tf.minimum(loss_ch_fact, ch_fact_ubound)
            loss_hat = tf.cond(not_first_iter, lambda: loss_hat_prev * loss_ch_fact, lambda: loss)

            d_den = tf.minimum(loss_hat, loss_hat_prev)  # tf.cond(tf.greater(loss_hat, loss_prev), )
            d_t = (self.feedback_decay * self.d) + (1. - self.feedback_decay) * tf.abs((loss_hat - loss_hat_prev) / d_den)
            d_t = tf.clip_by_value(
                tf.cond(not_first_iter, lambda: d_t, lambda: tf.constant(1.)),
                self.d_clip_lo,
                self.d_clip_hi
            )

            other_updates = [tf.assign(self.d, d_t, use_locking=False),
                             tf.assign(self.loss_prev, loss, use_locking=False),
                             tf.assign(loss_hat_prev, loss_hat)]

        return self.build_update_from_grads(grads, other_updates=other_updates, d_t=d_t)

    def build_update_from_grads(self, grads, other_updates, d_t):

        lr_t = self.learning_rate * (tf.sqrt(1. - tf.pow(self.beta2, self.t+1)) /
                                     (1. - tf.pow(self.beta1, self.t+1)))

        with tf.name_scope("GradientMomentUpdate"):
            m_t = [tf.assign(m, (self.beta1 * m) + (1. - self.beta1) * g)            for m, g in zip(self.ms, grads)]
            v_t = [tf.assign(v, (self.beta2 * v) + (1. - self.beta2) * tf.square(g)) for v, g in zip(self.vs, grads)]
        with tf.name_scope("RMSPropMinimize"):
            minimize = tf.group(
                *([
                    tf.assign_add(p, -lr_t * m / (tf.sqrt(v * (d_t if d_t is not None else 1) + self.epsilon)))
                    for p, m, v in zip(self.global_theta, m_t, v_t)
                ] + other_updates)
            )
        return minimize




class RMSPropMultiLoss(object):

    def __init__(self, session, learning_rate, decay=0.99, epsilon=1e-8, theta=None, momentum=0., feedback=False,
                 thl=0.5, thu=2., feedback_decay=0.99):
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

    def _init_from_prototype(self, theta):
        with tf.name_scope("MovingAverageGradient"):
            self.mean_square = [tf.Variable(tf.zeros(shape=var.get_shape().as_list(), dtype=tf.float32))
                                for var in theta]
            self.mom = [tf.Variable(tf.zeros(shape=var.get_shape().as_list(), dtype=tf.float32)) for var in theta]

    def set_global_theta(self, theta):
        self.global_theta = theta
        self._init_from_prototype(theta)

    def build_update_from_vars(self, theta, losses):
        assert self.global_theta and self.mean_square and self.mom

        with tf.name_scope("GradientInput"):
            all_grads = [
                [tf.clip_by_norm(grad, 40.0) for grad in tf.gradients(loss, theta)] for loss in losses
            ]

        #return tf.group(*[training_ops.apply_rms_prop(
        #    var=t, ms=ms, mom=mom, lr=self.learning_rate, rho=self.decay_tensor, momentum=self.momentum_tensor,
        #    epsilon=self.epsilon_tensor, grad=g, use_locking=False
        #) for t, ms, mom, g in zip(self.global_theta, self.g_moving_average, self.mom, grads)], name="RMSPropUpdate")

        return self.build_update_from_grads(all_grads)

    def build_update_from_grads(self, grads):

        main_grads = grads[0]
        main_norm = [tf.sqrt(tf.reduce_sum(tf.square(g))) for g in main_grads]

        accordance_coeff = []
        with tf.name_scope("GradientAccordance"):
            accordance_vars = [[tf.Variable(0., dtype=tf.float32) for _ in range(len(main_grads))]
                               for _ in range(len(grads) - 1)]

            for i in range(1, len(grads)):
                auxiliary_grads = grads[i]
                accordance_old = accordance_vars[i-1]
                accordance_new = [tf.reduce_sum(mg * ag) / (tf.sqrt(tf.reduce_sum(tf.square(ag))) * mn)
                                  for mg, ag, mn in zip(main_grads, auxiliary_grads, main_norm)]
                accordance_coeff.append([tf.assign(acc_var, acc_var * self.decay + (1 - self.decay) * acc)
                                         for acc_var, acc in zip(accordance_old, accordance_new)])
        combined_grads = []
        for grad_idx in range(len(main_grads)):
            combined_grads.append(tf.add_n([main_grads[grad_idx]] +
                                           [grads[i][grad_idx] ** ((accordance_coeff[i-1][grad_idx] - 1) * 10)
                                            for i in range(1, len(main_grads))]))

        with tf.name_scope("MovingAverageGradientUpdate"):
            g_update = [tf.assign(ms, self.decay * ms + (1 - self.decay) * tf.square(combined_grads), use_locking=False)
                        for ms, grad in zip(self.mean_square, combined_grads)]
        with tf.name_scope("RMSPropMinimize"):
            minimize = tf.group(*([tf.assign_add(t, -self.learning_rate *
                                      tf.div(grad, tf.sqrt(ms + self.epsilon)),
                                      use_locking=False)
                                   for t, grad, ms in zip(self.global_theta, grads, g_update)]),
                                name='minimize')
        return minimize
