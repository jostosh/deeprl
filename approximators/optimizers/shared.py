import tensorflow as tf

class SharedOptimizer(object):

    def __init__(self, session, learning_rate, theta, momentum=0., global_clipping=False, global_clip_norm=1.0,
                 d_clip_lo=0.1, d_clip_hi=10, ms_bias_correction=False, *args, **kwargs):
        self.learning_rate = learning_rate
        if theta:
            self.build_update(theta)
        self.session = session
        self.global_theta = None

        self.loss_prev = tf.Variable(0., dtype=tf.float32, trainable=False)
        self.t = tf.Variable(0.0, dtype=tf.float32, trainable=False)
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
            minimize = tf.group(*([tf.assign_add(t, -lr_t *
                                      tf.div(grad, tf.sqrt(ms * (d_t if self.feedback else 1.) + self.epsilon)),
                                      use_locking=False)
                                   for t, grad, ms in zip(self.global_theta, grads, g_update)] + other_updates),
                                name='minimize')
        return minimize
