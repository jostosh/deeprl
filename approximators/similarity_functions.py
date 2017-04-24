import tensorflow as tf


def eucclidean_similarity_squared(net, prototypes):
    diff = tf.expand_dims(net, 1) - tf.expand_dims(prototypes, 0)
    return -tf.reduce_sum(tf.square(diff), axis=2), []


def eucclidean_similarity(net, prototypes):
    diff = tf.expand_dims(net, 1) - tf.expand_dims(prototypes, 0)
    return -tf.sqrt(tf.reduce_sum(tf.square(diff), axis=2)), []


def correlation_similarity(net, prototypes):
    bias = tf.Variable(tf.zeros(prototypes.get_shape().as_list()[0]))
    correlation = tf.nn.xw_plus_b(net, tf.transpose(prototypes), bias)
    return correlation, [bias]


def manhattan_similarity(net, prototypes):
    diff = tf.expand_dims(net, 1) - tf.expand_dims(prototypes, 0)
    return -tf.reduce_sum(tf.abs(diff), axis=2), []


similarity_functions = {
    'euc': eucclidean_similarity,
    'cor': correlation_similarity,
    'man': manhattan_similarity,
    'euc_sq': eucclidean_similarity_squared
}