import tensorflow as tf

def euclidean_squared_neg(net, prototypes):
    diff = tf.expand_dims(net, 1) - tf.expand_dims(prototypes, 0)
    return -tf.reduce_sum(tf.square(diff), axis=2), []

def euclidean_neg(net, prototypes):
    diff = tf.expand_dims(net, 1) - tf.expand_dims(prototypes, 0)
    return -tf.sqrt(tf.reduce_sum(tf.square(diff), axis=2)), []

def correlation(net, prototypes):
    bias = tf.Variable(tf.zeros(prototypes.get_shape().as_list()[0]))
    correlation = tf.nn.xw_plus_b(net, tf.transpose(prototypes), bias)
    return correlation, [bias]

def manhattan_neg(net, prototypes):
    diff = tf.expand_dims(net, 1) - tf.expand_dims(prototypes, 0)
    return -tf.reduce_sum(tf.abs(diff), axis=2), []

def inv_euclidean_squared(net, prototypes):
    diff = tf.expand_dims(net, 1) - tf.expand_dims(prototypes, 0)
    return tf.div(1.0, 1.0 + tf.reduce_sum(tf.square(diff), axis=2)), []

def inv_euclidean(net, prototypes):
    diff = tf.expand_dims(net, 1) - tf.expand_dims(prototypes, 0)
    return tf.div(1.0, 1.0 + tf.sqrt(tf.reduce_sum(tf.square(diff), axis=2))), []

def glvq_score(distances, num_classes):
    scores = []
    for i in range(num_classes):
        jnoti = [j for j in range(num_classes) if j != i]

        distance_wrong = tf.reshape(tf.reduce_min(
            tf.transpose(tf.gather(tf.transpose(distances, (1, 0, 2)), jnoti), (1, 0, 2)),
            axis=[1, 2]),
            shape=(-1, 1)
        )

        distance_right = distances[:, i, :]

        scores.append(distance_right - distance_wrong / (distance_right + distance_wrong))

    return tf.stack(scores, axis=1)



similarity_functions = {
    'euc': euclidean_neg,
    'cor': correlation,
    'man': manhattan_neg,
    'euc_sq': euclidean_squared_neg,
    'inv_euc': inv_euclidean,
    'inv_euc_sq': inv_euclidean_squared
}