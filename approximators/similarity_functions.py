import tensorflow as tf
from deeprl.common.tf_py_functions import val_to_rank

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
    return tf.div(1.0, 0.001 + tf.reduce_sum(tf.square(diff), axis=2)), []


def inv_euclidean(net, prototypes):
    diff = tf.expand_dims(net, 1) - tf.expand_dims(prototypes, 0)
    return tf.div(1.0, 1.0 + tf.sqrt(tf.reduce_sum(tf.square(diff), axis=2))), []


def pearson(net, prototypes):
    net = net - tf.reduce_mean(net, axis=1, keep_dims=True)
    prototypes = prototypes - tf.reduce_mean(prototypes, axis=1, keep_dims=True)

    normalized_feature = net / tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(net), axis=1)), 1)
    normalized_prototypes = prototypes / tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(prototypes), axis=1)), 1)
    return - (1 - tf.matmul(normalized_feature, tf.transpose(normalized_prototypes))), []


def cosine(net, prototypes):
    normalized_feature = net / tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(net), axis=1)), 1)
    normalized_prototypes = prototypes / tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(prototypes), axis=1)), 1)
    return - (1 - tf.matmul(normalized_feature, tf.transpose(normalized_prototypes))), []


def glvq_score(distances, num_classes, neural_gas=False, tau0=None, tauN=None, N=None):
    scores = []
    _, na, num_p = distances.get_shape().as_list()
    for i in range(num_classes):
        jnoti = [j for j in range(num_classes) if j != i]

        if neural_gas:
            T = [v for v in tf.global_variables() if v.name == "T:0"][0]

            tau = tf.cast(T, tf.float32) * (tauN - tau0) / N + tau0

            wrong_prototypes = tf.transpose(tf.gather(tf.transpose(distances, (1, 0, 2)), jnoti), (1, 0, 2))

            ranks = val_to_rank(tf.reshape(wrong_prototypes, (-1, (na - 1) * num_p)), name='WrongRanks')
            distance_wrong = tf.reshape(tf.reduce_sum(
                tf.reshape(tf.nn.softmax(-tau*ranks), (-1, na-1, num_p)) *
                wrong_prototypes,
                axis=[1, 2]),
                shape=(-1, 1)
            )
        else:
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
    'inv_euc_sq': inv_euclidean_squared,
    'pearson': pearson,
    'cosine': cosine
}