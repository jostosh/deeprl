import tensorflow as tf


def hard_sigmoid(x):
    """Segment-wise linear approximation of sigmoid.
    Faster than sigmoid.
    Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
    In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    x = (0.2 * x) + 0.5
    zero = tf.convert_to_tensor(0., tf.float32)
    one = tf.convert_to_tensor(1., tf.float32)
    x = tf.clip_by_value(x, zero, one)
    return x


def get_activation(act):
    if callable(act):
        return act
    elif isinstance(act, str):
        return {
            'linear': tf.identity,
            'relu': tf.nn.relu,
            'elu': tf.nn.elu,
            'softmax': tf.nn.softmax
        }[act]
    else:
        raise TypeError("The activation {} is neither a function nor a string, but rather a {}".format(act, type(act)))