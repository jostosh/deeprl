import numpy as np
import tensorflow as tf
from scipy.stats import rankdata


def get_rank(arr, axis=-1):
    if axis == -1:
        axis = len(arr.shape) - 1
    ranks = np.apply_along_axis(rankdata, axis, arr).astype('float32')
    return ranks


def val_to_rank(val, axis=-1, name='RankData'):
    return tf.py_func(get_rank, [val, axis], tf.float32, stateful=False, name=name)