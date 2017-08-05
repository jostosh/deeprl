import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata


tau = tf.placeholder(tf.float32, shape=[])
sess = tf.Session()

n =5


def get_rank(arr, axis=-1):
    if axis == -1:
        axis = len(arr.shape) - 1
    ranks = np.apply_along_axis(rankdata, axis, arr).astype('float32')
    return ranks

#arr = np.asarray(np.random.rand(n)).astype('float32')
arr = tf.placeholder(tf.float32, [n])

_, indices = tf.nn.top_k(arr, n, sorted=True)

score = tf.nn.softmax(-tau * tf.cast(indices, tf.float32))

vals = [sess.run(score, feed_dict={arr: np.random.rand(n), tau: i}).max() for i in np.arange(0, 10, .01)]

#plt.plot(np.arange(0, 10, .01), vals)
#plt.show()

arr2 = tf.placeholder(tf.float32, [2, n])

ranks = tf.py_func(get_rank, [arr2, 1], tf.float32)

x = np.random.rand(2, n)
print(get_rank(x))
print(sess.run([ranks], feed_dict={arr2: x}), x)
