import numpy as np
import tensorflow as tf

from deeprl.approximators.layers import spatialsoftmax

input = tf.placeholder(tf.float32, [1, 3, 3, 1])
layer_output, xc, yc = spatialsoftmax(input, trainable_temperature=False, safe_softmax=False)

arr = np.zeros((3, 3))
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(3):
    for j in range(3):
        new_arr = np.copy(arr)
        new_arr[i, j] = 1
        new_arr = np.reshape(new_arr, (1, 3, 3, 1))

        x, y = sess.run([xc, yc], feed_dict={input: new_arr})
        print(x.shape)
        print(i, j, np.reshape(x, (3, 3)), np.sum(x))
