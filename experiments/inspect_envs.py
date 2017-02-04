from deeprl.common.environments import get_env
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import numpy as np
import cv2

session = tf.Session()

env = get_env('Catch', output_shape=[84, 84])
#plt.ion()

state = env.reset_random()
#plt.imshow(state[-1], cmap='gray')
#plt.pause(0.01)
while True:
    action = 1#np.random.randint(3)#env.env.action_space.sample()

    state, r, terminal, i = env.step(action)
    if terminal:
        print(r)
        print(type(state), state.shape)
        env.reset_random()

    cv2.imshow('Catch', state[-1])
    cv2.waitKey(1)

    #plt.imshow(state[-1], cmap='gray')
    #env.env.render()
    time.sleep(1/60. * 4)
    #plt.pause(0.01)


