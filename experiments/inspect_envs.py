from deeprl.common.environments import get_env
import matplotlib.pyplot as plt
import time
import tensorflow as tf


session = tf.Session()

env = get_env('Breakout-v0', output_shape=[84, 84], session=session)
plt.ion()

state = env.reset()
plt.imshow(state[-1], cmap='gray')
plt.pause(0.01)
while True:
    action = env.env.action_space.sample()

    state, r, terminal, i = env.step(action)
    if terminal:
        env.reset_random()

    plt.imshow(state[-1], cmap='gray')
    env.env.render()
    #time.sleep(1/60. * 4)
    plt.pause(0.01)


