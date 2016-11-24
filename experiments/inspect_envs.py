from deeprl.common.environments import get_env
import matplotlib.pyplot as plt
import time

env = get_env('Breakout-v0', output_shape=[84, 84])
plt.ion()

state = env.reset()
plt.imshow(state[-1], cmap='gray')
plt.pause(0.1)
while True:
    action = env.env.action_space.sample()

    state = env.step(action)[0]

    plt.imshow(state[-1], cmap='gray')
    env.env.render()
    plt.pause(0.1)
