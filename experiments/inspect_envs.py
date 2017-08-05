from deeprl.common.environments import get_env
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import numpy as np
import cv2

import os
from skvideo.io import VideoWriter
if os.path.exists('Catch.mp4'):
    os.remove('Catch.mp4')

video_out = 'Catch.mp4'

writer = VideoWriter(video_out, fourcc='XVID', frameSize=(84, 84), fps=15.0, isColor=True)
writer.open()
session = tf.Session()

env = get_env('Catch', output_shape=[84, 84], noiselevel=0.01)
#plt.ion()

state = env.reset_random()
#plt.imshow(state[-1], cmap='gray')
#plt.pause(0.01)

for _ in range(500):
    action = np.random.randint(env.num_actions()) #env.env.action_space.sample()

    state, r, terminal, i = env.step(action)
    if terminal:
        print(r)
        print(type(state), state.shape)
        env.reset_random()

    out = cv2.cvtColor(state[-1].astype(np.uint8), cv2.COLOR_GRAY2RGB)
    writer.write(out)
    cv2.imshow('Catch', out)
    cv2.waitKey(1)

    #plt.imshow(state[-1], cmap='gray')
    #env.env.render()
    time.sleep(1/60. * 4)
    #plt.pause(0.01)


writer.release()
