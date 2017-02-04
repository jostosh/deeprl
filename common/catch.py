import numpy as np
from scipy.misc import imresize


class CatchEnv():

    def __init__(self, frames_per_state):
        self.image = np.zeros((24, 24))
        self.state = []
        self.fps = frames_per_state
        self.output_shape = (84, 84)

    def reset_random(self):
        self.image.fill(0)
        self.pos = np.random.randint(2, 24-2)
        self.vx = np.random.randint(5) - 2
        self.vy = 1
        self.ballx, self.bally = np.random.randint(24), 4
        self.image[self.bally, self.ballx] = 1
        #print(self.image[-1, self.pos-2:self.pos+3].shape)
        #print(self.pos)
        self.image[-5, self.pos-2:self.pos+3] = np.ones(5)

        return self.step(2)[0]


    def step(self, action):
        def left():
            if self.pos > 2:
                self.image[-5, self.pos+2] = 0
                self.pos -= 1
                self.image[-5, self.pos-2] = 1
        def right():
            if self.pos < 21:
                self.image[-5, self.pos-2] = 0
                self.pos += 1
                self.image[-5, self.pos+2] = 1
        def noop():
            pass
        {0: left, 1: right, 2: noop}[action]()

        self.image[self.bally, self.ballx] = 0
        self.ballx += self.vx
        self.bally += self.vy
        if self.ballx > 23:
            self.ballx -= 2 * (self.ballx - 23)
            self.vx *= -1
        elif self.ballx < 0:
            self.ballx += 2 * (0 - self.ballx)
            self.vx *= -1
        self.image[self.bally, self.ballx] = 1


        terminal = self.bally == 23 - 4
        reward = int(self.pos - 2 <= self.ballx <= self.pos + 2) if terminal else 0

        [self.state.append(imresize(self.image, (84, 84))) for _ in range(self.fps - len(self.state) + 1)]
        self.state = self.state[-self.fps:]

        return np.copy(self.state), reward, terminal, None

    def num_actions(self):
        return 3

    def reset(self):
        return self.reset_random()

    def state_shape(self):
        return (self.fps,) + self.output_shape


