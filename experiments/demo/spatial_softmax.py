import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from matplotlib import rc
from scipy.spatial import Voronoi, voronoi_plot_2d
from deeprl.common.environments import get_env
from scipy.signal import convolve2d

env = get_env("BeamRider-v0", output_shape=(84, 84))
env.reset()

for i in range(20):
    env.step(np.random.randint(env.num_actions()))



#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')


rc('font',**{'family': 'serif', 'serif': ['Computer Modern Roman'], 'size': 20})

rc('xtick', labelsize=30)
rc('ytick', labelsize=30)

## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rc('figure', autolayout=False)

x = np.linspace(-1, 1, 84)
y = np.linspace(-1, 1, 84)

xx, yy = np.meshgrid(x, y)

scharr = np.array([[ 1, 2,  1],
                   [0, 0, 0],
                   [ -1, -2,  -1]]) # Gx + j*Gy
frame = env.state[-1]


fig = plt.figure(figsize=(7.5, 7.5))
ax = fig.add_subplot(111)
ax.imshow(frame, cmap='gray')
ax.set_axis_off()
plt.tight_layout()
plt.savefig('/home/jos/Dropbox/RUG/6e Jaar/mproj/thesis/im/ss_input.pdf')
plt.show()


grad = convolve2d(frame, scharr, boundary='symm', mode='same')

softmax = np.exp(np.absolute(grad))
softmax /= np.sum(softmax, keepdims=True)

fig = plt.figure(figsize=(7.5, 7.5))
ax = fig.add_subplot(111)
ax.imshow(np.absolute(grad), cmap='gray')
ax.set_axis_off()
plt.tight_layout()
plt.savefig('/home/jos/Dropbox/RUG/6e Jaar/mproj/thesis/im/ss_linear.pdf')
plt.show()

fig = plt.figure(figsize=(7.5, 7.5))
ax = fig.add_subplot(111)
ax.imshow(softmax, cmap='gray')
ax.set_axis_off()
plt.tight_layout()
plt.savefig('/home/jos/Dropbox/RUG/6e Jaar/mproj/thesis/im/ss_softmax.pdf')
plt.show()

fig = plt.figure(figsize=(7.5, 7.5))
ax = fig.add_subplot(111)
ax.imshow(softmax, cmap='gray')
start, end = ax.get_xlim()
ticks = np.linspace(start, end, 5)
labels = np.linspace(-1, 1, 5)
ax.set_xticks(ticks)
ax.set_xticklabels(labels)
ax.set_yticks(ticks)
ax.set_yticklabels(labels[::-1])
ax.grid(color='gray', linestyle='-', linewidth=1)
ax.plot([1.05 * end/2, 1.05 * end/2], [start, end], c='r', linestyle='-', linewidth=2)
ax.plot([start, end], [1.72 * end/2, 1.72 * end/2], c='r', linestyle='-', linewidth=2)
plt.tight_layout()
plt.savefig('/home/jos/Dropbox/RUG/6e Jaar/mproj/thesis/im/ss_cartesian.pdf')
plt.show()