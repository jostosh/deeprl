import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from matplotlib import rc

rc('font',**{'family': 'serif', 'serif': ['Computer Modern Roman']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


xs = np.asarray([0, 0.5, 1])
ys = np.asarray([0, 0.2, -0.2])

dim = 50
size = 1

xticks = np.linspace(0, size, dim)
yticks = np.linspace(-size, size, dim)

xmesh, ymesh = np.meshgrid(xticks, yticks)
xmesh = xmesh.reshape((dim, dim, 1))
ymesh = ymesh.reshape((dim, dim, 1))

softmax = np.exp((xs * xmesh + ys * ymesh) * 10)
softmax /= np.sum(softmax, axis=2, keepdims=True)
softmax[softmax != np.max(softmax, axis=2, keepdims=True)] *= 0


plt.imshow(softmax)
plt.show()

lvq = np.exp(-((xs - xmesh) ** 2 + (ys - ymesh) ** 2))
lvq /= np.sum(lvq, axis=2, keepdims=True)
lvq[lvq != np.max(lvq, axis=2, keepdims=True)] *= 0

plt.imshow(lvq)
plt.show()