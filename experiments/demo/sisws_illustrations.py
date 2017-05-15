import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from matplotlib import rc

rc('font',**{'family': 'serif', 'serif': ['Computer Modern Roman']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

np.random.seed(123)

fig = plt.figure(figsize=(15, 7.5))
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-1, 1, 8)
y = np.linspace(-1, 1, 8)

X, Y = np.meshgrid(x, y)
Z = np.zeros(X.shape)

xx = np.linspace(-1, 1, 8)
yy = np.linspace(-1, 1, 8)

XX, YY = np.meshgrid(xx, yy)

# Grab some test data.
#X, Y, Z = axes3d.get_test_data(0.025)

# Plot a basic wireframe.
zs0 = []
zs1 = []
for i in range(4):
    ax.plot_surface(X, Y, Z-i, color='r', linewidth=0.5, rstride=1, cstride=1, edgecolor='w')
    cx, cy = np.random.rand() * 2 - 1, np.random.rand() * 2 - 1
    zs0.append(np.exp(-((cx - XX) ** 2 + (cy - YY) ** 2)))
    ax.plot_surface(XX+2.5, YY+2.5, zs0[-1] - i, rstride=1, cstride=1,
                    linewidth=0.0, color='r')

for j, dz in enumerate(range(6, 10)):
    ax.plot_surface(X, Y, Z-dz, color='b', linewidth=0.5, rstride=1, cstride=1, edgecolor='w')
    cx, cy = np.random.rand() * 2 - 1, np.random.rand() * 2 - 1
    zs1.append(np.exp(-((cx - XX) ** 2 + (cy - YY) ** 2)))
    ax.plot_surface(XX+2.5, YY+2.5, zs1[-1] - dz, rstride=1, cstride=1,
                    linewidth=0.0, color='b')

zs = []
for z0, z1 in zip(zs0, zs1):
    combined = np.asarray([[[1.0, 0.0, 0.0]]]) * np.reshape(z0, (8, 8, 1)) + \
               np.asarray([[[0.0, 0.0, 1.0]]]) * np.reshape(z1, (8, 8, 1)) / \
               np.reshape(z0 + z1, (8, 8, 1))
    #combined -= combined.min(axis=(0, 1), keepdims=True)
    #combined /= combined.max(axis=(0, 1), keepdims=True)
    #combined += 0.9

    zs.append(combined)

for i, z in zip(np.linspace(3, 6, 4), zs):
    ax.plot_surface(XX+5, YY+5, Z - i, facecolors=z, edgecolor='w', rstride=1, cstride=1, linewidth=0.5)

ax.set_axis_off()
#sax.quiver(1.1, 1.1, -2, 0, 0, -1, color='k', length=0.5)
ax.text(1.2, 1.2, -1.5, r"$\odot$", size=18, multialignment='center')
ax.text(1.2, 1.2, -7.5, r"$\odot$", size=18, multialignment='center')

ax.text(0.0, 0.0, 1.5, r"Convolution output", size=18, ha='center')
ax.text(2.5, 2.5, 1.5, r"Spatial coefficients", size=18, ha='center')
ax.text(5, 5, -1, r"Summed and normalized", size=18, ha='center')

plt.show()