import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from matplotlib import rc
from scipy.spatial import Voronoi, voronoi_plot_2d



rc('font',**{'family': 'serif', 'serif': ['Computer Modern Roman'], 'size': 20})

rc('xtick', labelsize=14)
rc('ytick', labelsize=14)

## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rc('figure', autolayout=True)

plt.style.use('ggplot')

np.random.seed(123)

c1x = np.array([.1, .2, .35])
c1y = np.array([.3, .4, .2])

c2x = 1 - c1x + np.random.rand(3) * 0.1
c2y = 1 - c1y + np.random.rand(3) * 0.1

c3x = 1 - c1x + np.random.rand(3) * 0.1
c3y = c1y + np.random.rand(3) * 0.1

c1y += 0.25

datax = 0.7
datay = 0.55

sizes = np.array([50, 50, 50])
sizer = sizes + np.asarray([-45, -35, 80])
sizeb = sizes + np.asarray([-45, 80, -35])
sizeg = sizes + np.asarray([-45, 80, -35])

plt.scatter(c1x, c1y, color='r', s=sizer)
plt.scatter(c2x, c2y, color='b', s=sizeb)
plt.scatter(c3x, c3y, color='g', s=sizeg)
plt.scatter(datax, datay, color='k', marker='x', s=120, linewidths=2.5)

plt.xlabel('$w_1$', size=16)
plt.ylabel('$w_2$', size=16)
plt.title("$t=T$", size=18)



vor = Voronoi(list(zip(np.concatenate((c1x, c2x, c3x)), np.concatenate((c1y, c2y, c3y)))))
#voronoi_plot_2d(vor)
plt.show()
