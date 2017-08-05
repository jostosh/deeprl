import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from matplotlib import rc
from scipy.spatial import Voronoi, voronoi_plot_2d

#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')


rc('font',**{'family': 'serif', 'serif': ['Computer Modern Roman'], 'size': 20})

rc('xtick', labelsize=14)
rc('ytick', labelsize=14)

## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rc('figure', autolayout=False)

plt.style.use('ggplot')

p = 0.95

W = np.arange(2, 101)

'''
for p in [0.8, 0.9, 0.95, 0.99]:
    gamma = np.log(-(p * (W - 1) / (p - 1))) / 2
    max_sm = np.exp(gamma) / (np.exp(gamma) + (W - 1) * np.exp(-gamma))
    plt.plot(W, max_sm, label='$p={}$'.format(p))
plt.xlabel("$|\mathcal A|$", size=16)
plt.ylabel("Maximum $\pi(s,a;\\theta)$", size=16)
#plt.show()
'''

t = np.linspace(0, 1e6)
p0 = 3
pN = 5

gamma = pN - (pN - p0) * np.exp(-t/1e5)

plt.plot(gamma)
plt.show()


p = np.linspace(0.01, 0.999, 100)

pp, ww = np.meshgrid(p, W)
gamma = np.log(-(pp * (ww - 1) / (pp - 1))) / 2

#ax.plot_surface(pp, ww, gamma, linewidth=0.5, cmap='plasma')
#ax.set_xlabel('$p$', size=16)
#ax.set_ylabel('$|\mathcal A|$', size=16)
#ax.set_zlabel('$')
#plt.pcolor(pp, ww, gamma)
CS = plt.contourf(pp, ww, gamma, levels=np.linspace(0, 5, 11))
plt.xlabel('$p$', size=16)
plt.ylabel('$|\mathcal A|$', size=16)
plt.title("Softmax temperature vs. $|\mathcal A|$ and $p$", size=20)
cb = plt.colorbar()
cb.ax.set_title("$\gamma$", size=16)
plt.savefig('/home/jos/Dropbox/RUG/6e Jaar/mproj/thesis/im/lpq_gamma.pdf')
#plt.show()

plt.clf()
p = np.linspace(0.9, 0.99, 1000)
handles = []
for W in [2, 4, 8, 16, 32]:
    gamma = np.log(-(p * (W - 1) / (p - 1))) / 2
    handles.append(plt.plot(p, gamma, label='$|\mathcal A|={}$'.format(W))[0])
plt.legend(*plt.gca().get_legend_handles_labels(), fontsize=16, loc='upper left').get_frame().set_alpha(0.)
plt.xlabel('$p$', size=16)
plt.ylabel('$\gamma$', size=16)
ax1 = plt.gca()
ax1ticks = ax1.get_xticks()
ax2ticklocations = np.linspace(*ax1.get_xlim(), 5)
ax2ticks = ['0', r'$\frac{1}{4}T$', r'$\frac{1}{2}T$', r'$\frac{3}{4}T$', '$T$']
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(ax2ticklocations)
ax2.set_xticklabels(ax2ticks)
ax2.xaxis.grid(False)
ax2.set_xlabel("Time $(t)$", size=16)
plt.title("Temperature schedules", size=20, y=1.14)
plt.tight_layout()
plt.savefig('/home/jos/Dropbox/RUG/6e Jaar/mproj/thesis/im/lpq_schedules.pdf')
#plt.show()

'''
handles=[]
for p in [0.8, 0.9, 0.95, 0.99]:
    gamma = np.log(-(p * (W - 1) / (p - 1))) / 2
    max_sm = np.exp(gamma) / (np.exp(gamma) + (W - 1) * np.exp(-gamma))
    handles.append(plt.plot(W, gamma, label='$p={}$'.format(p), linewidth=2.0)[0])
plt.xlabel("$|\mathcal A|$", size=16)
plt.ylabel("$\gamma$", size=16)
plt.legend(handles=handles, loc='lower right', fontsize=16)
plt.show()
'''

