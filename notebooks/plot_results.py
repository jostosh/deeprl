
# coding: utf-8

# In[1]:

import pandas as pd
import os


logbase = '/home/jos/tensorflowlogs/peregrine/v0.9.8'
lpq_infix = 'a3c_ff/lpq=True/glpq=True'
ff_infix = 'a3c_ff/lpq=False/glpq=False'
lpq2_infix= 'a3c_ff/lpq=True/glpq=False'

plot_items = [('A3C softmax', ff_infix), ('A3C LPQ', lpq2_infix), ('A3C GLPQ', lpq_infix)]


# In[14]:

fontsize = 22
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')
mpl.rc('text', usetex=True)
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman'], 'size': fontsize})
mpl.rc('xtick', labelsize=fontsize)
mpl.rc('ytick', labelsize=fontsize)
mpl.rc('figure', autolayout=True)


# In[3]:

import colorlover as cl

colorscale = cl.scales['8']['qual']['Dark2']
colorscalen = []

for c in cl.to_numeric(colorscale):
    colorscalen.append((c[0]/255., c[1]/255, c[2]/255))
colorscalen.append((0., 0., 0.))
colorscalen.append((1., 0., 0.))
linestyles = ['solid', 'dashed', 'dashed', 'dashdot']
dashed = [[], (10, 3), (20, 3)]



# In[5]:

envs = os.listdir(logbase)
print(envs)


# In[10]:

# In[4]:

def read_all_runs(path):
    dfs = []
    dirs = sorted(os.listdir(path), reverse=True)
    for subdir in dirs[:min(5, len(dirs))]:
        results_path = os.path.join(path, subdir, 'evals.csv')
        dfs.append(pd.read_csv(results_path, names=['Epoch', 'Score'], sep='\t'))
    return pd.concat(dfs)
        


# In[15]:

fig, ax = plt.subplots()

dfs_ff = []

all_dfs = []
for env_name in ['Pong']:
    #dfs_lpq = read_all_runs(os.path.join(logbase, env_name, lpq_infix))

    handles = []
    for i, (name, infix) in enumerate(plot_items):
        print("Reading {}".format(os.path.join(logbase, env_name, infix)))
        df = read_all_runs(os.path.join(logbase, env_name, infix))
        averages = df.groupby(('Epoch',)).mean().reset_index()
        stds = df.groupby(('Epoch',)).std().reset_index()
        
        all_dfs.append(df)
        
        plt.fill_between(averages['Epoch'], averages['Score'] - stds['Score'],
                         averages['Score'] + stds['Score'], alpha=0.2, color=colorscalen[i])
        handles.append(plt.plot(averages['Epoch'], averages['Score'], linewidth=2, color=colorscalen[i], ls=linestyles[i], dashes=dashed[i], label=name)[0])
        
        
    #dfs_ff_env = read_all_runs(os.path.join(logbase, env_name, ff_infix))
    #dfs_ff_env['Environment'] = env_name
    #dfs_ff_env['Model'] = 'A3C Softmax'
    
    #dfs_ff.append(dfs_ff_env)
    #averages = dfs_ff_env.groupby(('Epoch','Model')).mean().reset_index()
    #stds = dfs_ff_env.groupby(('Epoch', 'Model')).std().reset_index()
    
    #plt.fill_between(averages['Epoch'], averages['Score'] - stds['Score'],
    #                 averages['Score'] + stds['Score'], alpha=0.2, color=colorscalen[0])
    #handles = [plt.plot(averages['Epoch'], averages['Score'], linewidth=1.5, color=colorscalen[0])]

    ax.set_xlabel("Epoch", fontsize=fontsize)
    ax.set_ylabel("Score", fontsize=fontsize)
    ax.set_title(env_name, fontsize=fontsize + 4)
    ax.set_xlim([0, 80])
    plt.legend(handles=handles, framealpha=0., fontsize=fontsize, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join('/home/jos/Dropbox/RUG/6e Jaar/mproj/thesis/lpq/im', 'PongAtari.pdf'))
    plt.show()
    plt.clf()









