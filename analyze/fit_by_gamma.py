import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# %%
# estimation results
dir_ = 'results/kannai/estimation/20230609_different_gammas/'
model_files = [
    'RL_GlobaldifGammas_20230609T1504.csv',
    'RL_LocaldifGammas_20230609T1519.csv',
    'RL_LocalMudifGammas_20230609T1540.csv',
]
model_names = [
    'Global',
    'Local_wo_mu',
    'Local_w_mu',
]

# %%
df = pd.read_csv(dir_+model_files[0], index_col=0).T
axis_names = ['gamma']
for f, mname in zip(model_files, model_names):
    ll = pd.read_csv(dir_+f, index_col=0).T['LL'].values
    df[f'{mname}_LL'] = ll
    axis_names.append(f'{mname}_LL')
df = df[axis_names].set_index('gamma')
axis_names.remove('gamma')
print(df)

# %%
dfplot = df.copy()
x = dfplot.index.values

# %%
stats = dfplot.describe()
print(stats)
stats = stats.T
v_min = stats['min'].min()
v_max = stats['max'].max()
lb = np.round(v_min,-2)
ub = np.round(v_max,-2)
lrange = -25 if lb < v_min else -50
urange = 25 if ub > v_max else 50

# %%
def plot(save=False):
    plt.rcdefaults()
    p = plt.rcParams
    p["font.family"] = "Roboto"
    p["figure.figsize"] = 10, 4
    p["figure.dpi"] = 100
    p["figure.facecolor"] = "#ffffff"
    p["font.sans-serif"] = ["Roboto Condensed"]
    # p["font.weight"] = "light"

    p["ytick.minor.visible"] = False
    p["xtick.minor.visible"] = False
    p["axes.grid"] = True
    p["grid.color"] = "0.5"
    p["grid.linewidth"] = 0.5
    p['axes.axisbelow'] = True # put grid behind

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1) #, projection="3d"
    ax.set_xlim(0.899, 1.001)
    ax.set_ylim(-2310, -1640)
    ax.set_xticks(np.linspace(0.9, 1.0, 11))
    ax.set_yticks(np.linspace(-2300, -1650, int((-1650+2300)//50) + 1))
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Log likelihood')

    # contents
    colors = ['b', 'r', 'g', 'y', 'k']
    for a, aname in enumerate(axis_names):
        model_name = model_names[a]
        ax.plot(x, dfplot[aname].values, marker='o', markeredgecolor="white",
                color=colors[a], zorder=3+a, label=model_name)
    # ax.plot(x, local_lv, marker='o', markeredgecolor="white", color='b', zorder=3, label='Local')
    ax.legend()
    if save:
        plt.savefig(dir_+'loglikelihood_gamma.eps')
    else:
        plt.show()

# %%
plot(save=True)
