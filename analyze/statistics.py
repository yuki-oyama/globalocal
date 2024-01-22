# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
%matplotlib inline

# %%
# this is for raw data
# but this data was processed by read_mm_results()
# so the numbers were different from data actually used for estimation
# data_dir = os.path.join('data', 'kannai')
# obs_data = pd.read_csv(os.path.join(data_dir, 'observations_link.csv'))
# obs_data["モニターID"].unique().shape
# obs_data["ダイアリーID"].unique().shape

# %%
user_path = pd.read_csv('../data/kannai/user_path_used.csv', index_col=0)

# %%
def plot(save=False):
    plt.rcdefaults()
    p = plt.rcParams
    p["font.family"] = "Helvetica"
    p["figure.figsize"] = 8, 4
    p["figure.dpi"] = 200
    p["figure.facecolor"] = "#ffffff"
    p["font.sans-serif"] = ["Helvetica"]
    p["font.weight"] = "light"
    p["axes.grid"] = False
    p["grid.color"] = "0.5"
    p["grid.linewidth"] = 0.5
    p['axes.axisbelow'] = True # put grid behind
    p['pdf.fonttype'] = 42

    bins = np.arange(21)

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1) #, projection="3d"
    ax.set_xlim(0, 20)
    ax.set_xticks(np.linspace(0, 20, int(10) + 1))
    ax.set_xlabel('Number of observed paths')
    ax.set_ylabel('Number of individuals')

    bins = np.arange(20) + 0.5
    user_path.hist(ax=ax, bins=bins, 
                                # density = 1,
                                # alpha = 0.7,
                                color ='black')
        
    if not save:
        plt.show()
    else:
        plt.savefig(save + '.pdf')

# %%
user_path.describe()
# %%
