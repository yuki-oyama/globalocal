import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# %%
# validation results
dir_ = 'results/kannai/validation/'
model_files = [
    'Global_m_20230426T1240.csv',
    'Local_wo_m_greenlen_20230426T1209.csv',
    'Local_w_mu_newb_20230507T1546.csv',
    # 'GlobaLocal_wo_m_greenlen_20230426T1208.csv',
    # 'GlobaLocal_w_mu_newb_20230507T1421.csv'
]
model_names = [
    'Global',
    'Local_wo_mu',
    'Local_w_mu',
    # 'GlobalLocal_wo_mu',
    # 'GlobalLocal_w_mu'
]

# %%
df = pd.read_csv(dir_+model_files[0], index_col=0)
axis_names = []
for f, mname in zip(model_files, model_names):
    lv = pd.read_csv(dir_+f, index_col=0)['Lv'].values
    df[f'{mname}_LV'] = lv
    axis_names.append(f'{mname}_LV')
df = df[axis_names]
df = df.dropna(how='any', axis=0) # drop samples for which nan results are produced for any model
# df = df[12:]
for mname in model_names:
    lv = df[f'{mname}_LV']
    lv /= 82
    lv_new = np.array([
        np.sum(lv[:i+1])/(i+1) for i in range(lv.shape[0])
    ])
    df[f'{mname}_LV'] = lv_new
if len(df) == 30:
    dflen = 30
elif len(df) >= 20:
    dflen = 20
else:
    dflen = 10
df = df[:dflen]
print(df)
(df['Local_wo_mu_LV'] > df['Local_w_mu_LV']).sum()

# %%
dfplot = df.copy()
x = np.arange(len(dfplot)) + 1
# global_lv = global_df['Lv'].values
# local_lv = local_df['Lv'].values

# %%
stats = dfplot.describe()
print(stats)
stats = stats.T
v_min = stats['min'].min()
v_max = stats['max'].max()
lb = np.round(v_min,1)
ub = np.round(v_max,1)
lrange = -0.01 if lb < v_min else -0.05
urange = 0.01 if ub > v_max else 0.05

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

    p["ytick.minor.visible"] = True
    p["xtick.minor.visible"] = False
    p["axes.grid"] = True
    p["grid.color"] = "0.5"
    p["grid.linewidth"] = 0.5
    p['axes.axisbelow'] = True # put grid behind

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1) #, projection="3d"
    ax.set_xlim(0.8, len(dfplot)+0.2)
    ax.set_ylim(lb+lrange, ub+urange)
    ax.set_xticks(np.linspace(x.min(), x.max(), int(x.max()-x.min()) + 1))
    ax.set_yticks(np.linspace(lb, ub, int((ub-lb)//0.049) + 1))
    ax.set_xlabel('Sample number')
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
        plt.savefig(dir_+'validation_res_1_30.eps')
    else:
        plt.show()

# %%
plot(save=False)
