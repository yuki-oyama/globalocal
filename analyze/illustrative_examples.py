"""
Braess network (update: 2023/5/7)
    - with three routes
    - Example 1: with congestion (perception update)
    - Example 2: with scape (local perception)
"""

# %%
import numpy as np
import pandas as pd
from model import RL, PrismRL
from graph import Graph
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(111)

# %%
# networks
network_ = 'ladder'
dir_ = f'data/{network_}/'
node_data = pd.read_csv(dir_+'node.csv')
link_data = pd.read_csv(dir_+'link.csv')
od_data = pd.read_csv(dir_+'od.csv')
output = {}

# %%
routes = {}
with open(dir_+'route.csv') as f:
    lines = f.readlines()
for r, line in enumerate(lines):
    line = line.replace('\n','')
    items = line.split(',')
    route = [int(item) for item in items]
    routes[r+1] = route
print(routes)

# %%
mu_g = 1.0

# %%
def compute_path_prob(P):
    probs = {}
    for key_, route in routes.items():
        prob = 1.
        for k in range(len(route)-1):
            prob *= P[route[k],route[k+1]]
        probs[f'P{key_}'] = prob
    return probs

def eval_path_prob(case, P):
    output[case] = {}
    for key_, route in routes.items():
        prob = 1.
        for k in range(len(route)-1):
            prob *= P[route[k],route[k+1]]
        print(f"prob. of route {key_} = {prob}")
        output[case][f'P{key_}'] = prob

# %%
# Graph
g = Graph()
g.read_data(node_data=node_data, link_data=link_data, od_data=od_data, features=['exp_time', 'dif_time', 'scape'])
d = g.nodes[-1]

## Case 1: only expected time
# %%
x_free = list(g.link_features.values())
xs = {
    'exp_time': (x_free[0], 'link', 0),
    'dif_time': (x_free[1], 'link', 0),
    'scape': (x_free[2], 'link', 1),
}

# true parameters & probtbility
betas = [
    ('b_e_time', -1., None, None, 'exp_time', 0),
]

rl = RL(g, xs, betas, mu=1., mu_g=mu_g, estimate_mu=False)
rl.eval_prob()
P1 = rl.p[d].toarray()
eval_path_prob('case1', P1)

# %%
## Case 2: time difference is globally perceived
xs = {
    'exp_time': (x_free[0], 'link', 0),
    'dif_time': (x_free[1], 'link', 0),
    'scape': (x_free[2], 'link', 1),
}
betas = [
    ('b_e_time', -1., None, None, 'exp_time', 0),
    ('b_d_time', -1., None, None, 'dif_time', 0),
]

rl = RL(g, xs, betas, mu=1., mu_g=mu_g, estimate_mu=False)
rl.eval_prob()
P2 = rl.p[d].toarray()
eval_path_prob('case2', P2)

# %%
## Case 3: difference is perceived locally
xs = {
    'exp_time': (x_free[0], 'link', 0),
    'dif_time': (x_free[1], 'link', 1),
    'scape': (x_free[2], 'link', 1),
}
betas = [
    ('b_e_time', -1., None, None, 'exp_time', 0),
    ('b_d_time', -1., None, None, 'dif_time', 0),
]

rl = RL(g, xs, betas, mu=1., mu_g=mu_g, estimate_mu=False)
rl.eval_prob()
P3 = rl.p[d].toarray()
eval_path_prob('case3', P3)

# %%
## Case 4: scape is perceived locally
xs = {
    'exp_time': (x_free[0], 'link', 0),
    'dif_time': (x_free[1], 'link', 1),
    'scape': (x_free[2], 'link', 1),
}
betas = [
    ('b_e_time', -1., None, None, 'exp_time', 0),
    ('b_scape', 2., None, None, 'scape', 0),
]

rl = RL(g, xs, betas, mu=1., mu_g=mu_g, estimate_mu=False)
rl.eval_prob()
P4 = rl.p[d].toarray()
eval_path_prob('case4', P4)

# %%
results = pd.DataFrame(output).T
print(results)
results.to_csv(f'results/{network_}/path_probs.csv', index=True)

# %%
# routes
# P1[0,2] * P2[2,5] * P2[5,8] * P2[8,-1]
# P1[0,2] * P2[2,3] * P2[3,4] * P2[4,7] * P2[7,-1]


# %%
## Sensitivity analysis for mu_g
sens_results_case3 = {}
sens_results_case4 = {}
sens_results = {3: sens_results_case3, 4: sens_results_case4}

mu_gs = [1.0]
delta_mu = 0.5
mu_gs += [1 + delta_mu * i for i in range(1,31)]

# %%
case = 3

# %%
if case == 3:
    xs = {
        'exp_time': (x_free[0], 'link', 0),
        'dif_time': (x_free[1], 'link', 1),
        'scape': (x_free[2], 'link', 1),
    }
    betas = [
        ('b_e_time', -1., None, None, 'exp_time', 0),
        ('b_d_time', -1., None, None, 'dif_time', 0),
        ('b_scape', 0., None, None, 'scape', 0),
    ]
elif case == 4:
    xs = {
        'exp_time': (x_free[0], 'link', 0),
        'dif_time': (x_free[1], 'link', 1),
        'scape': (x_free[2], 'link', 1),
    }
    betas = [
        ('b_e_time', -1., None, None, 'exp_time', 0),
        ('b_d_time', -1., None, None, 'dif_time', 0),
        ('b_scape', 2., None, None, 'scape', 0),
    ]

# %%
for mu_g in mu_gs:
    rl = RL(g, xs, betas, mu=1., mu_g=mu_g, estimate_mu=False)

    rl.eval_prob()
    P = rl.p[d].toarray()
    z = rl.get_z(d)
    V = np.log(z) / mu_g
    probs = compute_path_prob(P)

    sens_results[case][mu_g] = {
        **probs,
        **{f'V{i}':V[i] for i in range(1,len(V)-1)},
        'z1': z[1],
        'z2': z[2],
        'P01': P[0,1],
        'P02': P[0,2],
        'P23': P[2,3],
        'P25': P[2,5],
        'P47': P[4,7],
        'P46': P[4,6],
    }


# %%
sens_df = pd.DataFrame(sens_results[case]).T
print(sens_df)
# results.to_csv(f'results/illustrative_examples/mug_sens_analysis_for_paper.csv', index=True)

# %%
x = sens_df.index.values

# %%
def plot(save=False):
    plt.rcdefaults()
    p = plt.rcParams
    p["font.family"] = "Roboto"
    p["figure.figsize"] = 6, 4
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
    ax.set_xlim(1-delta_mu, x.max()+delta_mu)
    ax.set_ylim(-0.025, .525)
    ax.set_xticks(np.linspace(1, x.max(), int((x.max()-1))+1))
    ax.set_yticks(np.linspace(0, 0.5, 11))
    ax.set_xlabel('mu_g')
    ax.set_ylabel('Probability')

    # contents
    colors = ['r', 'b', 'g', 'y', 'c']
    for key_, c in zip(routes.keys(), colors):
        p_key = f'P{key_}'
        ps = sens_df[p_key].values
        ax.plot(x, ps, marker='o', markeredgecolor="white", color=c, zorder=4, label=p_key)
    ax.legend()
    if save:
        plt.savefig(f'results/{network_}/mug_sens_analysis_case{case}_ladder3_revMu.eps')
    else:
        plt.show()

# %%
plot(save=False)

# %%
def plotV(save=False):
    plt.rcdefaults()
    p = plt.rcParams
    p["font.family"] = "Roboto"
    p["figure.figsize"] = 6, 4
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
    ax.set_xlim(1-delta_mu, x.max()+delta_mu)
    ax.set_ylim(-4.5, 0.5)
    ax.set_xticks(np.linspace(1, x.max(), int((x.max())-1)+1))
    ax.set_yticks(np.linspace(-4, 0, 9))
    ax.set_xlabel('mu_g')
    ax.set_ylabel('V')

    # contents
    colors = ['r', 'b', 'g', 'y', 'c', 'k', 'gray', 'purple']
    for i, c in zip(range(1,9), colors):
        v_key = f'V{i}'
        vs = sens_df[v_key].values
        ax.plot(x, vs, marker='', markeredgecolor="white", color=c, zorder=4, label=v_key)
    ax.legend()
    if save:
        plt.savefig(f'results/{network_}/mug_V_case{case}.eps')
    else:
        plt.show()

# %%
plotV(save=True)
# sens_df['V1'] - sens_df['V2']
# sens_df['V3'] - sens_df['V5']

# %%
def plotP(keys_, save=False):
    plt.rcdefaults()
    p = plt.rcParams
    p["font.family"] = "Roboto"
    p["figure.figsize"] = 6, 4
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
    ax.set_xlim(-0.05, x.max()+0.05)
    ax.set_ylim(-0.025, 1.025)
    ax.set_xticks(np.linspace(0, x.max(), int((x.max())*5)+1))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_xlabel('mu_g')
    ax.set_ylabel('Probability')

    # contents
    colors = ['r', 'b', 'g', 'y', 'c', 'k', 'gray', 'purple']
    for key_, c in zip(keys_, colors):
        p_key = f'P{key_}'
        ps = sens_df[p_key].values
        ax.plot(x, ps, marker='o', markeredgecolor="white", color=c, zorder=4, label=p_key)
    ax.legend()
    if save:
        plt.savefig(f'results/{network_}/mug_element_probs_case{case}.eps')
    else:
        plt.show()

# %%
keys_ = ['01', '02', '23', '25', '47', '46']
plotP(keys_, save=False)


# %%
## Sensitivity analysis for gamma
sens_results_case3 = {}
sens_results_case4 = {}
sens_results = {3: sens_results_case3, 4: sens_results_case4}

gammas = [1.]
gammas += [1 - 0.05 * i for i in range(1,11)]

# %%
case = 3

# %%
if case == 3:
    xs = {
        'exp_time': (x_free[0], 'link', 0),
        'dif_time': (x_free[1], 'link', 1),
        'scape': (x_free[2], 'link', 1),
    }
    betas = [
        ('b_e_time', -1., None, None, 'exp_time', 0),
        ('b_d_time', -1., None, None, 'dif_time', 0),
        ('b_scape', 0., None, None, 'scape', 0),
    ]
elif case == 4:
    xs = {
        'exp_time': (x_free[0], 'link', 0),
        'dif_time': (x_free[1], 'link', 1),
        'scape': (x_free[2], 'link', 1),
    }
    betas = [
        ('b_e_time', -1., None, None, 'exp_time', 0),
        ('b_d_time', -1., None, None, 'dif_time', 0),
        ('b_scape', 2., None, None, 'scape', 0),
    ]

# %%
for gamma in gammas:
    rl = RL(g, xs, betas, mu=1., mu_g=1., estimate_mu=False, gamma=gamma)

    rl.eval_prob()
    P = rl.p[d].toarray()
    probs = compute_path_prob(P)

    sens_results[case][gamma] = {
        **probs
    }

# %%
sens_df = pd.DataFrame(sens_results[case]).T
print(sens_df)
# results.to_csv(f'results/illustrative_examples/mug_sens_analysis_for_paper.csv', index=True)

# %%
x = sens_df.index.values

# %%
def plot(save=False):
    plt.rcdefaults()
    p = plt.rcParams
    p["font.family"] = "Roboto"
    p["figure.figsize"] = 6, 4
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
    ax.set_xlim(0.48, x.max()+0.02)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.linspace(0.5, x.max(), int((x.max()-0.5)*10)+1))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_xlabel('gamma')
    ax.set_ylabel('Probability')

    # contents
    colors = ['r', 'b', 'g', 'y', 'c']
    for key_, c in zip(routes.keys(), colors):
        p_key = f'P{key_}'
        ps = sens_df[p_key].values
        ax.plot(x, ps, marker='o', markeredgecolor="white", color=c, zorder=4, label=p_key)
    ax.legend()
    if save:
        plt.savefig(f'results/braess/gamma_sens_analysis_case{case}.eps')
    else:
        plt.show()

# %%
plot(save=False)
