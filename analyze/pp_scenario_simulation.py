import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from graph import Graph
from dataset import *
import geopandas as gpd
from shapely import wkt
from model import RL, PrismRL
from copy import deepcopy

# %%
delta_green = 0.4

# %%
# networks
dir_ = 'data/kannai/'
res_dir = 'results/kannai/simulation/'
# original network data
link_data = pd.read_csv(dir_+'link_bidir_rev2302.csv')
node_data = pd.read_csv(dir_+'node.csv')
# for visualization focus
sc_links = pd.read_csv(dir_+'link_bidir_reduced_scenario2.csv')
sc_links = sc_links.fillna(0)
sc_link_idxs = sc_links['fid'].values - 1
sc_links['vegetation'] = link_data['vegetation'].values[sc_link_idxs]
sc_links['sidewalk'] = link_data['walkwidth2'].values[sc_link_idxs]
sc_links['vegetation_sc_g'] = sc_links['vegetation'] + sc_links['green_g']*delta_green
sc_links['vegetation_sc_l'] = sc_links['vegetation'] + sc_links['green_l']*delta_green
sc_links['vegetation_sc_gl'] = sc_links['vegetation'] + (sc_links['green_g']+sc_links['green_l'])*delta_green
## define variables
# for scenario
green_g = np.zeros(len(link_data))
green_g[sc_link_idxs] = sc_links['green_g'].values
green_l = np.zeros(len(link_data))
green_l[sc_link_idxs] = sc_links['green_l'].values
link_data['green_g'] = green_g
link_data['green_l'] = green_l
# link_data = links.iloc[sc_links.index.values]
# scenarios = sc_links[['green_g', 'green_l']]
# link_data = pd.concat([link_data, scenarios], axis=1)
# variables
link_data['length'] /= 10
link_data['sidewalklen'] = link_data['walkwidth2']/10 * link_data['length']
link_data['greenlen'] = link_data['vegetation'] * link_data['length']
link_data['skylen'] = link_data['sky'] * link_data['length']
link_data['greenlen_g'] = (link_data['vegetation'] + delta_green*link_data['green_g']) * link_data['length']
link_data['greenlen_l'] = (link_data['vegetation'] + delta_green*link_data['green_l']) * link_data['length']
features = link_data

# %%
# scenarios
green_scenarios = {
    # np.zeros_like(link_data['greenlen_g'].values, dtype=np.float),
    'base': link_data['greenlen'].values,
    'global': link_data['greenlen_g'].values,
    'local': link_data['greenlen_g'].values + link_data['greenlen_l'].values
}

# %%
# models
df_G = pd.read_csv('results/kannai/estimation/20230425_sidewalk_bootstrapping/Global_BS_20230424T1921.csv', index_col=0).T
df_dG = pd.read_csv('results/kannai/estimation/20230428_discounted_RL_bootstrapping/DRL0.99_Global_20230428T1512.csv', index_col=0).T
df_L = pd.read_csv('results/kannai/estimation/20230425_sidewalk_bootstrapping/Local_wo_mu_greenlen_BS_20230425T1153.csv', index_col=0).T
df_Lm = pd.read_csv('results/kannai/estimation/20230507_with_mu_new_bound/Local_w_mu_newb_20230507T1545.csv', index_col=0).T
df_dL = pd.read_csv('results/kannai/estimation/20230428_discounted_RL_bootstrapping/DRL0.99_Local_wo_m_greenlen_20230428T1510.csv', index_col=0).T

# %%
# true parameters & probtbility
beta_g = [
    ('b_len', df_G['beta_length_g'].values[0], None, None, 'length', 0),
    ('b_cross', df_G['beta_crosswalk_g'].values[0], None, None, 'crosswalk', 0),
    ('b_sidewalk', df_G['beta_sidewalklen_g'].values[0], None, None, 'sidewalklen', 0),
    ('b_green', df_G['beta_greenlen_g'].values[0], None, None, 'greenlen', 0),
    ('b_uturn', -20., None, None, 'uturn', 1)
]
beta_dg = [
    ('b_len', df_dG['beta_length_g'].values[0], None, None, 'length', 0),
    ('b_cross', df_dG['beta_crosswalk_g'].values[0], None, None, 'crosswalk', 0),
    ('b_sidewalk', df_dG['beta_sidewalklen_g'].values[0], None, None, 'sidewalklen', 0),
    ('b_green', df_dG['beta_greenlen_g'].values[0], None, None, 'greenlen', 0),
    ('b_uturn', -20., None, None, 'uturn', 1)
]
beta_l = [
    ('b_len', df_L['beta_length_g'].values[0], None, None, 'length', 0),
    ('b_cross', df_L['beta_crosswalk_g'].values[0], None, None, 'crosswalk', 0),
    ('b_sidewalk', df_L['beta_sidewalklen_g'].values[0], None, None, 'sidewalklen', 0),
    ('b_green', df_L['beta_greenlen_l'].values[0], None, None, 'greenlen', 0),
    ('b_uturn', -20., None, None, 'uturn', 1)
]
beta_lm = [
    ('b_len', df_Lm['beta_length_g'].values[0], None, None, 'length', 0),
    ('b_cross', df_Lm['beta_crosswalk_g'].values[0], None, None, 'crosswalk', 0),
    ('b_sidewalk', df_Lm['beta_sidewalklen_g'].values[0], None, None, 'sidewalklen', 0),
    ('b_green', df_Lm['beta_greenlen_l'].values[0], None, None, 'greenlen', 0),
    ('b_uturn', -20., None, None, 'uturn', 1)
]
beta_dL = [
    ('b_len', df_dL['beta_length_g'].values[0], None, None, 'length', 0),
    ('b_cross', df_dL['beta_crosswalk_g'].values[0], None, None, 'crosswalk', 0),
    ('b_sidewalk', df_dL['beta_sidewalklen_g'].values[0], None, None, 'sidewalklen', 0),
    ('b_green', df_dL['beta_greenlen_l'].values[0], None, None, 'greenlen', 0),
    ('b_uturn', -20., None, None, 'uturn', 1)
]

# %%
obs_data = reset_index(link_data, node_data, None)

# %%
# origin candidates: 70: station link 195 (70,725); 139: left-bottom corner of park link 1546 (139,138)
# dest candidates: 81: top; 74: after crosswalk
o = node_data.query('fid == 70')['node_id'].values[0] #653
d = node_data.query('fid == 118')['node_id'].values[0] #81
od_data = {'origin': [o], 'destination': [d], 'flow': [0]}
od_data = pd.DataFrame(od_data)
# Graph
g = Graph()
g.read_data(node_data=node_data, link_data=link_data, od_data=od_data) #features=['length', 'crosswalk', 'greenlen']

# %%
# add uturn dummy
U = (g.senders[:,np.newaxis] == g.receivers[np.newaxis,:]) * (g.receivers[:,np.newaxis] == g.senders[np.newaxis,:])
U = np.where(U == True)[0]
uturns = np.zeros_like(g.senders)
uturns[U] = 1.

# %%
models = {}
for s, green in green_scenarios.items():
    features_g = {
        'length': [features['length'].values, 'link', 0],
        'crosswalk': [features['crosswalk'].values, 'link', 0],
        'sidewalklen': [features['sidewalklen'].values, 'link', 0],
        'greenlen': [green, 'link', 0],
        'uturn': [uturns, 'edge', 0],
    }
    # global model
    model_G = RL(g, features_g, beta_g, mu=1., mu_g=df_G['mu_g'].values[0], estimate_mu=False)
    models[f'modelG_{s}'] = model_G
    model_dG = RL(g, features_g, beta_dg, mu=1., mu_g=df_dG['mu_g'].values[0], estimate_mu=False, gamma=df_dG['gamma'].values[0])
    models[f'modeldG_{s}'] = model_dG
    # local model
    features_l = deepcopy(features_g)
    features_l['greenlen'][2] = 1
    model_L = RL(g, features_l, beta_l, mu=1., mu_g=df_L['mu_g'].values[0], estimate_mu=False)
    models[f'modelL_{s}'] = model_L
    model_Lm = RL(g, features_l, beta_l, mu=1., mu_g=df_Lm['mu_g'].values[0], estimate_mu=False)
    models[f'modelLm_{s}'] = model_Lm
    # model_dL = RL(g, features_l, beta_l, mu=1., mu_g=df_dL['mu_g'].values[0], estimate_mu=False, gamma=df_dL['gamma'].values[0])
    # models[f'modeldL_{s}'] = model_dL

# %%
N = 500
# 195 is the original fid connecting from nodes 70 to 725
# 1546 is the original fid connecting from nodes 139 to 138
# 1396 is the original fid connecting from nodes 642 to 70
olink_id = link_data[link_data['fid'] == 1396]['link_id'].values[0]
# onode_id = link_data[link_data['fid'] == 195]['to_'].values[0]
os = [olink_id] # o
sample_paths = {}
for scenario, model in models.items():
    print(scenario)
    model.eval_prob()
    # sampling
    seq = model.sample_path(os, d, N, origin='link', max_len=100)
    sample_paths[scenario] = seq


# %%
seq[:,0]

# %%
link_flows = {}
for scenario, seq in sample_paths.items():
    print(scenario)
    # plot(key_=None, file_path=None, path_=seq[:,50])
    link_flow = np.zeros(len(link_data), dtype=np.float)
    for n in range(N):
        path_ = seq[:,n]
        for link in path_:
            if link != link_flow.shape[0]:
                link_flow[link] += 1
    link_flows[scenario] = link_flow

# %%
fids = [201, 1397]
for fid in fids:
    a = link_data[link_data['fid'] == fid]['link_id'].values[0]
    print(a)
    fstars = models['modelL_base'].p[d][a].nonzero()
    print(fstars[1])
    for s, model in models.items():
        print(s, model.p[d][a][fstars], link_flows[s][fstars[1]])

# %%
"""visualization"""
# %%
reduced_nodes = np.unique(np.concatenate([sc_links['O'].values, sc_links['D'].values]))
reduced_idxs = []
for i in node_data.index:
    fid = node_data.loc[i, 'fid']
    if fid in reduced_nodes:
        reduced_idxs.append(i)
sc_nodes = node_data.iloc[reduced_idxs]
sc_nodes = sc_nodes.reset_index()

# %%
_ = reset_index(sc_links, sc_nodes, None)

# %%
# convert to GeoDataFrame
node_gdf = gpd.GeoDataFrame(
    sc_nodes, geometry=gpd.points_from_xy(sc_nodes.x, sc_nodes.y))
sc_links['WKT'] = gpd.GeoSeries.from_wkt(sc_links['WKT'])
link_gdf = gpd.GeoDataFrame(sc_links, geometry='WKT')

# %%
node_pos = {}
node_idx = {}
for i in sc_nodes.index:
    node = sc_nodes.loc[i]
    node_pos[node['node_id']] = (node['x'], node['y'])
    node_idx[node['node_id']] = i

# %%
xs = sc_nodes['x'].values
ys = sc_nodes['y'].values
xmin = xs.min()
xmax = xs.max()
ymin = ys.min()
ymax = ys.max()

# %%
pos = np.array(list(node_pos.values()))

# %%
xy = []
for j in sc_links.index:
    link = sc_links.loc[j]
    from_, to_  = link[['from_', 'to_']]
    x1, y1 = node_pos[from_]
    x2, y2 = node_pos[to_]
    xy.append((x1, y1, x2, y2))

# %%
## detour trips
# get paths
# links = {link_id: (from_, to_) for link_id, from_, to_ in sc_links[['link_id', 'from_', 'to_']].values}

# %%
## Visualization
def plot(key_='len', file_path=None, path_=None, flow=None):
    plt.rcdefaults()
    p = plt.rcParams
    p["font.family"] = "Roboto"
    p["figure.figsize"] = 6, 4
    p["figure.dpi"] = 100
    p["figure.facecolor"] = "#ffffff"
    p["font.sans-serif"] = ["Roboto Condensed"]
    # p["font.weight"] = "light"

    # p["ytick.minor.visible"] = True
    # p["xtick.minor.visible"] = True
    # p["axes.grid"] = True
    # p["grid.color"] = "0.5"
    # p["grid.linewidth"] = 0.5
    p['axes.axisbelow'] = True # put grid behind

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1) #, projection="3d"
    # ax.set_xlim(xmin-5, xmax+5)
    # ax.set_ylim(ymin-5, ymax+5)

    # contents
    # node plot
    if flow is not None:
        cmap = plt.get_cmap("Blues")
        link_gdf.plot(ax=ax, linewidths=0.1, column=None, color='gray') #
        # ax.scatter(pos[:,0], pos[:,1], marker='o', color=cmap(1.0), s=8, zorder=100)
        for (x1, y1, x2, y2), f in zip(xy, flow):
            i_rev = xy.index((x2,y2,x1,y1))
            if f > flow[i_rev]:
                zorder = (f > 0.025*N)*10 + 1
                ax.plot((x1, x2), (y1, y2), marker=None, c=cmap((f/(N/2))),
                        linewidth=1.5, zorder=zorder) #, alpha=0.9
    # short_links.plot(ax=ax, column=key_, linewidths=1.5, color='r')
    else:
        if key_[:10] == 'vegetation':
            cmap = plt.get_cmap("Greens")
            link_gdf.plot(ax=ax, linewidths=1., column=key_, cmap=cmap, vmin=0., vmax=.8, legend=True) #, color='gray'
        else:
            link_gdf.plot(ax=ax, linewidths=1., column=key_, legend=True) #, color='gray'
        if path_ is not None:
            # link_gdf.plot(ax=ax, linewidths=1.0, column='green')
            obs_link_gdf = link_gdf.query(f'link_id in {list(path_)}')
            obs_link_gdf.plot(ax=ax, linewidths=1.5, color='b', alpha=0.5)

    if file_path is not None:
        plt.savefig(f'{res_dir}{file_path}.eps')
    else:
        plt.show()

# %%
for key_ in ['vegetation', 'vegetation_sc_g', 'vegetation_sc_gl']:
    plot(key_=key_, file_path=key_+f'_{delta_green}')

# %%
plot(key_='sidewalk')

# %%
for scenario, link_flow in link_flows.items():
    print(scenario)
    if scenario[:7] == 'modelLm':
        plot(key_=None, flow=link_flow[sc_link_idxs], file_path=f'{scenario}_{delta_green}') #
