import pandas as pd
import numpy as np

dir_ = 'data/kannai/'
sc_links = pd.read_csv(dir_+'link_bidir_reduced_scenario.csv')
sc_links = sc_links.fillna(0)

new_links_g = [2392,1193,286,1485,1492,293,2375,1176,2374,1175,1194,2393]
# new_links_l = [2385,1186]
new_links_l = [1114,2313,1399,200,1490,291]
fids = sc_links['fid'].values

new_links_g_idxs = []
for link in new_links_g:
    i = np.where(fids == link)[0][0]
    new_links_g_idxs.append(i)

new_links_l_idxs = []
for link in new_links_l:
    i = np.where(fids == link)[0][0]
    new_links_l_idxs.append(i)

green_g = np.zeros(len(sc_links))
green_l = np.zeros(len(sc_links))
green_g[new_links_g_idxs] = 1
green_l[new_links_l_idxs] = 1

sc_links['green_g'] = green_g
sc_links['green_l'] = green_l

sc_links.to_csv(dir_+'link_bidir_reduced_scenario2.csv', index=False)
