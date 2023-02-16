import numpy as np
import pandas as pd
from copy import deepcopy

mm_renames = {
    'モニターID': 'monitor_id',
    'ダイアリーID': 'trip_id',
    '交通手段': 'mode',
    'リンクID': 'link_id',
    'リンク出発時刻': 'dep_time',
    '所要時間(s)': 'dur_sec',
    '速度(km/h)': 'speed_km_h',
    'リンクコスト': 'link_cost',
    'リンク長(m)': 'link_length'
}

def count_loops(obs):
    n_loops = {k:0 for k in obs.keys()}
    for k in obs.keys():
        fin_state = obs[k][-1,-1]
        N = obs[k].shape[1]
        for n in range(N):
            t = np.where(obs[k][:,n] == fin_state)[0][0]
            traj = obs[k][:,n][:t+1]
            if len(np.unique(traj)) < traj.shape[0]:
                n_loops[k] += 1
    return n_loops

def reset_index(link_data, node_data, obs_data):
    """method to convert link/node ids to indexes
    """
    # define index as link_id
    link_data['link_id'] = link_data.index.values
    node_data['node_id'] = node_data.index.values
    link_idx = {num:idx_ for num, idx_ in link_data[['fid', 'link_id']].values}
    node_idx = {num:idx_ for num, idx_ in node_data[['fid', 'node_id']].values}

    # convert from and to nodes into indexes
    from_nodes = link_data['O'].values
    to_nodes = link_data['D'].values
    link_data['from_'] = [node_idx[n] for n in from_nodes]
    link_data['to_'] = [node_idx[n] for n in to_nodes]

    if obs_data is not None:
        # rename japanese to english
        if 'モニターID' in obs_data.columns.values:
            renamed_obs_data = obs_data.rename(columns=mm_renames)
        else:
            renamed_obs_data = obs_data

        links = renamed_obs_data['link_id'].values
        renamed_obs_data['link_id'] = [link_idx[l] for l in links]
        return renamed_obs_data
    else:
        return None

def read_mm_results(obs_df, links, min_n_paths=1, n_samples=1, test_ratio=0., seed_=111):
    """reading map matching results and obtain samples
    Arguments:
        obs_df (pd.DataFrame): map matching 'link' file
        links (dict): {link_id: (from_, to_)}
        min_n_paths (int): minimum number of paths for destination to be included
        n_samples (int): number of samples
        test_ratio (float): value in (0, 1); train_ratio is 1 - test_ratio
        seed_ (int): seed number, computer random state
    """
    # dummy link for destination (common among all destinations)
    d_dummy_link = max(list(links.keys())) + 1

    # rename japanese to english
    if 'モニターID' in obs_df.columns.values:
        obs_df = obs_df.rename(columns=mm_renames)

    # obtain paths
    dests, obs = [], {}
    path, path_len = [], 0.
    trip_id = None
    n_paths, max_len = 0, 0
    for i in obs_df.index:
        link = obs_df.loc[i]
        # trip change
        if trip_id != link['trip_id']:
            # do not record when n_links <= 1
            if len(path) > 1 and path_len >= 100:
                d = links[path[-1]][1]
                if d not in dests:
                    dests.append(d)
                    obs[d] = []
                path.append(d_dummy_link)
                obs[d].append(path)
                n_paths += 1
                if len(path) > max_len: max_len = len(path)
            # renew path
            path = [link['link_id']]
            path_len = link['link_length']
        else:
            path.append(link['link_id'])
            path_len += link['link_length']
        trip_id = link['trip_id']

    # set the lengths of all paths for the same d to the same length
    ods = []
    obs_filled = {}
    obs_for_sampling = [[], []] # for validation, or to devide into sub samples
    dests_in_sample = []
    for d in dests:
        if len(obs[d]) < min_n_paths:
            continue
        dests_in_sample.append(d)
        obs_d = deepcopy(obs[d])
        path_size = max([len(path) for path in obs_d])
        # print(path_size)
        for n, path in enumerate(obs_d):
            if len(path) < path_size:
                obs_d[n] = path + [d_dummy_link for _ in range(path_size-len(path))]
            o = links[path[0]][0]
            if [o,d] not in ods: ods.append([o,d])
            # for sampling observations
            obs_for_sampling[0].append(d)
            obs_for_sampling[1].append(obs_d[n])
        obs_filled[d] = np.array(obs_d, dtype=np.int).T

    # sampling
    rawdf = pd.DataFrame({'d': obs_for_sampling[0], 'path': obs_for_sampling[1]})
    samples = []
    for i in range(n_samples):
        train_df = rawdf.sample(frac=(1-test_ratio), random_state=seed_+i)
        test_df = rawdf.drop(train_df.index)
        if i == 0: print(f'n_trains is {len(train_df)}; n_tests is {len(test_df)}')
        sample = {
            'train': {d: np.array([path for path in train_df.query(f'd == {d}')['path']]).T for d in train_df['d'].unique()},
            'test': {d: np.array([path for path in test_df.query(f'd == {d}')['path']]).T for d in test_df['d'].unique()},
        }
        samples.append(sample)

    # od data
    ods = np.array(ods)
    od_data = pd.DataFrame({'origin': ods[:,0], 'destination':ods[:,1]})
    return dests_in_sample, obs, obs_filled, n_paths, max_len, od_data, samples

def analyze_detour_rate(g, obs):
    """analyze detour rate
    g: graph object
    obs: observation dictionary with key being destination; value is list of paths
    """
    min_steps = []
    obs_steps = []
    origins, dests = [], []
    paths, min_paths = [], []
    for d in g.dests:
        _, Dd, __, min_path_d = g._compute_minimum_steps(d=d, return_path=True)
        for path in obs[d]:
            min_steps.append(Dd[path[0]])
            obs_steps.append(len(path)-1)
            origins.append(path[0])
            dests.append(d)
            paths.append(path)
            min_paths.append(min_path_d[path[0]])
    detour_df = pd.DataFrame(
        {'min_step': min_steps, 'obs_step': obs_steps, 'origin': origins, 'destination': dests, 'path': paths, 'min_path': min_paths}
        )
    detour_df['detour_rate'] = detour_df['obs_step']/detour_df['min_step']
    return detour_df
