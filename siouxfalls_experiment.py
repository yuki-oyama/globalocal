import numpy as np
import pandas as pd
from optimparallel import minimize_parallel
from core.model import RL
from core.graph import Graph
from core.utils import Timer
from core.dataset import count_loops
import argparse
np.random.seed(124)

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
  return v.lower() in ('true', '1')

def float_or_none(value):
    try:
        return float(value)
    except:
        return None

# Model parameters
model_arg = add_argument_group('Model')
model_arg.add_argument('--rl', type=str2bool, default=True, help='if estimate RL or not')
model_arg.add_argument('--init_beta', nargs='+', type=float, default=[-1.,1.], help='initial parameter values')
model_arg.add_argument('--vars', nargs='+', type=str, default=['length', 'caplen'], help='explanatory variables')
model_arg.add_argument('--true_beta_g', nargs='+', type=float, default=[-2.5,0.5], help='initial parameter values')
model_arg.add_argument('--true_beta_l', nargs='+', type=float, default=[-2.5,2.0], help='initial parameter values')
model_arg.add_argument('--uturn_penalty', type=float, default=-20., help='penalty for uturn')
model_arg.add_argument('--uturn', type=str2bool, default=True, help='if add uturn dummy or not')
model_arg.add_argument('--n_obs', type=int, default=1000, help='number of samples for each od')
model_arg.add_argument('--n_samples', type=int, default=10, help='number of samples')
model_arg.add_argument('--parallel', type=str2bool, default=False, help='if implement parallel computation or not')

def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed

Niter = 1
def callbackF(x):
    global Niter
    txt = f'{Niter: d}'
    for i in range(len(x)): txt += f'\t{x[i]:.4f}'
    print(txt)
    Niter += 1

# %%
if __name__ == '__main__':
    config, _ = get_config()
    timer = Timer()
    _ = timer.stop()

    # networks
    dir_ = 'data/SiouxFalls/'
    node_data = pd.read_csv(dir_+'node.csv')
    link_data = pd.read_csv(dir_+'link.csv')
    od_data = pd.read_csv(dir_+'od.csv')

    # link features
    features = link_data.rename(columns={'Length ': 'length'})
    features['caplen'] = features['length'] * (features['capacity']/features['capacity'].max())

    # Graph
    g = Graph()
    g.read_data(node_data=node_data, link_data=link_data, od_data=od_data)

    # variables and parameters in the model
    xs_g, xs_l, xs_gl = {}, {}, {}
    betas_g, betas_l, betas_gl = [], [], []
    # global variables
    for k, (var_, b0_g, b0_l) in enumerate(zip(config.vars, config.true_beta_g, config.true_beta_l)):
        xs_g[var_] = [features[var_].values, 'link', 0]
        xs_l[var_] = [features[var_].values, 'link', 1*(k>0)]
        xs_gl[var_+'G'] = [features[var_].values, 'link', 0]
        if k > 0: xs_gl[var_+'L'] = [features[var_].values, 'link', 1]
        betas_g.append(
            (f'b_{var_}', b0_g, None, None, var_, 0)
        )
        betas_l.append(
            (f'b_{var_}', b0_l, None, None, var_, 0)
        )
        betas_gl.append(
            (f'b_{var_}G', b0_g, None, None, var_+'G', 0)
        )
        if k > 0:
            betas_gl.append(
                (f'b_{var_}L', b0_l, None, None, var_+'L', 0)
            )
    # add uturn dummy
    if config.uturn:
        U = (g.senders[:,np.newaxis] == g.receivers[np.newaxis,:]) * (g.receivers[:,np.newaxis] == g.senders[np.newaxis,:])
        U = np.where(U == True)[0]
        uturns = np.zeros_like(g.senders)
        uturns[U] = 1.
        xs_g['uturn'] = (uturns, 'edge', 0)
        xs_l['uturn'] = (uturns, 'edge', 0)
        xs_gl['uturn'] = (uturns, 'edge', 0)
        betas_g.append(('b_uturn', config.uturn_penalty, None, None, 'uturn', 1))
        betas_l.append(('b_uturn', config.uturn_penalty, None, None, 'uturn', 1))
        betas_gl.append(('b_uturn', config.uturn_penalty, None, None, 'uturn', 1))

    # define models
    rl_g = RL(g, xs_g, betas_g)
    rl_l = RL(g, xs_l, betas_l)
    rl_gl = RL(g, xs_gl, betas_gl)
    rl_g.eval_prob()
    rl_l.eval_prob()
    models = {'G': rl_g, 'L': rl_l, 'GL': rl_gl}
    true_beta = {'G': config.true_beta_g, 'L': config.true_beta_l}

    ### Data Generation
    # %%
    os = [1,2,3,4,5,6]
    ds = [8,12,16,20]
    seq_g, seq_l = {}, {}
    N = config.n_obs
    max_len_g, max_len_l = 0, 0
    for d in ds:
        path_g = rl_g.sample_path(os, d, N)
        path_l = rl_l.sample_path(os, d, N)

        if path_g.shape[0] >= max_len_g:
            max_len_g = path_g.shape[0]
        if path_l.shape[0] >= max_len_l:
            max_len_l = path_l.shape[0]

        seq_g[d] = path_g
        seq_l[d] = path_l

    obs = {'G': {i:{} for i in range(config.n_samples)}, 'L': {i:{} for i in range(config.n_samples)}}
    # multiple samples
    if config.n_samples > 1:
        # define for each sample
        sample_size = (N * len(os) * len(ds)) // config.n_samples
        for i in range(config.n_samples):
            d_counts = np.bincount(np.random.choice(ds, sample_size))
            for d in ds:
                idx_ = np.random.choice(np.arange(N * len(os)), d_counts[d], replace=False)
                obs['G'][i][d] = seq_g[d][:, idx_]
                obs['L'][i][d] = seq_l[d][:, idx_]
    else:
        obs['G'][0] = seq_g
        obs['L'][0] = seq_l

    # loop counts
    n_loops_g = count_loops(seq_g)
    print(n_loops_g)
    n_loops_l = count_loops(seq_l)
    print(n_loops_l)

    # %%
    ### Model Estimation
    # output
    datas = ['G', 'L']
    ests = ['G', 'L', 'GL']
    outputs = {d_:{} for d_ in datas}
    for d_ in datas:
        for e_ in ests:
            outputs[d_][e_] = {i:{} for i in range(config.n_samples)}

    # function for record results
    def record_res(d_, e_, i, res, stderr, L0, runtime):
        if e_ != 'GL':
            outputs[d_][e_][i] = {
                'beta_len': res.x[0], 'se_len': stderr[0],
                'beta_cap': res.x[1], 'se_cap': stderr[1],
                'L0': L0,
                'LL': -res.fun,
                'runtime': runtime,
            }
        elif e_ == 'GL':
            outputs[d_][e_][i] = {
                'beta_len': res.x[0], 'se_len': stderr[0],
                'beta_capG': res.x[1], 'se_cap': stderr[1],
                'beta_capL': res.x[2], 'se_cap': stderr[2],
                'L0': L0,
                'LL': -res.fun,
                'runtime': runtime,
            }

    # %%
    init_beta = {'G': [-1.0, 0.1], 'L': [-1.0, 1.0], 'GL': [-1.0, 0.1, 1.0]}
    for d_ in datas:
        for e_ in ['GL']:
            for i in range(config.n_samples):
                rl = models[e_]
                rl.beta = np.array(init_beta[e_])
                LL0_rl = rl.calc_likelihood(observations=obs[d_][i])
                print(f'Estimating model {e_} with data generated by model {d_}')
                print('Initial log likelihood:', LL0_rl)

                try:
                    print(f"Model estimation for sample {i}...")
                    timer.start()
                    results_rl = rl.estimate(observations=obs[d_][i], method='L-BFGS-B', disp=False, hess='res')
                    rl_time = timer.stop()
                    print(f"estimation time is {rl_time}s.")
                    # rl.print_results(results_rl[0], results_rl[2], t_rl, LL0_rl)
                    record_res(d_, e_, i, results_rl[0], results_rl[2], LL0_rl, rl_time)
                except:
                    print(f'RL failed for sample {i}')

    # %%
    for d_ in datas:
        for e_ in ests:
            print(f'Estimation result of model {e_} with data generated by model {d_}')
            df_rl = pd.DataFrame(outputs[d_][e_]).T
            print(df_rl)
            df_rl.to_csv(f'results/SiouxFalls/estRes_model{e_}_data{d_}.csv', index=True)
