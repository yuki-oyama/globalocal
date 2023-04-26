import numpy as np
import pandas as pd
from optimparallel import minimize_parallel
from numdifftools import Hessian
from autograd import grad
from model import PrismRL, RL
from graph import Graph
from utils import Timer
from dataset import *
import time
import json
import argparse
np.random.seed(111)

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
model_arg.add_argument('--parallel', type=str2bool, default=False, help='if implement parallel computation or not')
model_arg.add_argument('--version', type=str, default='test', help='version name')

# Hyperparameters
model_arg.add_argument('--uturn', type=str2bool, default=True, help='if add uturn dummy or not')
model_arg.add_argument('--uturn_penalty', type=float, default=-20., help='penalty for uturn')
model_arg.add_argument('--min_n', type=int, default=0, help='minimum number observed for d')

# parameters
model_arg.add_argument('--mu_g', type=float, default=1., help='scale for global utility')
model_arg.add_argument('--estimate_mu', type=str2bool, default=False, help='if estimate mu_g or not')

# Validation
model_arg.add_argument('--n_samples', type=int, default=1, help='number of samples')
model_arg.add_argument('--test_ratio', type=float, default=0., help='ratio of test samples')

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
    config.version += '_' + time.strftime("%Y%m%dT%H%M")

    # variations
    vars_g, vars_l = [], []
    init_betas_g = []
    init_betas_l = []
    ubs_g, lbs_g = [], []
    ubs_l, lbs_l = [], []
    # 0: global, 1: globalocal, 2: local
    for i_green in range(3):
        for i_side in range(3):
            for i_sky in range(3):
                vg, vl = ['length', 'crosswalk'], []
                beta_g, beta_l = [-0.3, -1.0], []
                ub_g, lb_g = [0., 0.], [None, None]
                ub_l, lb_l = [], []
                if i_green <= 1:
                    vg.append('greenlen')
                    beta_g.append(0.05)
                    ub_g.append(None)
                    lb_g.append(None)
                if i_green >= 1:
                    vl.append('vegetation')
                    beta_l.append(1.0)
                    ub_l.append(None)
                    lb_l.append(None)
                if i_side <= 1:
                    vg.append('sidewalklen')
                    beta_g.append(0.05)
                    ub_g.append(None)
                    lb_g.append(None)
                if i_side >= 1:
                    vl.append('sidewalk')
                    beta_l.append(1.0)
                    ub_l.append(None)
                    lb_l.append(None)
                if i_sky <= 1:
                    vg.append('skylen')
                    beta_g.append(0.2)
                    ub_g.append(None)
                    lb_g.append(None)
                if i_sky >= 1:
                    vl.append('sky')
                    beta_l.append(2.0)
                    ub_l.append(None)
                    lb_l.append(None)
                vars_g.append(vg)
                vars_l.append(vl)
                init_betas_g.append(beta_g)
                init_betas_l.append(beta_l)
                ubs_g.append(ub_g)
                lbs_g.append(lb_g)
                ubs_l.append(ub_l)
                lbs_l.append(lb_l)

    timer = Timer()
    _ = timer.stop()

    # %%
    network_ = 'kannai'
    dir_ = f'data/{network_}/'
    link_data = pd.read_csv(dir_+'link_bidir_rev2302.csv')
    node_data = pd.read_csv(dir_+'node.csv')
    obs_data = pd.read_csv(dir_+'observations_link.csv')

    # add negative var of street without sidewalk
    link_data['length'] /= 10
    link_data['carst'] = (link_data['walkwidth2'] == 0) * 1 * (link_data['crosswalk'] == 0) * 1
    link_data['sidewalklen'] = link_data['walkwidth2']/10 * link_data['length']
    link_data['carstlen'] = link_data['carst'] * link_data['length']
    # landscape features
    link_data['greenlen'] = link_data['vegetation'] * link_data['length']
    link_data['skylen'] = link_data['sky'] * link_data['length']
    features = link_data

    # %%
    obs_data = reset_index(link_data, node_data, obs_data)
    links = {link_id: (from_, to_) for link_id, from_, to_ in link_data[['link_id', 'from_', 'to_']].values}

    # %%
    dests, obs, obs_filled, n_paths, max_len, od_data, samples = read_mm_results(
        obs_data, links, min_n_paths=config.min_n, n_samples=config.n_samples, test_ratio=config.test_ratio, seed_=111, isBootstrap=False)

    # %%
    # number of paths
    print(f"number of paths observed: {n_paths}")
    # loop counts
    n_loops = count_loops(obs_filled)
    for d, n_loop in n_loops.items():
        if n_loop > 0:
            print(f"number of paths including loops observed: {n_loop} for destination {d}")

    # %%
    # Graph
    g = Graph()
    g.read_data(node_data=node_data, link_data=link_data, od_data=od_data) #features=['length', 'walkwidth', 'green', 'gradient', 'crosswalk', 'walkratio', 'width', 'carst', 'greenlen']
    # g.update(T=T)
    # g.links.shape
    # g.edges.shape

    # output
    outputs = {c:{} for c in range(27*config.n_samples)}
    # function for record results
    def record_res(c, i, res, L0, L_val, runtime, init_beta):
        n = c*27 + i
        outputs[n] = {
            'Sample': i+1,
            'L0': L0,
            'LL': -res.fun,
            'Lv': L_val,
            'runtime': runtime,
        }
        Vg = len(vars_g[c])
        for var_name, b, b0 in zip(var_names, res.x, init_beta):
            outputs[n].update({
                f'beta_{var_name}': b, f'b0_{var_name}': b0
            })
        if config.estimate_mu:
            outputs[n].update({
                'mu_g': res.x[-1]
            })
        else:
            outputs[n].update({
                'mu_g': config.mu_g,
            })

    # %%
    for c, (vg, vl, beta_g, beta_l, ub_g, ub_l, lb_g, lb_l) in enumerate(zip(
            vars_g, vars_l, init_betas_g, init_betas_l, ubs_g, ubs_l, lbs_g, lbs_l)):

        # variables and parameters in the model
        xs = {}
        var_names = []
        betas = []
        init_beta = []
        # global variables
        for var_, b0, lb, ub in zip(vg, beta_g, lb_g, ub_g):
            var_name = var_ + '_g'
            var_names.append(var_name)
            xs[var_name] = [features[var_].values, 'link', 0]
            init_beta.append(b0)
            betas.append(
                (f'b_{var_name}', b0, lb, ub, var_name, 0)
            )
        # local variables
        for var_, b0, lb, ub in zip(vl, beta_l, lb_l, ub_l):
            var_name = var_ + '_l'
            var_names.append(var_name)
            xs[var_name] = [features[var_].values, 'link', 1]
            init_beta.append(b0)
            betas.append(
                (f'b_{var_name}', b0, lb, ub, var_name, 0)
            )

        # %%
        # add uturn dummy
        if config.uturn:
            U = (g.senders[:,np.newaxis] == g.receivers[np.newaxis,:]) * (g.receivers[:,np.newaxis] == g.senders[np.newaxis,:])
            U = np.where(U == True)[0]
            uturns = np.zeros_like(g.senders)
            uturns[U] = 1.
            xs['uturn'] = [uturns, 'edge', 0]
            betas.append(('b_uturn', config.uturn_penalty, None, None, 'uturn', 1))

        # %%
        models = {}
        rl = RL(g, xs, betas, mu=1., mu_g=config.mu_g, estimate_mu=config.estimate_mu)
        models['RL'] = rl

        ### Model Estimation
        if config.estimate_mu: init_beta += [1.]
        for i, sample in enumerate(samples):
            print(pd.DataFrame(outputs).T)

            train_obs = sample['train']
            test_obs = sample['test']

            # %%
            # only observed destinations in samples
            rl.partitions = list(train_obs.keys())

            # %%
            rl.beta = np.array(init_beta)
            LL0_rl = rl.calc_likelihood(observations=train_obs)
            print('Initial param values:', init_beta)
            print('RL model initial log likelihood:', LL0_rl)

            # %%
            try:
                # %%
                # print(f"RL model estimation for sample {i}...")
                # timer.start()
                # results_rl = rl.estimate(observations=train_obs, method='L-BFGS-B', disp=False, hess='res')
                # rl_time = timer.stop()
                # print(f"estimation time is {rl_time}s.")
                # rl.print_results(results_rl[0], results_rl[2], results_rl[3], LL0_rl)

                # %%
                def f(x):
                    # compute probability
                    rl.eval_prob(x)
                    # calculate log-likelihood
                    LL = 0.
                    for key_, paths in train_obs.items():
                        p = rl.p[key_]
                        max_len, N = paths.shape
                        Lk = np.zeros(N, dtype=np.float)
                        for j in range(max_len - 1):
                            L = np.array(p[paths[j], paths[j+1]])[0]
                            assert (L > 0 ).all(), f'L includes zeros: key_={key_}, j={j}, pathj={paths[j]}, pathj+1={paths[j+1]}'
                            Lk += np.log(L)
                        LL += np.sum(Lk)
                    return -LL

                # %%
                # estimation
                Niter = 1
                timer.start()
                # myfactr = 1e-10: for tolerance, add ", 'ftol': 1e-2 * np.finfo(float).eps, 'gtol': 1e-20 * np.finfo(float).eps"
                results_rl = minimize_parallel(f, x0=rl.beta, bounds=rl.bounds,
                        options={'disp':False, 'maxiter':100}, callback=callbackF) #, parallel={'max_workers':4, 'verbose': True}
                rl_time = timer.stop()
                print(f"estimation time is {rl_time}s.")
                rl.beta = results_rl.x

                # %%
                if config.test_ratio > 0:
                    # validation
                    rl.partitions = list(test_obs.keys())
                    LL_val_rl = rl.calc_likelihood(observations=test_obs)
                    print('RL model validation log likelihood:', LL_val_rl)
                else:
                    LL_val_rl = 0.
                # %%
                # record results
                # record_res(i, 'RL', results_rl[0], results_rl[2], results_rl[3], LL0_rl, LL_val_rl, rl_time, init_beta)
                record_res(c, i, results_rl, LL0_rl, LL_val_rl, rl_time, init_beta)
                print(f"RL was successfully estimated for sample {i} with spec {c}; param. values = {rl.beta}")

            except:
                print(f"RL is not feasible for sample {i} with spec {c}; param. values = {rl.beta}")


    df_rl = pd.DataFrame(outputs).T
    print(df_rl)
    if config.test_ratio > 0:
        # for validation
        df_rl.to_csv(f'results/{network_}/validation/{config.version}.csv', index=True)
    else:
        # for estimation
        df_rl.T.to_csv(f'results/{network_}/estimation/{config.version}.csv', index=True)

    # %%
    # write config file
    dir_ = f'results/{network_}/'
    dir_ = dir_ + 'validation/' if config.test_ratio > 0 else dir_ + 'estimation/'
    with open(f"{dir_}{config.version}.json", mode="w") as f:
        json.dump(config.__dict__, f, indent=4)
