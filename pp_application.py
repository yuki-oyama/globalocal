import os
import numpy as np
import pandas as pd
from optimparallel import minimize_parallel
from core.model import RL
from core.graph import Graph
from core.utils import Timer
from core.dataset import *
import time
import json
import argparse


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

## General parameters
gen_arg = add_argument_group('General')
gen_arg.add_argument('--seed', type=int, default=111, help='random seed')
gen_arg.add_argument('--root', type=str, default=None, help='root directory')
gen_arg.add_argument('--data_dir', type=str, default='data', help='data directory')
gen_arg.add_argument('--out_dir', type=str, default='test', help='output directory')
gen_arg.add_argument('--net_name', type=str, default='kannai', help='network for application')

# Model parameters
model_arg = add_argument_group('Model')
model_arg.add_argument('--parallel', type=str2bool, default=False, help='if implement parallel computation or not')

# Hyperparameters
model_arg.add_argument('--uturn', type=str2bool, default=True, help='if add uturn dummy or not')
model_arg.add_argument('--uturn_penalty', type=float, default=-20., help='penalty for uturn')
model_arg.add_argument('--min_n', type=int, default=0, help='minimum number observed for d')

# parameters
model_arg.add_argument('--mu_g', type=float, default=1., help='scale for global utility')
model_arg.add_argument('--vars_g', nargs='+', type=str, default=['length', 'crosswalk', 'sidewalklen'], help='explanatory variables')
model_arg.add_argument('--init_beta_g', nargs='+', type=float, default=[-0.3, -1.2, 0.1], help='initial parameter values')
model_arg.add_argument('--lb_g', nargs='+', type=float_or_none, default=[None,None,None], help='lower bounds')
model_arg.add_argument('--ub_g', nargs='+', type=float_or_none, default=[0.,0.,None], help='upper bounds')
model_arg.add_argument('--vars_l', nargs='+', type=str, default=['greenlen'], help='explanatory variables')
model_arg.add_argument('--init_beta_l', nargs='+', type=float, default=[0], help='initial parameter values')
model_arg.add_argument('--lb_l', nargs='+', type=float_or_none, default=[None], help='lower bounds')
model_arg.add_argument('--ub_l', nargs='+', type=float_or_none, default=[None], help='upper bounds')
model_arg.add_argument('--estimate_mu', type=str2bool, default=False, help='if estimate mu_g or not')
model_arg.add_argument('--gamma', type=float, default=1., help='discount factor')

# Validation
model_arg.add_argument('--n_samples', type=int, default=1, help='number of samples')
model_arg.add_argument('--test_ratio', type=float, default=0., help='ratio of test samples')
model_arg.add_argument('--isBootstrap', type=str2bool, default=False, help='if bootstrapping or not')

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
    np.random.seed(config.seed)

    ## directories
    # data directory
    data_dir = os.path.join(config.data_dir, config.net_name)

    # output directory
    case_ = 'validation' if config.test_ratio > 0 else 'estimation'
    if config.root is not None:
        out_dir = os.path.join(config.root, "results", config.net_name, case_, config.out_dir)
    else:
        out_dir = os.path.join("results", config.net_name, case_, config.out_dir)
    
    try:
        os.makedirs(out_dir, exist_ok = False)
    except:
        os.makedirs(out_dir + '_' + time.strftime("%Y%m%dT%H%M"), exist_ok = False)
    
    print("Run ", out_dir)

    ## parameters
    # for consistency
    init_beta_l = [0. for _ in range(len(config.vars_l))]
    config.lb_l = [None for _ in range(len(config.vars_l))]
    config.ub_l = [None for _ in range(len(config.vars_l))]
    for l, init_val in enumerate(config.init_beta_l):
        init_beta_l[l] = init_val
    config.init_beta_l = init_beta_l

    if len(config.lb_g) != len(config.vars_g):
        n_lack = len(config.vars_g) - len(config.lb_g)
        if len(config.init_beta_g) < len(config.vars_g):
            config.init_beta_g += [0. for _ in range(n_lack)]
        config.lb_g += [None for _ in range(n_lack)]
        config.ub_g += [None for _ in range(n_lack)]

    ## set timer
    timer = Timer()
    _ = timer.stop()

    ## read network
    link_data = pd.read_csv(os.path.join(data_dir, 'link.csv'))
    node_data = pd.read_csv(os.path.join(data_dir, 'node.csv'))
    obs_data = pd.read_csv(os.path.join(data_dir, 'observations_link.csv'))

    # add negative var of street without sidewalk
    link_data['length'] /= 10
    link_data['carst'] = (link_data['walkwidth2'] == 0) * 1 * (link_data['crosswalk'] == 0) * 1
    link_data['sidewalklen'] = link_data['walkwidth2']/10 * link_data['length']
    link_data['carstlen'] = link_data['carst'] * link_data['length']
    # landscape features
    link_data['greenlen'] = link_data['vegetation'] * link_data['length']
    link_data['skylen'] = link_data['sky'] * link_data['length']
    features = link_data
    # set observation data
    obs_data = reset_index(link_data, node_data, obs_data)
    links = {link_id: (from_, to_) for link_id, from_, to_ in link_data[['link_id', 'from_', 'to_']].values}
    # prepare data for estimation
    dests, obs, obs_filled, n_paths, max_len, od_data, samples = read_mm_results(
        obs_data, links, min_n_paths=config.min_n, n_samples=config.n_samples, test_ratio=config.test_ratio, seed_=111, isBootstrap=config.isBootstrap)

    # number of paths
    print(f"number of paths observed: {n_paths}")
    # loop counts
    n_loops = count_loops(obs_filled)
    for d, n_loop in n_loops.items():
        if n_loop > 0:
            print(f"number of paths including loops observed: {n_loop} for destination {d}")

    ## Define Graph
    g = Graph()
    g.read_data(node_data=node_data, link_data=link_data, od_data=od_data) #features=['length', 'walkwidth', 'green', 'gradient', 'crosswalk', 'walkratio', 'width', 'carst', 'greenlen']

    ## Define variables and parameters in the model
    xs = {}
    var_names = []
    betas = []
    init_beta = []
    # global variables
    for var_, b0, lb, ub in zip(config.vars_g, config.init_beta_g, config.lb_g, config.ub_g):
        var_name = var_ + '_g'
        var_names.append(var_name)
        xs[var_name] = [features[var_].values, 'link', 0]
        init_beta.append(b0)
        betas.append(
            (f'b_{var_name}', b0, lb, ub, var_name, 0)
        )
    # local variables
    for var_, b0, lb, ub in zip(config.vars_l, config.init_beta_l, config.lb_l, config.ub_l):
        var_name = var_ + '_l'
        var_names.append(var_name)
        xs[var_name] = [features[var_].values, 'link', 1]
        init_beta.append(b0)
        betas.append(
            (f'b_{var_name}', b0, lb, ub, var_name, 0)
        )

    # add uturn dummy
    if config.uturn:
        U = (g.senders[:,np.newaxis] == g.receivers[np.newaxis,:]) * (g.receivers[:,np.newaxis] == g.senders[np.newaxis,:])
        U = np.where(U == True)[0]
        uturns = np.zeros_like(g.senders)
        uturns[U] = 1.
        xs['uturn'] = [uturns, 'edge', 0]
        betas.append(('b_uturn', config.uturn_penalty, None, None, 'uturn', 1))

    ## Define model
    rl = RL(g, xs, betas, mu=1., mu_g=config.mu_g, estimate_mu=config.estimate_mu, gamma=config.gamma)

    ### Model Estimation
    # output
    outputs = {i:{} for i in range(config.n_samples)}

    # function for record results
    def record_res(i, res, stderr, t_val, L0, L_val, runtime, init_beta, gamma):
        outputs[i] = {
            'L0': L0,
            'LL': -res.fun,
            'Lv': L_val,
            'runtime': runtime,
            'gamma': gamma
        }
        Vg = len(config.vars_g)
        for var_name, b, s, t, b0 in zip(var_names, res.x, stderr, t_val, init_beta):
            outputs[i].update({
                f'beta_{var_name}': b, f'se_{var_name}': s, f't_{var_name}': t,
                f'b0_{var_name}': b0
            })
        if config.estimate_mu:
            outputs[i].update({
                'mu_g': res.x[-1], 'se_mu_g': stderr[-1], 't_mu_g': (res.x[-1] - 1)/stderr[-1],
            })
        else:
            outputs[i].update({
                'mu_g': rl.mu_g,
            })

    # main estimation phase
    if config.estimate_mu: init_beta += [config.mu_g]
    n_success = 0
    for i, sample in enumerate(samples):
        if n_success >= 100 and config.isBootstrap:
            break

        train_obs = sample['train']
        test_obs = sample['test']

        # only observed destinations in samples
        rl.partitions = list(train_obs.keys())

        # initial parameter and likelihood
        rl.beta = np.array(init_beta)
        # LL0_rl = rl.calc_likelihood(observations=train_obs)
        LL0_rl = rl.eval_init_likelihood(observations=train_obs)
        print('RL model initial log likelihood:', LL0_rl)
        hoge

        try:
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

            # estimation
            Niter = 1
            timer.start()
            # myfactr = 1e-10: for tolerance, add ", 'ftol': 1e-2 * np.finfo(float).eps, 'gtol': 1e-20 * np.finfo(float).eps"
            results_rl = minimize_parallel(f, x0=rl.beta, bounds=rl.bounds,
                    options={'disp':False, 'maxiter':100}, callback=callbackF) #, parallel={'max_workers':4, 'verbose': True}
            rl_time = timer.stop()
            print(f"estimation time is {rl_time}s.")
            rl.beta = results_rl.x
            # DO NOT use this t-value: it should be computed by bootstrapping
            cov_matrix = results_rl.hess_inv if type(results_rl.hess_inv) == np.ndarray else results_rl.hess_inv.todense()
            stderr = np.sqrt(np.diag(cov_matrix))
            t_val = results_rl.x / stderr
            if config.estimate_mu:
                t_val[-1] = (results_rl.x[-1] - 1) / stderr[-1]
            rl.print_results(results_rl, stderr, t_val, LL0_rl)

            # validation
            if config.test_ratio > 0:
                # validation
                rl.partitions = list(test_obs.keys())
                LL_val_rl = rl.calc_likelihood(observations=test_obs)
                print('RL model validation log likelihood:', LL_val_rl)
            else:
                LL_val_rl = 0.

            # record results
            record_res(i, results_rl, stderr, t_val, LL0_rl, LL_val_rl, rl_time, init_beta, rl.gamma)
            n_success += 1
            print(f"RL was successfully estimated for sample {i}, and so far {n_success}; param. values = {rl.beta}")
        except:
            print(f"RL is not feasible for sample {i}; param. values = {rl.beta}")


    ## record results
    df_rl = pd.DataFrame(outputs).T
    print(df_rl)
    model_type = 'RL' if config.gamma == 1 else f'DRL{config.gamma}'
    df_rl.to_csv(os.path.join(out_dir, f"{model_type}.csv"), index=True)    
    
    # write config file
    with open(os.path.join(out_dir, "config.json"), mode="w") as f:
        json.dump(config.__dict__, f, indent=4)