import numpy as np
import pandas as pd
from optimparallel import minimize_parallel
from numdifftools import Hessian
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
model_arg.add_argument('--rl', type=str2bool, default=False, help='if estimate RL or not')
model_arg.add_argument('--prism', type=str2bool, default=True, help='if estimate prism RL or not')
model_arg.add_argument('--parallel', type=str2bool, default=False, help='if implement parallel computation or not')
model_arg.add_argument('--version', type=str, default='test', help='version name')

# Hyperparameters
model_arg.add_argument('--state_key', type=str, default='d', help='od or d')
model_arg.add_argument('--T', type=int, default=15, help='time constraint')
model_arg.add_argument('--uturn', type=str2bool, default=True, help='if add uturn dummy or not')
model_arg.add_argument('--uturn_penalty', type=float, default=-10., help='penalty for uturn')
model_arg.add_argument('--min_n', type=int, default=0, help='minimum number observed for d')

# parameters
model_arg.add_argument('--mu_g', type=float, default=1., help='scale for global utility')
model_arg.add_argument('--vars_g', nargs='+', type=str, default=['length', 'crosswalk'], help='explanatory variables')
model_arg.add_argument('--init_beta_g', nargs='+', type=float, default=[-2.,-2.], help='initial parameter values')
model_arg.add_argument('--lb_g', nargs='+', type=float_or_none, default=[None,None], help='lower bounds')
model_arg.add_argument('--ub_g', nargs='+', type=float_or_none, default=[0.,0.], help='upper bounds')
model_arg.add_argument('--vars_l', nargs='+', type=str, default=['vegetation', 'sky', 'wall'], help='explanatory variables')
model_arg.add_argument('--init_beta_l', nargs='+', type=float, default=[0.1], help='initial parameter values')
model_arg.add_argument('--lb_l', nargs='+', type=float_or_none, default=[None], help='lower bounds')
model_arg.add_argument('--ub_l', nargs='+', type=float_or_none, default=[None], help='upper bounds')
model_arg.add_argument('--estimate_mu', type=str2bool, default=False, help='if estimate mu_g or not')

# Validation
model_arg.add_argument('--n_samples', type=int, default=1, help='number of samples')
model_arg.add_argument('--test_ratio', type=float, default=0, help='ratio of test samples')

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
    link_data['carst'] = (link_data['walkwidth'] == 0) * 1 * (link_data['crosswalk'] == 0) * 1
    link_data['sidewalklen'] = link_data['walkwidth'] * link_data['length']
    link_data['carstlen'] = link_data['carst'] * link_data['length']
    # landscape features
    link_data['greenlen'] = link_data['vegetation'] * link_data['length']
    features = link_data

    # %%
    obs_data = reset_index(link_data, node_data, obs_data)
    links = {link_id: (from_, to_) for link_id, from_, to_ in link_data[['link_id', 'from_', 'to_']].values}

    # %%
    dests, obs, obs_filled, n_paths, max_len, od_data, samples = read_mm_results(
        obs_data, links, min_n_paths=config.min_n, n_samples=config.n_samples, test_ratio=config.test_ratio, seed_=111)

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

    if config.prism:
        # %%
        detour_df = analyze_detour_rate(g, obs)
        # detour_df['detour_rate'].describe()
        # detour_df.plot.scatter('min_step', 'obs_step')

        # %%
        g.define_T_from_obs(detour_df)
        print(f"T = {g.T}")

        # %%
        timer.start()
        g.get_state_networks(method=config.state_key, parallel=True)
        snet_time = timer.stop()
        print(f"time to get snets is {snet_time}s.")

    # %%
    # variables and parameters in the model
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
    if config.rl:
        rl = RL(g, xs, betas, mu=1., mu_g=config.mu_g, estimate_mu=config.estimate_mu)
        models['RL'] = rl
    if config.prism:
        prism = PrismRL(g, xs, betas, mu=1., mu_g=config.mu_g, method=config.state_key, estimate_mu=config.estimate_mu)
        models['PrismRL'] = prism

    ### Model Estimation
    # output
    outputs = {}
    if config.rl: outputs['RL'] = {i:{} for i in range(config.n_samples)}
    if config.prism: outputs['PrismRL'] = {i:{} for i in range(config.n_samples)}
    # function for record results
    def record_res(i, model_type, res, stderr, t_val, L0, L_val, runtime, init_beta):
        outputs[model_type][i] = {
            'L0': L0,
            'LL': -res.fun,
            'Lv': L_val,
            'runtime': runtime,
        }
        Vg = len(config.vars_g)
        for var_name, b, s, t, b0 in zip(var_names, res.x, stderr, t_val, init_beta):
            outputs[model_type][i].update({
                f'beta_{var_name}': b, f'se_{var_name}': s, f't_{var_name}': t,
                f'b0_{var_name}': b0
            })
        if config.estimate_mu:
            outputs[model_type][i].update({
                'mu_g': res.x[-1], 'se_mu_g': stderr[-1], 't_mu_g': (res.x[-1] - 1)/stderr[-1],
            })
        else:
            outputs[model_type][i].update({
                'mu_g': models.get(model_type).mu_g,
            })

    for i, sample in enumerate(samples):
        train_obs = sample['train']
        test_obs = sample['test']
        # if config.min_n > 0:
        #     train_obs = {d:train_obs[d] for d in dests_reduced}
        #     test_obs = {d:test_obs[d] for d in dests_reduced}

        # %%
        if config.rl:
            # fixed mu_g
            if not config.estimate_mu and config.n_samples > 1 and config.test_ratio == 0:
                rl.mu_g = config.mu_g - 0.1 * i

            # %%
            # only observed destinations in samples
            rl.partitions = list(train_obs.keys())

            # %%
            if config.estimate_mu: init_beta += [1.]
            rl.beta = np.array(init_beta)
            LL0_rl = rl.calc_likelihood(observations=train_obs)
            print('RL model initial log likelihood:', LL0_rl)

            # %%
            try:
                # %%
                print(f"RL model estimation for sample {i}...")
                timer.start()
                results_rl = rl.estimate(observations=train_obs, method='L-BFGS-B', disp=False, hess='res')
                rl_time = timer.stop()
                print(f"estimation time is {rl_time}s.")
                rl.print_results(results_rl[0], results_rl[2], results_rl[3], LL0_rl)
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
                record_res(i, 'RL', results_rl[0], results_rl[2], results_rl[3], LL0_rl, LL_val_rl, rl_time, init_beta)
            except:
                print(f"RL is not feasible for sample {i}; param. values = {rl.beta}")

        # %%
        if config.prism:
            # fixed mu_g
            if not config.estimate_mu and config.n_samples > 1 and config.test_ratio == 0:
                prism.mu_g = config.mu_g - 0.1 * i

            # %%
            s_train_obs = prism.translate_observations(train_obs)
            s_test_obs = prism.translate_observations(test_obs)

            # %%
            if config.estimate_mu: init_beta += [1.]
            prism.beta = np.array(init_beta)
            LL0 = prism.calc_likelihood(observations=s_train_obs)
            print('prism model initial log likelihood:', LL0)

            def f(x):
                # prism.beta = x
                # if config.estimate_mu: prism.mu_g = prism.beta[-1]
                # compute probability
                prism.eval_prob(x)
                # calculate log-likelihood
                LL = 0.
                for key_, paths in s_train_obs.items():
                    p = prism.p[key_]
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
            print(f"Prism RL model estimation for sample {i}...")
            print(f"Started with initial value {prism.beta}")
            Niter = 1
            timer.start()
            prism_res = minimize_parallel(f, x0=prism.beta, bounds=prism.bounds, options={'disp':False},
                                            callback=callbackF) #, parallel={'max_workers':4, 'verbose': True}
            prism_time = timer.stop()
            print(f"estimation time is {prism_time}s.")
            # after estimation
            prism.beta = prism_res.x
            cov_matrix = prism_res.hess_inv if type(prism_res.hess_inv) == np.ndarray else prism_res.hess_inv.todense()
            # hess_fn = Hessian(f)(prism_res.x)
            # hess = hess_fn(prism_res.x)
            # cov_matrix = np.linalg.inv(hess)
            stderr = np.sqrt(np.diag(cov_matrix))
            t_val = prism_res.x / stderr
            prism.print_results(prism_res, stderr, t_val, LL0)
            print(prism_res)

            # %%
            # print("Prism RL model estimation...")
            # timer.start()
            # results = prism.estimate(observations=s_train_obs, method='L-BFGS-B', disp=False, hess='res')
            # prism_time = timer.stop()
            # print(f"estimation time is {prism_time}s.")
            # prism.print_results(results[0], results[2], results[3], LL0)

            # %%
            # validation
            if config.test_ratio > 0:
                prism.partitions = list(s_test_obs.keys())
                LL_val = prism.calc_likelihood(observations=s_test_obs)
                print('Prism RL model validation log likelihood:', LL_val)
            else:
                LL_val = 0.

            # %%
            # record results
            record_res(i, 'PrismRL', prism_res, stderr, t_val, LL0, LL_val, prism_time, init_beta)

    if config.rl:
        df_rl = pd.DataFrame(outputs['RL']).T
        print(df_rl)
        if config.test_ratio > 0:
            # for validation
            df_rl.to_csv(f'results/pp_application/{network_}/validation/RL_{config.version}.csv', index=True)
        else:
            # for estimation
            df_rl.T.to_csv(f'results/pp_application/{network_}/estimation/RL_{config.version}.csv', index=True)
    if config.prism:
        df_prism = pd.DataFrame(outputs['PrismRL']).T
        print(df_prism)
        if config.test_ratio > 0:
            # for validation
            df_prism.to_csv(f'results/{network_}/validation/PrismRL_{config.version}.csv', index=True)
        else:
            # for estimation
            df_prism.T.to_csv(f'results/{network_}/estimation/PrismRL_{config.version}.csv', index=True)

    # %%
    # write config file
    dir_ = f'results/pp_application/{network_}/'
    dir_ = dir_ + 'validation/' if config.test_ratio > 0 else dir_ + 'estimation/'
    with open(f"{dir_}{config.version}.json", mode="w") as f:
        json.dump(config.__dict__, f, indent=4)
