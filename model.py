import numpy as np
import multiprocessing as mp
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse import linalg as splinalg
from scipy.optimize import minimize
from numdifftools import Hessian
# from optimparallel import minimize_parallel
# from memory_profiler import profile
import gc

Niter = 1
def callbackF(x):
    global Niter
    txt = f'{Niter: d}'
    for i in range(len(x)): txt += f'\t{x[i]:.4f}'
    print(txt)
    Niter += 1

class RL(object):
    def __init__(self,
                graph,
                features,
                betas,
                mu=1.,
                mu_g=1.,
                parallel=False,
                print_process=False,
                estimate_mu=False
                ):

        # setting
        self.model_type = 'rl'
        self.eps = 1e-8
        self.inf = 1e+10
        self.parallel = parallel
        self.estimate_mu = estimate_mu
        self.print_process = print_process

        # inputs
        self.graph = graph
        self.x = []
        self.beta = []
        self.freebetaNames = []
        self.bounds = []
        self.fixed_v = np.zeros(len(graph.edges), dtype=np.float)
        self.fixed_v_link = np.zeros(len(graph.links), dtype=np.float)
        # self.features = []
        # self.f_idxs = {'g_link': [], 'g_edge': [], 'l_link': [], 'l_edge': []}
        for k, (name, init_val, lower, upper, var_name, to_estimate) in enumerate(betas):
            f, var_type, is_local = features[var_name]
            if to_estimate == 0:
                self.beta.append(init_val)
                self.freebetaNames.append(name)
                self.bounds.append((lower, upper))
                self.x.append(features[var_name])
                # self.features.append(f)
                # if is_local == 0:
                #     self.f_idxs[f'g_{var_type}'].append(k)
                # else:
                #     self.f_idxs[f'l_{var_type}'].append(k)
            else:
                assert is_local == 0, 'fixed parameter for local variable is not yet implemented'
                if var_type == 'link':
                    self.fixed_v += init_val * f[graph.receivers]
                    self.fixed_v_link += init_val * f
                elif var_type == 'edge':
                    self.fixed_v += init_val * f
        # self.features = np.vstack(self.features).T
        # when estimating mu_global
        # this does not effect on eval_z if it is placed at tail (due to zip iteration)
        if self.estimate_mu:
            self.beta.append(mu) #mu_g
            self.freebetaNames.append('mu')
            self.bounds.append((0., None)) # (0., None)
        self.beta = np.array(self.beta, dtype=np.float)
        self.beta_hist = [self.beta]

        # parameters
        self.mu = mu
        self.mu_g = mu_g

        # probtbility
        self.p_pair = {}    # |OD| x E x 1
        self.p = {}         # |OD| x |S| x |S|

        # value function
        self.z = {}

        # partitions: destinations for RL
        self.partitions = graph.dests

    def assign_flows(self, ods):
        """network loading by using probability
        ods (dict): {d:{o:flow}}
        """
        L = len(self.graph.links)
        I = sp.identity(L+1)
        x = np.zeros(L, dtype=np.float)
        for d, qods in ods.items():
            qd = csr_matrix(
                (list(qods.values()), (list(qods.keys()), np.zeros(len(qods)))),
                shape=(L+1, 1)
                )
            # solve the system of linear equations
            xd = splinalg.spsolve((I - self.p[d].T), qd)    # L+1 x 1
            x += xd.toarray().squeeze()
        return x

    def sample_path(self, os, d, sample_size, max_len=False, origin='link'):
        p = self.p[d]
        # init nodes
        seq0 = []
        for o in os:
            if origin == 'node':
                olinks = self.graph.dummy_forward_stars[o]
                seq0 += [np.random.choice(olinks) for _ in range(sample_size)]
            elif origin == 'link':
                seq0 += [o for _ in range(sample_size)]

        # sampling
        seq = [seq0]
        while True:
            states = seq[-1]
            actions = self._random_choice(p[states].toarray())
            seq.append(actions)
            if (actions == len(self.graph.links)).all() or len(seq) == max_len:
                break
        return np.array(seq) # max_len x sample_size

    def _random_choice(self, p_array):
        p_cumsum = np.cumsum(p_array, axis=1)
        udraws = np.random.uniform(size=p_array.shape[0])
        choices = p_array.shape[1] - np.sum(udraws[:,np.newaxis] < p_cumsum, axis=1)
        return choices

    def check_feasibility(self, beta):
        v, _, __, ___, ____ = self._eval_v(beta)
        feasibility = np.array(
            [self._check_feasibility_partition(key_, v)
                for key_ in self.partitions]
        )
        return feasibility.all()

    def _check_feasibility_partition(self, d, v):
        # update v by dummy edges
        L = len(self.graph.links)
        dlinks = self.graph.dummy_backward_stars[d]
        vd = np.concatenate([v, np.zeros(len(dlinks))], axis=0)
        add_edges = [(dlink, L) for dlink in dlinks]
        edges = np.concatenate([self.graph.edges, add_edges], axis=0)
        senders, receivers = edges[:,0], edges[:,1]
        # compute z
        z, _ = self._eval_z(vd, senders, receivers)
        return np.min(z) > 0.

    def get_z(self, d):
        v, _, __, ___, ____ = self._eval_v(self.beta)
        # update v by dummy edges
        L = len(self.graph.links)
        dlinks = self.graph.dummy_backward_stars[d]
        vd = np.concatenate([v, np.zeros(len(dlinks))], axis=0)
        add_edges = [(dlink, L) for dlink in dlinks]
        edges = np.concatenate([self.graph.edges, add_edges], axis=0)
        senders, receivers = edges[:,0], edges[:,1]
        # compute z
        z, _ = self._eval_z(vd, senders, receivers)
        return z

    def eval_prob(self, beta=None):
        if beta is None: beta = self.beta
        assert beta.shape == self.beta.shape, f'betafree shape is not appropriate, it was {beta.shape} but should be {self.beta.shape}!!'
        # take mu_global as a parameter if estimating it
        if self.estimate_mu: self.mu = beta[-1] #self.mu_g = beta[-1]

        v, v_dict, v_link, v_local, v_local_link = self._eval_v(beta)
        if self.parallel:
            n_threads = len(self.partitions)
            pool = mp.Pool(n_threads)
            argsList = [
                [[self.partitions[r]], v, v_dict, v_link, v_local, v_local_link] for r in range(n_threads)
                ]
            probsList = pool.map(self._eval_prob_parallel, argsList)
            pool.close()
            for probs, args in zip(probsList, argsList):
                self.p[args[0][0]] = probs[0][0]
                self.p_pair[args[0][0]] = probs[1][0]
        else:
            for key_ in self.partitions:
                self.p[key_], self.p_pair[key_] = self._eval_prob_partition(key_, v, v_dict, v_link, v_local, v_local_link)

    def _eval_prob_parallel(self, params):
        keys_, v, v_dict, v_link, v_local, v_local_link = params # for multiprocessing
        probs = [[], []]
        for key_ in keys_:
            p, p_pair = self._eval_prob_partition(key_, v, v_dict, v_link, v_local, v_local_link)
            probs[0].append(p)
            probs[1].append(p_pair)
        return probs

    def _eval_prob_partition(self, d, v, v_dict, v_link, v_local, v_local_link):
        # input
        L = len(self.graph.links)

        # update v by dummy edges
        dlinks = self.graph.dummy_backward_stars[d]
        vd = np.concatenate([v, np.zeros(len(dlinks), dtype=np.float)], axis=0)
        vd_local = np.concatenate([v_local, np.zeros(len(dlinks), dtype=np.float)], axis=0)
        add_edges = [(dlink, L) for dlink in dlinks]
        edges = np.concatenate([self.graph.edges, add_edges], axis=0)
        senders, receivers = edges[:,0], edges[:,1]

        # compute z
        z, exp_v = self._eval_z(vd, senders, receivers)
        assert np.min(z) > 0., 'z includes zeros or negative values!!: beta={}, d={}, z={}'.format(self.beta, d, z)
        self.z[d] = z

        # compute the probtbility
        logit = np.exp((vd + vd_local)/self.mu) * (senders != L) *\
                    (z[receivers] ** (self.mu_g/self.mu))
        deno = np.zeros((L+1,), dtype=np.float)
        np.add.at(deno, senders, logit)
        # W = csr_matrix(
        #     (logit, (senders, receivers)), shape=(L+1,L+1)
        # )
        # deno = W.toarray().sum(axis=1)
        p_pair = logit / deno[senders] # L+1 x 1
        # p = np.zeros((L+1, L+1), dtype=np.float)
        # p[senders, receivers] = p_pair
        # p[L,L] = 1.
        p = csr_matrix(
                        (np.append(p_pair, [1.0]),
                            (np.append(senders, [L]), np.append(receivers, [L]))
                        )
                        , shape=(L+1,L+1)) # L+1 x L+1
        return p, p_pair

    def _eval_z(self, vd, senders, receivers):
        # weight matrix of size N x N
        L = len(self.graph.links)
        exp_v = np.exp(vd / self.mu_g) * (senders != L) # E x 1
        M = csr_matrix((exp_v, (senders, receivers)), shape=(L+1,L+1)) # L+1 x L+1
        I = sp.identity(L+1)
        b = csr_matrix(([1.], ([L], [0])), shape=(L+1, 1)) # L+1 x 1

        # solve the system of linear equations
        z = splinalg.spsolve((I - M), b)    # L+1 x 1
        return z, exp_v

    def _eval_z_vi(self, vd, senders, receivers):
        # weight matrix of size N x N
        L = len(self.graph.links)
        exp_v = np.exp(vd / self.mu_g) * (senders != L) # E x 1
        M = csr_matrix((exp_v, (senders, receivers)), shape=(L+1,L+1)) # L+1 x L+1
        b = np.zeros((L+1,), dtype=np.float) # L+1 x 1
        b[L] = 1.
        z = np.ones((L+1,), dtype=np.float) # L+1 x 1
        t = 0
        while True:
            t += 1
            zt = M @ z + b
            dif = np.linalg.norm(zt - z)
            if dif < 1e-20 or t == 100:
                z = zt
                break
            z = zt
        return z, exp_v

    def _eval_v(self, beta):
        # # weight vectors: g_link, g_edge, l_link, l_edge
        # if len(self.f_idxs['g_link']) > 0:
        #     v = self.fixed_v + self.features[:,self.f_idxs['g_link']][self.graph.receivers] @ beta[self.f_idxs['g_link']]
        #     v_link = self.fixed_v_link + self.features[:,self.f_idxs['g_link']] @ beta[self.f_idxs['g_link']]
        # if len(self.f_idxs['g_edge']) > 0:
        #     v += self.features[:,self.f_idxs['g_edge']] @ beta[self.f_idxs['g_edge']]
        # if len(self.f_idxs['l_link']) > 0:
        #     v_local = self.features[:,self.f_idxs['l_link']][self.graph.receivers] @ beta[self.f_idxs['l_link']]
        #     v_local_link = self.features[:,self.f_idxs['l_link']] @ beta[self.f_idxs['l_link']]
        # if len(self.f_idxs['l_edge']) > 0:
        #     v_local += self.features[:,self.f_idxs['l_edge']] @ beta[self.f_idxs['l_edge']]

        v = self.fixed_v.copy()
        v_link = self.fixed_v_link.copy()
        v_local = np.zeros_like(self.fixed_v)
        v_local_link = np.zeros_like(self.fixed_v_link)
        for b, (f, var_type, is_local) in zip(beta, self.x):
            if is_local == 0:
                if var_type == 'link':
                    v += b * f[self.graph.receivers]
                    v_link += b * f
                elif var_type == 'edge':
                    v += b * f
            elif is_local == 1:
                if var_type == 'link':
                    v_local += b * f[self.graph.receivers]
                    v_local_link += b * f
                elif var_type == 'edge':
                    v_local += b * f

        # convert it to dictionary
        v_dict = {tuple(edge): ve for edge, ve in zip(self.graph.edges, v)}
        return v, v_dict, v_link, v_local, v_local_link

    def calc_likelihood(self, observations, beta=None):
        if beta is None: beta = self.beta
        self.beta = beta
        # curr_beta = ''
        # for b in beta: curr_beta += str(b) + ','
        # print(f'current beta: {curr_beta}')

        # obtain probability with beta
        if self.print_process: print('Computing probabilities...')
        self.eval_prob(beta)

        # calculate log-likelihood
        if self.print_process: print('Evaluating likelihood...')
        LL = 0.
        # Do not use parallel computing, which is too slow for this:
        for key_, paths in observations.items():
            p = self.p[key_]
            max_len, N = paths.shape
            Lk = np.zeros(N, dtype=np.float)
            for j in range(max_len - 1):
                L = np.array(p[paths[j], paths[j+1]])[0]
                assert (L > 0 ).all(), f'L includes zeros: key_={key_}, j={j}, pathj={paths[j]}, pathj+1={paths[j+1]}'
                Lk += np.log(L)
            LL += np.sum(Lk)

        return LL

    def _likelihood_partition(self, obs):
        key_, paths = obs
        p = self.p[key_]
        max_len, N = paths.shape
        Lk = np.zeros(N, dtype=np.float)
        for j in range(max_len - 1):
            L = np.array(p[paths[j], paths[j+1]])[0]
            assert (L > 0 ).all(), f'L includes zeros: key_={key_}, j={j}, pathj={paths[j]}, pathj+1={paths[j+1]}'
            Lk += np.log(L)
        return np.sum(Lk)

    def estimate(self, observations, init_beta=None, method='L-BFGS-B', disp=False, hess='numdif'):
        global Niter
        Niter = 1
        if init_beta is None: init_beta = self.beta
        # print initial values
        header, txt = 'Niter', '0'
        for i, b in enumerate(init_beta):
            header += f'\tx{i}'
            txt += f'\t{b:.4f}'
        print(header+'\n', txt)

        # negative log-likelihood function
        f = lambda x: -self.calc_likelihood(observations, x)
        res = minimize(f, x0=init_beta, method=method, bounds=self.bounds, options={'disp': disp}, callback=callbackF)
        # res = minimize_parallel(f, x0=init_beta, bounds=self.bounds, options={'disp': disp}, callback=callbackF)
        # print(res)

        # stats using numdifftools
        if hess == 'numdif':
            hess_fn = Hessian(f)
            hess = hess_fn(res.x)
            cov_matrix = np.linalg.inv(hess)
        else:
            cov_matrix = res.hess_inv if type(res.hess_inv) == np.ndarray else res.hess_inv.todense()
        stderr = np.sqrt(np.diag(cov_matrix))
        t_val = res.x / stderr

        return res, res.x, stderr, t_val, cov_matrix, self.freebetaNames

    def update_var(self, idxs, new_fs):
        # use the index of freebeta variables list
        only_local = 1
        for i, new_f in zip(idxs, new_fs):
            f, var_type, is_local = self.x[i]
            self.x[i] = (new_f, var_type, is_local)
            only_local *= is_local

        # update value function if global var
        if not only_local:
            self.eval_prob()

        # update only prob if local var
        if only_local:
            L = len(self.graph.links)
            v, _, __, v_local, ____ = self._eval_v(self.beta)

            for d in self.partitions:
                # update v by dummy edges
                dlinks = self.graph.dummy_backward_stars[d]
                vd = np.concatenate([v, np.zeros(len(dlinks))], axis=0)
                vd_local = np.concatenate([v_local, np.zeros(len(dlinks))], axis=0)
                add_edges = [(dlink, L) for dlink in dlinks]
                edges = np.concatenate([self.graph.edges, add_edges], axis=0)
                senders, receivers = edges[:,0], edges[:,1]

                # compute the probtbility
                z = self.z[d]
                logit = np.exp((vd + vd_local)/self.mu) * (senders != L) *\
                            (z[receivers] ** (self.mu_g/self.mu))
                W = csr_matrix(
                    (logit, (senders, receivers)), shape=(L+1,L+1)
                )
                deno = W.toarray().sum(axis=1)
                p_pair = logit / deno[senders] # L+1 x 1
                p = csr_matrix(
                                (np.append(p_pair, [1.0]),
                                    (np.append(senders, [L]), np.append(receivers, [L]))
                                )
                                , shape=(L+1,L+1)) # L+1 x L+1
                self.p_pair[d] = p
                self.p[d] = p

    def confidence_intervals(self, beta, cov_matrix, n_draws=100):
        """Krinsky and Robb (1986) method
        see e.g. Bliemer and Rose (2013) for the details
        """
        L = np.linalg.cholesky(cov_matrix)
        r = np.random.normal(size=(len(beta), n_draws))
        z = beta[:,np.newaxis] + L.T.dot(r)
        lowers, uppers = np.percentile(z, [2.5, 97.5], axis=1)
        means = np.mean(z, axis=1)
        # stderr = np.sqrt(np.var(z, axis=1))
        return z, lowers, uppers, means

    def print_results(self, res, stderr, t_val, L0):
        print('{0:9s}   {1:9s}   {2:9s}   {3:9s}'.format('Parameter', ' Estimate', ' Std.err', ' t-stat'))
        # print('Parameter\tEstimate\tStd.Err.\tt-stat.')
        for name, b, s, t in zip(self.freebetaNames, res.x, stderr, t_val):
            print('{0:9s}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}'.format(name, b, s, t))
            # print(f'{name}\t{b:.3f}\t{s:.3f}\t{t:.2f}')
        print(f'Initial log likelihood: {L0:.3f}')
        print(f'Final log likelihood: {-res.fun:.3f}')
        print(f'Adjusted rho-squared: {1-(-res.fun-len(res.x))/(L0):.2f}')
        print(f'AIC: {2*res.fun + 2*len(res.x):.3f}')

class PrismRL(RL):

    def __init__(self,
                graph,
                features,
                betas,
                mu=1.,
                mu_g=1.,
                parallel=False,
                print_process=False,
                method='d',
                estimate_mu=False
                ):

        super().__init__(graph, features, betas, mu, mu_g, parallel, print_process, estimate_mu)
        self.model_type = 'prism'

        # od or d specific model
        self.method = method
        self.partitions = graph.ods if method == 'od' else graph.dests
        self.sample_path = self.sample_path_od if method == 'od' else self.sample_path_d

        # inputs
        self.T = graph.T

    def translate_observations(self, observations):
        # translate static routes into state paths
        s_observations = {}
        for key_, paths in observations.items():
            # state network
            net = self.graph.state_networks[key_]
            init_idx, fin_idx = net['init_idx'], net['fin_idx']
            T, d = net['states'][fin_idx] # T, dlink_idx
            states_idx = net['states_idx']
            # translation
            max_len, N = paths.shape
            RLd = paths[-1,-1]
            newpaths = paths.copy()
            # if init_idx is not None:
            #     init_path = np.ones(shape=(1,N), dtype=np.int) * init_idx
            #     newpaths = np.concatenate([init_path, newpaths], axis=0)
            for n in range(N):
                # don't get why but it doesn't go well with for loop of t
                t, newt = 0, 0
                if init_idx is not None: newt += 1
                while True:
                    if paths[t,n] == RLd:
                        newpaths[t,n] = states_idx[(newt, d)]
                        newpaths[t+1:,n] = fin_idx
                        t = T - 1
                    else:
                        newpaths[t,n] = states_idx[(newt, paths[t,n])]
                    t += 1
                    newt += 1
                    if t == T:
                        if newpaths[:,n].shape[0] == T+1: newpaths[t,n] = fin_idx
                        break
            s_observations[key_] = newpaths
        return s_observations

    def sample_path_od(self, o, d, sample_size, max_len=None):
        if max_len is None: max_len = self.T

        # read od_data
        s_net = self.graph.state_networks[(o,d)]
        idx_mtrx = s_net['trans_idx']
        init_idx, fin_idx = s_net['init_idx'], s_net['fin_idx']

        # sampling
        p = self.p[(o,d)]
        seq = [[init_idx for _ in range(sample_size)]]
        # seq_pair = []
        # sampling
        while True:
            states = seq[-1]
            actions = self._random_choice(p[states].toarray())
            seq.append(actions)
            # seq_pair.append(
            #     np.array(idx_mtrx[states, actions])[0]
            # )
            if (actions == fin_idx).all() or len(seq) == max_len:
                break
        return np.array(seq) #, np.array(seq_pair) # max_len x sample_size

    def sample_path_d(self, os, d, sample_size, max_len=None):
        if max_len is None: max_len = self.T

        # read od_data
        s_net = self.graph.state_networks[d]
        idx_mtrx = s_net['trans_idx']
        fin_idx  = s_net['fin_idx']
        p = self.p[d]

        # init states
        seq0 = []
        for o in os:
            init_idx = s_net['states_idx'][(0,o)]
            seq_o = [init_idx for _ in range(sample_size)]
            seq0 += seq_o

        # sampling
        seq = [seq0]
        while True:
            states = seq[-1]
            actions = self._random_choice(p[states].toarray())
            seq.append(actions)
            if (actions == fin_idx).all() or len(seq) == max_len:
                break
        return np.array(seq) # max_len x (sample_size x O)

    def sample_static_path_d(self, os, d, sample_size, max_len=None):
        if max_len is None: max_len = self.T

        # read od_data
        s_net = self.graph.state_networks[d]
        idx_mtrx = s_net['trans_idx']
        fin_idx  = s_net['fin_idx']
        states = s_net['states']
        p = self.p[d]

        # init states
        seq0 = []
        static_seq0 = []
        for o in os:
            init_idx = s_net['states_idx'][(0,o)]
            seq_o = [init_idx for _ in range(sample_size)]
            seq0 += seq_o
            static_seq0 += [o for _ in range(sample_size)]

        # sampling
        cur_states = seq0
        static_seq = [static_seq0]
        while True:
            actions = self._random_choice(p[cur_states].toarray())
            links = [states[action][1] for action in actions]
            static_seq.append(links)
            if (actions == fin_idx).all() or len(static_seq) == max_len:
                break
            cur_states = actions
        return np.array(static_seq) # max_len x (sample_size x O)

    # @profile()
    def _eval_prob_partition(self, key_, v, v_dict, v_link, v_local, v_local_link):
        # inputs
        net = self.graph.state_networks[key_]
        state_space = net['state_space'] # dictionary with key: time, value: s_t
        states = net['states'] # list of all states
        states_idx = net['states_idx']
        static_edges = net['static_edges'] # len: number of state pairs
        fin_idx = net['fin_idx']
        init_idx = net['init_idx']
        BS = net['backward_stars']

        T, d = states[fin_idx]
        S = len(states)
        E = len(static_edges)

        ## update utilities
        v_rev, vdict_rev, vlocal_rev = self._modify_v(v, v_dict, v_link, v_local, v_local_link, net)

        ## compute value functions
        # initialize them
        z = np.zeros(len(states), dtype=np.float) # S x 1
        # update for d states
        d_senders, d_receivers = [], [] # to add transitions from (t, d) to (T, d) with prob 1
        for t in range(0,T+1):
            if (t,d) in states:
                i = states_idx[(t,d)]
                z[i] = 1. # this is needed because backward_stars of d does not contain d
                d_senders.append(i)
                d_receivers.append(fin_idx)
        Ed = len(d_senders)
        s_senders = list(net['s_senders']) + d_senders # E + Ed
        s_receivers = list(net['s_receivers']) + d_receivers # E + Ed

        # backward computation
        t = T
        while True:
            st = state_space[t]
            for a in st:
                j = states_idx[(t, a)]
                in_states = BS[a]
                for k in in_states:
                    if k in state_space[t-1]:
                        i = states_idx[(t-1, k)]
                        z[i] += np.exp(vdict_rev[(k, a)]/self.mu_g) * z[j]
            t -= 1
            if t == 0:
                break

        assert np.min(z[s_senders]) > 0., 'z includes zeros or negative values!!: key_={}, z={}, senders={}'.format(key_, z[s_senders][:10], s_senders[:10])

        ## compute probabilities
        exp_v = np.exp((v_rev[static_edges] + vlocal_rev[static_edges])/self.mu) # E x 1
        exp_v = np.concatenate([exp_v, np.ones(Ed)]) # (E+Ed) x 1
        logit = exp_v * (z[s_receivers]**(self.mu_g/self.mu))
        W = csr_matrix(
            (logit, (s_senders, s_receivers)), shape=(S,S)
        ) # S x S
        deno = W.toarray().sum(axis=1) # S x 1
        p_pair = logit / deno[s_senders] # (E+Ed) x 1
        p = csr_matrix(
            (p_pair, (s_senders, s_receivers)), shape=(S, S)
        ) # S x S

        # free memory: otherwise, run out of memory
        del z, W, exp_v, logit, deno, s_senders, s_receivers
        gc.collect()

        return p, p_pair

    def _modify_v(self, v, v_dict, v_link, v_local, v_local_link, s_net):
        dummy_edges = s_net['dummy_static_edges']
        if s_net['init_idx'] is not None:
            # state network
            _, o = s_net['states'][s_net['init_idx']]
            _, d = s_net['states'][s_net['fin_idx']]
            # update v
            dummy_v, dummy_v_local = [], []
            for s, r in dummy_edges:
                if s == o and r != d:
                    dummy_v.append(v_link[r]) ## this is not correct!! v takes edge index as key.
                    dummy_v_local.append(v_local_link[r]) ## this is not correct!! v takes edge index as key.
                else:
                    dummy_v.append(0)
        else:
            dummy_v = np.zeros(len(dummy_edges), dtype=np.float)
            dummy_v_local = np.zeros(len(dummy_edges), dtype=np.float)
        v_rev = np.concatenate([v, dummy_v], axis=0)
        vlocal_rev = np.concatenate([v_local, dummy_v_local], axis=0)
        dummy_vdict = {e:x for e, x in zip(dummy_edges, dummy_v)}
        vdict_rev = v_dict.copy()
        vdict_rev.update(dummy_vdict)
        return v_rev, vdict_rev, vlocal_rev
