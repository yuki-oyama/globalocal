import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
from copy import deepcopy
from tqdm import tqdm
import multiprocessing as mp

class Graph(object):

    def __init__(self):
        self.T = None
        self.ods = None
        pass

    def read_data(self, node_data, link_data, features=[], od_data=None):
        """
        Arguments:
            node_data: pandas dataframe ('node_id', 'x', 'y')
            link_data: pandas dataframe ('link_id', 'from_', 'to_', 'length', ...)
            *** link id (sender, receiver) should be consistent with index ***
        """

        # node data
        self.nodes = node_data['node_id'].values # N,
        self.pos = np.vstack([node_data['x'].values, node_data['y'].values]).T # N,2

        # link data
        self.links = link_data.index.values # L,
        self.L = len(self.links)
        # self.links = np.append(links, [self.L, self.L+1]) # L + 2: for o and d
        link_nodes = np.vstack([link_data['from_'].values, link_data['to_'].values]).T # L,2

        # edge (link pair) data
        self.edges = []
        for k, (ki, kj) in enumerate(link_nodes):
            As = np.where(kj == link_nodes[:,0])[0]
            for a in As: self.edges.append((k,a))
        self.edges = np.array(self.edges)
        self.senders = self.edges[:,0]
        self.receivers = self.edges[:,1]
        # get forward/backward stars
        self.get_stars()

        # link features
        self.link_features = {}
        for f in features:
            self.link_features[f] = link_data[f].values

        # edge (pair) features
        self.edge_features = {}

        # od data
        if od_data is not None:
            self.origins = od_data['origin'].unique()
            self.dests = od_data['destination'].unique()
            self.ods = od_data[['origin', 'destination']].values

        # dummy edges to be added
        self.dummy_forward_stars = {} # for origin nodes
        self.dummy_backward_stars = {} # for dest nodes
        for o in self.origins:
            olinks = np.where(link_nodes[:,0] == o)[0]
            self.dummy_forward_stars[o] = olinks
        for d in self.dests:
            dlinks = np.where(link_nodes[:,1] == d)[0]
            self.dummy_backward_stars[d] = dlinks

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    def define_T_from_obs(self, detour_df, range=1.34):
        T = {} # dict for each destination
        for d in self.dests:
            df = detour_df.query(f'destination == {d}')
            if len(df) > 1:
                Td = int(df['obs_step'].max())
            else:
                Td = 0
                for min_step, obs_step in df[['min_step', 'obs_step']].values:
                    cand = max(min_step * range, obs_step)
                    if Td < cand: Td = int(cand)
            T[d] = Td
        self.T = T

    def set_dummy_edges(self, o=None, d=None):
        # additions
        links = self.links.copy()
        forward_stars = deepcopy(self.forward_stars) # don't know why but, for this, dictionary will be updated when using .copy()
        backward_stars = deepcopy(self.backward_stars) # don't know why but, for this, dictionary will be updated when using .copy()

        dummy_edges = []
        olink, dlink = None, None
        if o is not None:
            olink = len(links)
            links = np.append(links, [olink])
            backward_stars[olink] = []
            forward_stars[olink] = self.dummy_forward_stars[o]
            for a in self.dummy_forward_stars[o]:
                backward_stars[a].append(olink)
                dummy_edges.append((olink, a))
        if d is not None:
            dlink = len(links)
            links = np.append(links, [dlink])
            forward_stars[dlink] = []
            backward_stars[dlink] = self.dummy_backward_stars[d]
            for k in self.dummy_backward_stars[d]:
                forward_stars[k].append(dlink)
                dummy_edges.append((k, dlink))
        edges = np.concatenate(
            [self.edges, dummy_edges], axis=0
        )
        return links, edges, forward_stars, backward_stars, olink, dlink, dummy_edges

    def get_stars(self):
        self.forward_stars = {i:[] for i in self.links}
        self.backward_stars = {j:[] for j in self.links}
        for sender, receiver in zip(self.senders, self.receivers):
            self.forward_stars[sender].append(receiver)
            self.backward_stars[receiver].append(sender)

    def _compute_minimum_steps(self, o=None, d=None, return_path=False):
        # set dummy edges
        newnet = self.set_dummy_edges(o=o, d=d)
        links, edges, forward_stars, backward_stars, olink, dlink, dummy_edges = newnet

        ## compute minimum steps using network x
        # define graph
        G = nx.DiGraph()
        G.add_nodes_from(links)
        G.add_edges_from(edges)

        # computation
        Do, Dd = None, None

        if o is not None:
            # minimum steps from origin link
            min_step_from_o = nx.shortest_path_length(G, source=olink)
            min_step_from_o = dict(sorted(min_step_from_o.items(), key=lambda x:x[0]))
            # convert to arrays Do, Dd
            Do = np.array(list(min_step_from_o.values()))

        if d is not None:
            # minimum steps to dest link
            G_rev = G.reverse(copy=False)
            min_step_to_d   = nx.shortest_path_length(G_rev, source=dlink)
            min_step_to_d   = dict(sorted(min_step_to_d.items(), key=lambda x:x[0]))
            # convert to arrays Do, Dd
            Dd = np.array(list(min_step_to_d.values()))
            if return_path:
                # obtain shortest paths
                path_d = nx.shortest_path(G_rev, source=dlink)
                for k, v in path_d.items(): v.reverse()
                return Do, Dd, newnet, path_d

        return Do, Dd, newnet