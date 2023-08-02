import numpy as np


link = [0,1,2,3,4]
edge = [[0,1],[0,2],[1,3],[1,4],[2,4],[3,4]]
link = np.array(link)
edge = np.array(edge)
L = link.shape[0]
E = edge.shape[0]
cost = np.ones_like(edge.shape[0], dtype=np.float)
c_sum = np.zeros_like(link)
np.add.at(c_sum, edge[:,0], cost)
c_sum

c_sum[edge[:,0]] += c_sum[edge[:,0], edge[:,1]]

c_sum[edge[:,0]].shape
c_sum[edge].shape
c_sum[edge]
c_sum
