import numpy as np
import multiprocessing as mp
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse import linalg as splinalg

# %%
L = 50
edges = np.vstack([np.random.choice(np.arange(L+1), 2, replace=False) for _ in range(1000)])
senders, receivers = edges[:,0], edges[:,1]
exp_v = np.ones(edges.shape[0], np.float)
M = csr_matrix((exp_v, (senders, receivers)), shape=(L+1,L+1)) # L+1 x L+1
# M[receivers[5],L] = 1.
I = sp.identity(L+1)
b = csr_matrix(([1.], ([L], [0])), shape=(L+1, 1)) # L+1 x 1
b = np.zeros((L+1,), dtype=np.float) # L+1 x 1
b[L] = 1.
z = np.ones((L+1,), dtype=np.float) # L+1 x 1
zt = M @ z + b
zt.shape

np.linalg.norm(zt - z)
z = zt
zt.toarray()
np.squeeze(z.toarray())
