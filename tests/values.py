import numpy as np

def v4(mu_g):
    return np.log(np.exp(-2*mu_g) + np.exp(-2*mu_g))/mu_g

mugs = 1 + np.arange(60)*0.25
v4(mugs)

mugs = 0.01 + np.arange(50)*0.02
v4(mugs)
