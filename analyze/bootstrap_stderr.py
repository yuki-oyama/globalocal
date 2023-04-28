import pandas as pd
import numpy as np

# %%
file_path = 'results/kannai/estimation/DRL0.99_Local_w_mu_greenlen_20230428T1515.csv'
df = pd.read_csv(file_path, index_col=0)
df = df.T
df = df.dropna(how='any', axis=0)
print(len(df))
beta_names_g = []
beta_names_l = []
for vname in df.columns.values:
    if vname[:5] == 'beta_':
        if vname[-2:] == '_g':
            beta_names_g.append(vname)
        elif vname[-2:] == '_l':
            beta_names_l.append(vname)

# %%
stderrs = {}
for bname in beta_names_g + beta_names_l:
    betas = df[bname].values[:100]
    bmean = betas.mean()
    bvar = (betas - bmean)**2/(len(betas)-1)
    stderr = np.sqrt(bvar.sum())
    stderrs[bname] = stderr
    print(bname, betas[0], stderr, betas[0]/stderr)

# stats = df[bname].agg(['mean', 'sem'])
# stats['std'] = df[bname].std(ddof=1)
# stats['ci95_hi'] = stats['mean'] + 1.96* stats['std']
# stats['ci95_lo'] = stats['mean'] - 1.96* stats['std']
# df[bname].quantile(0.95)
# df[bname].quantile(0.05)


# %%
bname = 'mu_g'
betas = df[bname].values[:100]
bmean = betas.mean()
bvar = (betas - bmean)**2/(len(betas)-1)
stderr = np.sqrt(bvar.sum())
stderrs[bname] = stderr
if stderr != 0:
    print(bname, betas[0], stderr, (betas[0]-1)/stderr)

# %%
stderrs
