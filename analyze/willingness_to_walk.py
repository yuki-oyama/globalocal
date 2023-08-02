import pandas as pd
import numpy as np

# %%
dir_ = 'results/kannai/estimation/20230515_mu_rev/'
file_path = dir_ + 'Local_w_mu_20230515T1744.csv'
df = pd.read_csv(file_path, index_col=0)
df = df.T
df = df.dropna(how='any', axis=0)
beta_names_g = []
beta_names_l = []
for vname in df.columns.values:
    if vname[:5] == 'beta_':
        if vname[-2:] == '_g':
            beta_names_g.append(vname)
        elif vname[-2:] == '_l':
            beta_names_l.append(vname)

# %%
print(beta_names_g)
print(beta_names_l)

# %%
WTP_keys = []
if 'beta_crosswalk_g' in beta_names_g:
    df['WTP_crosswalk_once'] = df['beta_crosswalk_g'] / df['beta_length_g']
    WTP_keys.append('WTP_crosswalk_once')
if 'beta_sidewalklen_g' in beta_names_g:
    df['WTP_sidewalk_1m'] = (df['beta_sidewalklen_g']*10) / df['beta_length_g']
    WTP_keys.append('WTP_sidewalk_1m')
if 'beta_greenlen_g' in beta_names_g:
    df['WTP_green_10p_global'] = (df['beta_greenlen_g']*10) / df['beta_length_g']
    WTP_keys.append('WTP_green_10p_global')
if 'beta_greenlen_l' in beta_names_l:
    df['WTP_green_10p_local'] = (df['beta_greenlen_l']*10) / df['beta_length_g']
    WTP_keys.append('WTP_green_10p_local')
    df['WTP_green_10p_local_scaled'] = (df['mu_g'] * df['beta_greenlen_l']*10) / df['beta_length_g']
    WTP_keys.append('WTP_green_10p_local_scaled')


# %%
stats = df[WTP_keys].describe().T
stats['CI95_hi'] = df[WTP_keys].quantile(0.95)
stats['CI95_lo'] = df[WTP_keys].quantile(0.05)

# %%
print(stats.T)
