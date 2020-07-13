# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: dev
#     language: python
#     name: dev
# ---

# +
import os
import joblib
import datetime
from re import search

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
# -

import pandas as pd
import numpy as np

from sklearn import set_config
set_config(display='diagram')

# %load_ext watermark

from scipy.special import logit

import lifelines
from lifelines import KaplanMeierFitter, NelsonAalenFitter

# +
import pymc3 as pm
import arviz as az

pm.__version__, az.__version__
# -

import geopandas as gpd

import statsmodels.api as sm

# +
# %matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
# -

sns.set()
plt.rcParams.update({
    "font.family": "Source Sans Pro",
    "font.serif": ["Source Sans Pro"],  # use latex default serif font
    "font.sans-serif": ["Source Sans Pro"],  # use a specific sans-serif font
    "font.size": 10,
})

external_files_dir = "/home/gsinha/admin/db/dev/Python/projects/models/data/"

# %run "/home/gsinha/admin/db/dev/Python/projects/models/defers/common.py"

from numpy.random import RandomState
RANDOM_SEED = 12345
random_state = np.random.RandomState(RANDOM_SEED)

states = states_df["state"].to_list()

fname = external_files_dir + "claims.pkl"
with open(fname, "rb") as f:
    claims_dict = joblib.load(f)

# +
# %%time

fred_fname = external_files_dir + "fred_data"
with open(fred_fname + ".pkl", "rb") as f:
    fred_dict = joblib.load(f)
    ic_df = fred_dict["ic_df"]
    fred_df = fred_dict["fred_df"]
    ic_date = fred_dict["ic_date"]
    w_52_pct_chg_df = fred_dict["w_52_pct_chg_df"]
# -

ic_long_df = pd.melt(w_52_pct_chg_df.reset_index().rename(
    columns={"index": "edate"}
), id_vars="edate", var_name="state", value_name="pct_ic")

# +
LAG_DAYS = 12

now = datetime.datetime.now()
today6pm = now.replace(hour=18, minute=0, second=0, microsecond=0)
ASOF_DATE = min(
    datetime.datetime(ic_date.year, ic_date.month, ic_date.day) + pd.Timedelta(LAG_DAYS, "D"),
    now if now > today6pm else now - datetime.timedelta(days=1)
).date()

override_asof_date = False
if override_asof_date:
    ASOF_DATE = datetime.date(2020, 7, 2)

print(f'As Of Date: {ASOF_DATE}')

# +
ORIGINATOR = None

omap = {"LC": "I", "PR": "II", "ALL": None}

base_dir = "/home/gsinha/admin/db/dev/Python/projects/models/"
results_dir = {
  "LC": base_dir + "defers/pymc3/" + "originator_" + omap["LC"] + "/results",
  "PR": base_dir + "defers/pymc3/" + "originator_" + omap["PR"] + "/results",
  "ALL": base_dir + "defers/pymc3/" + "results/"
}

numeric_features = [
    "fico", "original_balance", "dti", "stated_monthly_income", "age", "pct_ic"
]
std_numeric_features = ["std_" + x for x in numeric_features]

categorical_features = [
    "grade", "purpose", "employment_status", "term", "home_ownership", "loanstatus",
]

group_features = ["st_code", "originator"]
group_type = "crossed"

dep_var = "defer"

# +
# %%time

anonymize = True

df = []
for i in ["PR", "LC"]:
    df.append(make_covid_df(i, ASOF_DATE, anonymize))
hard_df = pd.concat(df, sort=False, ignore_index=True)
hard_df.drop_duplicates(subset=["loan_id"], keep="first", inplace=True, ignore_index=True)
# -

hard_df_train = hard_df.groupby(['state', 'originator', dep_var], group_keys=False).apply(
    lambda x: x.sample(frac=0.80, random_state=random_state)
).reset_index(drop=True).copy()
hard_df_test = hard_df[~hard_df["loan_id"].isin(hard_df_train["loan_id"])].reset_index(drop=True).copy()

hard_df_train.shape, hard_df_test.shape, hard_df.shape

# +
# %%time

p_s_1 = Pipeline(
    steps=[
        ('select_originator', SelectOriginator(ORIGINATOR)),
        ('winsorize', Winsorize(["stated_monthly_income"], p=0.01)),
        ('wide_to_long', WideToLong(id_col="note_id", duration_col="dur", event_col=dep_var)),
        ('add_state_macro_vars', AddStateMacroVars(ic_long_df)),
    ]
)
s_1_df = p_s_1.fit_transform(hard_df_train)
p_s_1

# +
# %%time

p_s_2 = Pipeline(
    steps=[
        ('hier_index', HierarchicalIndex(group_features, group_type)),
        ('standardize', Standardize(numeric_features)),
        ('poly', OrthoPolyFeatures(std_numeric_features[:4], degree=3)),
        ("interval", IntervalInterceptFeatures()),
    ]
)
s_2_df = p_s_2.fit_transform(s_1_df)
p_s_2

# +
# %%time

p_s_3 = Pipeline(
    steps=[
        ('dummy', Dummy(categorical_features)),
    ]
)
s_3_df = p_s_3.fit_transform(s_2_df)
p_s_3
# -

generic = True
if generic:
    obs_covars = [
        x for x in s_3_df.columns if search("T.", x) and not search("grade", x)
    ] + np.array(p_s_2.named_steps.poly.colnames).ravel().tolist() + [std_numeric_features[-2]]
else:
    obs_covars = (
        [x for x in s_3_df.columns if search("T.", x)] + 
        np.array(p_s_2.named_steps.poly.colnames).ravel().tolist() + [std_numeric_features[-2]]
    )

# +
exp_covars = "exposure"
pop_covars = "std_pct_ic"

n_intervals = p_s_2.named_steps.interval.n_intervals

# +
y = s_3_df[dep_var].astype(int).values
X = s_3_df[obs_covars + [pop_covars]].values
E = s_3_df[exp_covars].values

A = s_3_df["a_indx"].values
# -

print(
    f'Wide format records: {hard_df.shape[0]:>6,.0f}\n'
    f'Wide format features: {hard_df.shape[1]:>3,.0f}\n'
    f'Long format records: {s_3_df.shape[0]:>6,.0f}\n'
    f'Long format features {s_3_df.shape[1]:>3,.0f}'
)

# ## Pooled 

with pm.Model() as pooled:
    μ_a = pm.Normal("μ_a", logit(y.mean()), sigma=0.35)
    a = hierarchical_normal("a", μ=μ_a, sigma=0.20, shape=n_intervals)
    
    μ_b = pm.Normal("μ_b", mu=0, sigma=0.35)
    b = hierarchical_normal("b", μ=μ_b, sigma=0.20, shape=X.shape[1])
    
    xbeta = a[A] + pm.math.dot(X, b)
    yobs = pm.Poisson("yobs", mu=pm.math.exp(xbeta) * E, observed=y)

pooled.check_test_point()

# +
prior = pm.sample_prior_predictive(model=pooled, random_seed=RANDOM_SEED)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.distplot([x for x in prior["yobs"].mean(axis=1)], ax=ax)
ax.axvline(y.mean(), color="tab:red", label="Sample μ")
ax.axvline(np.array([x for x in prior["yobs"].mean(axis=1)]).mean(), color="tab:blue", label="Prior μ")
ax.legend()
# -

n_draws = 1000
n_tune = 1000

with pooled:
    pooled_trace = pm.sample(
        n_draws, tune=n_tune, random_seed=RANDOM_SEED,
        target_accept=0.99,
    )

fname, pooled_out_dict = save_results(
    ORIGINATOR, "pooled", ASOF_DATE, pooled, pooled_trace, 
    hard_df, hard_df_train, hard_df_test, s_3_df, 
    p_s_1, p_s_2, p_s_3, numeric_features, categorical_features,
    group_features, group_type, dep_var, pop_covars,
    exp_covars, obs_covars, None
)
with open("results/" + fname, "wb") as f:
    joblib.dump(pooled_out_dict, f)

# ## Hierarchical

# +
y = s_3_df[dep_var].astype(int).values
X = s_3_df[obs_covars].values
U = s_3_df[pop_covars].values
E = s_3_df[exp_covars].values

A = s_3_df["a_indx"].values
O = s_3_df["level_1"].values

frailty = True
# -

if group_type == "nested":
    state_indexes = p_s_2.named_steps.hier_index.grp_0_indexes
    state_count = p_s_2.named_steps.hier_index.grp_0_count

    state_originator_indexes = p_s_2.named_steps.hier_index.grp_0_grp_1_indexes
    state_originator_count = len(state_originator_indexes)

    st_idx = s_3_df["level_0_0"].astype(int).values
    st_originator_idx = s_3_df["level_0_01"].astype(int).values
    
    print(f'Num. States:{state_count}, Num. States/Originators:{state_originator_count}')
else:
    state_indexes = p_s_2.named_steps.hier_index.grp_0_indexes
    state_count = p_s_2.named_steps.hier_index.grp_0_count
    
    orig_indexes = p_s_2.named_steps.hier_index.grp_1_indexes
    orig_count = p_s_2.named_steps.hier_index.grp_1_count
    
    st_idx = s_3_df["level_0"].astype(int).values
    orig_idx = s_3_df["level_1"].astype(int).values
    
    print(f'Num. States:{state_count}, Num. Originators:{orig_count}')

with pm.Model() as hier:
    
    # Data elements, all have N rows.
    # X, U, A, E, y, st_originator_idx
    
    # globals for coefficients
    
    # global mean for a
    a_μ = pm.Normal("a_μ", mu=logit(y.mean()), sigma=1.0, shape=n_intervals)
    a = hierarchical_normal("a", μ=a_μ, sigma=0.5, shape=n_intervals)
    
    # global mean for b
    b_μ = pm.Normal("b_μ", mu=0.0, sigma=0.25, shape=X.shape[1])
    b = hierarchical_normal("b", μ=b_μ, sigma=0.2, shape=X.shape[1])
    
    # global mean for c
    c_μ = pm.Normal("c_μ", mu=0.0, sigma=0.25)
    c = hierarchical_normal("c", μ=c_μ, sigma=0.2, shape=state_count)
    
    # likelihood    
    xbeta = a[A] + pm.math.dot(X, b) + c[st_idx] * U + np.log(E)

    if frailty:
        # log-normal frailty
        γ_μ = pm.Normal("γ_μ", mu=0.0, sigma=1, shape=2)
        γ = hierarchical_normal("γ", μ=γ_μ, sigma=1, shape=2)
        xbeta += γ[orig_idx]
        
    rate = pm.math.exp(xbeta)
    yobs = pm.Poisson('yobs', mu=rate, observed=y)

hier.check_test_point()

# +
prior = pm.sample_prior_predictive(model=hier, random_seed=RANDOM_SEED)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.distplot([x for x in prior["yobs"].mean(axis=1)], ax=ax)
ax.axvline(y.mean(), color="tab:red", label="Sample μ")
ax.axvline(np.array([x for x in prior["yobs"].mean(axis=1)]).mean(), color="tab:blue", label="Prior μ")
ax.legend()
# -

with hier:
    hier_trace = pm.sample(
        draws=n_draws, tune=n_tune, random_seed=RANDOM_SEED, 
        target_accept=0.99,
    )

fname, hier_out_dict = save_results(
    ORIGINATOR, "hier", ASOF_DATE, hier, hier_trace, 
    hard_df, hard_df_train, hard_df_test, s_3_df, 
    p_s_1, p_s_2, p_s_3, numeric_features, categorical_features,
    group_features, group_type, dep_var, pop_covars,
    exp_covars, obs_covars, frailty
)
with open("results/" + fname, "wb") as f:
    joblib.dump(hier_out_dict, f)






