# -*- coding: utf-8 cspell: disable -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
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
import QuantLib
import patsy
import feather

# %load_ext watermark

from scipy.special import logit

# +
import pymc3 as pm
import arviz as az

pm.__version__, az.__version__
# -

import geopandas as gpd

import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy

from scipy import stats

# +
# %matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
# -

from sklearn.model_selection import train_test_split

from sklearn import set_config
set_config(display='diagram')

# %run "/home/gsinha/admin/db/dev/Python/projects/models/defers/common.py"

sns.set()
plt.rcParams.update({
    "font.family": "Source Sans Pro",
    "font.serif": ["Source Sans Pro"],  # use latex default serif font
    "font.sans-serif": ["Source Sans Pro"],  # use a specific sans-serif font
    "font.size": 10,
})

external_files_dir = "/home/gsinha/admin/db/dev/Python/projects/models/data/"
models_dir = "/home/gsinha/admin/db/dev/Python/projects/models/"

fname = external_files_dir + "claims.pkl"
with open(fname, "rb") as f:
    claims_dict = joblib.load(f)

RANDOM_SEED = 8112
np.random.seed(370)

states = states_df["state"].to_list()

# # Deferment Model

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
    "grade", "purpose", "employment_status", "term", "home_ownership", "loanstatus"
]

group_features = ["st_code", "originator"]
group_type = "crossed"

dep_var = "defer"
frailty = True

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
    lambda x: x.sample(frac=0.80, random_state=RANDOM_SEED)
).reset_index().copy()
hard_df_test = hard_df[~hard_df["loan_id"].isin(hard_df_train["loan_id"])].reset_index().copy()

hard_df_train.shape, hard_df_test.shape, hard_df.shape

# ## Transform

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
# -

if group_type is "crossed":
    s_2_df.to_feather("s_2_df.feather")
else:
    s_2_df.reset_index("s_2_df.feather")

# +
# %%time

p_s_3 = Pipeline(
    steps=[
        ('dummy', Dummy(categorical_features)),
        ('cross', Interaction(["grade", "std_fico"])),
    ]
)
s_3_df = p_s_3.fit_transform(s_2_df)
p_s_3
# -

print(
    f'Wide format records: {hard_df.shape[0]:>6,.0f}\n'
    f'Wide format features: {hard_df.shape[1]:>3,.0f}\n'
    f'Long format records: {s_3_df.shape[0]:>6,.0f}\n'
    f'Long format features {s_3_df.shape[1]:>3,.0f}'
)

data_scaler = p_s_2.named_steps["standardize"].numeric_transformer.named_steps["scaler"]
for x, y, z in zip(numeric_features, data_scaler.mean_ , data_scaler.scale_):
    print(f'{x}: μ={y:.2f}, σ={z:.2f}')

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
X = s_3_df[obs_covars].values
U = s_3_df[pop_covars].values
E = s_3_df[exp_covars].values

A = s_3_df["a_indx"].values
O = s_3_df["level_1"].values
# -

print(
    f'Observation-level features {X.shape[1]:>3,.0f}'
)

plt.figure(figsize=(10, 10))
corr = s_3_df[obs_covars + [pop_covars]].corr()
mask = np.tri(*corr.shape).T 
sns.heatmap(corr.abs(), mask=mask, annot=False, cmap='viridis')
plt.xticks(rotation=60);

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

# ## Models

with pm.Model() as model:
    
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
    xbeta = pm.Deterministic("xbeta", a[A] + pm.math.dot(X, b) + c[st_idx] * U + np.log(E))

    if frailty:
        # log-normal frailty
        γ_μ = pm.Normal("γ_μ", mu=0.0, sigma=1, shape=2)
        γ = hierarchical_normal("γ", μ=γ_μ, sigma=1, shape=2)
        xbeta += γ[orig_idx]
        
    rate = pm.math.exp(xbeta)
    yobs = pm.Poisson('yobs', mu=rate, observed=y)

# ### Hierarchical

prior = pm.sample_prior_predictive(model=model, random_seed=RANDOM_SEED)

prior["yobs"].max(), prior["yobs"].min()

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.distplot([x for x in prior["yobs"].mean(axis=1)], ax=ax)
ax.axvline(y.mean(), color="tab:red", label="Sample μ")
ax.axvline(np.array([x for x in prior["yobs"].mean(axis=1)]).mean(), color="tab:blue", label="Prior μ")
ax.legend()

model

model.check_test_point()

pm.model_to_graphviz(model)

# ## Fit

n_draws = 1000 # 5000 
n_tune = 1000 # 1000

with model:
    trace = pm.sample(
        draws=n_draws, tune=n_tune, random_seed=RANDOM_SEED, 
        target_accept=0.99,
    )

posterior_predictive = pm.sample_posterior_predictive(trace, model=model)
y_hat = posterior_predictive["yobs"].mean(axis=0)

fig, ax= plt.subplots(1, 1, figsize=(10, 5))
ax.hist(y_hat, bins="sqrt", color="red");

# ## Analyze

state_indexes_df = p_s_2.named_steps.hier_index.grp_0_index
state_indexes_df = pd.merge(state_indexes_df, states_df, on="st_code")
state_index_map = state_indexes_df.drop_duplicates(subset=["state"])

a_names = ["t_" + str(x) for x in np.arange(n_intervals)]
b_names =  obs_covars
c_names = pop_covars

hier_data = az.from_pymc3(
    trace=trace, model=model, prior=prior,
    posterior_predictive=posterior_predictive,
    coords={
        "intercepts": a_names, "obs_covars": b_names, 
        'st_code': state_index_map.state.to_list(),
    },
    dims={
        "b_μ": ["obs_covars"], "b_σ": ["obs_covars"], "b": ["obs_covars"],
        "a_μ": ["intercepts"], "a_σ": ["intercepts"], "a": ["intercepts"],
        "c": ["st_code"], 
    }
)

# +
# %%time

do_tests = True
if do_tests:
    loo = az.loo(hier_data, pointwise=True)
    az.plot_khat(loo, bin_format=True)
loo
# -

# ## Summary

az.plot_trace(hier_data, var_names=["γ_μ", "γ_σ"]);

az.summary(hier_data, var_names=["γ_μ", "γ_σ", "γ"], round_to=3)

# ### Hierarchical

az.plot_trace(hier_data, var_names=["a_μ"]);

az.plot_forest(hier_data, var_names=["c"], combined=True);

a_μ_out = az.summary(hier_data, var_names=["a_μ"], round_to=3)
a_μ_out.index = [x + "(μ)" for x in a_names]
a_μ_out

az.summary(hier_data, var_names=["a_σ"], round_to=3)

a_out = az.summary(hier_data, var_names=["a"], round_to=3)
a_out.index = a_names
a_out

b_out = az.summary(hier_data, var_names=["b"], round_to=3)
b_out.index = b_names
b_out

st_c_out = az.summary(hier_data, var_names=["c"], round_to=3)
st_c_out.index = state_index_map.state.to_list()
st_c_out

st_c_samples_df = pd.DataFrame(trace["c"], columns=state_index_map.state.to_list())

st_c_pos = []
for v in state_index_map.state.to_list():
    st_c_pos.append([v, 100 - stats.percentileofscore(st_c_samples_df[v], 0.0)])
st_c_pos_df = pd.DataFrame(st_c_pos, columns=["state", "P(η > 0 | X)"])
st_c_pos_df.set_index("state", inplace=True)

st_c_pos_df.loc[["NJ", "NY", "CA", "FL", "WA", "NV", "TX", "LA", "MN", "KS", "IN"]].T

st_c_out.loc[["NJ", "NY", "CA", "FL", "WA", "NV", "TX", "LA", "MN", "KS", "IN"]]

az.plot_ess(
    hier_data, kind="evolution", var_names=["a_μ"],
);

az.plot_ess(
    hier_data, kind='evolution', var_names=["b_μ"]
);

az.plot_ess(
    hier_data, var_names=["a_μ"], kind="local",
    drawstyle="steps-mid", color="k",
    linestyle="-", marker=None, rug=True, rug_kwargs={"color": "r"}
);

az.plot_energy(hier_data)

az.plot_pair(
    hier_data, var_names=["b_μ"], coords={"obs_covars": ["std_fico_0", "std_stated_monthly_income_0"]},
    divergences=True,
);

az.plot_rank(hier_data, var_names=("b_μ"));

# +
a = a_out["mean"]
b = b_out["mean"]
c = trace["c_μ"].mean()
d = trace["γ_μ"].mean()

dp_dx = pd.DataFrame(
    {"param": b_names, "dp_dx": 10000 * np.array([d_poisson(X, A, U, E, a, b, c, d, v, "hier", frailty) for v in b_names])}
)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
sns.barplot(data=dp_dx, y="param", x="dp_dx", ax=ax)
ax.set_ylabel("Parameter")
ax.set_xlabel("dP/dX (bps)");

# +
a = a_out["mean"]
b = b_out["mean"]
d = trace["γ_μ"].mean()
    
cbeta_d = []
for i, v in enumerate(state_index_map.state):
    indx = state_index_map.set_index("state").loc[v, "level_0"]
    c = trace["c"][:, indx].mean()
    dp_dx =  10000 * np.array(d_poisson(X, A, U, E, a, b, c, d, "std_pct_ic", "hier", frailty))
    cbeta_d.append([v, dp_dx])

cbeta_df = pd.DataFrame(cbeta_d, columns=["state", "ame"])
cbeta_df.sort_values(by=["ame"], ascending=False, inplace=True)

fig, ax = plt.subplots(1, 1, figsize=(8,10))
sns.barplot(x="ame", y="state", data=cbeta_df, ax=ax)
ax.set_xlabel("Average Marginal Effect (bps)")
ax.set_ylabel("State")
ax.set_yticklabels(ax.get_yticklabels(), size=10);
# -

# ## Save Results

fname, out_dict = save_results(
    ORIGINATOR, "hier", ASOF_DATE, model, trace, 
    hard_df, hard_df_train, hard_df_test, s_3_df, 
    p_s_1, p_s_2, p_s_3, loo, None,
    numeric_features, categorical_features,
    group_features, group_type, dep_var, pop_covars,
    exp_covars, obs_covars, frailty
)

save_output = True
if save_output:
    with open("results/" + fname, "wb") as f:
        joblib.dump(out_dict, f)

us_states = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip")

# +
merged_us_states_c = pd.merge(us_states, cbeta_df, left_on="STUSPS", right_on="state", how="right")

fig, ax = plt.subplots(1, 1, figsize=(15, 18))

albers_epsg = 2163
ax = us_states[~us_states["STATEFP"].isin(['02', '15'])].to_crs(epsg=albers_epsg).plot(
    ax=ax, linewidth=0.25, edgecolor='white', color='grey'
)

ax = merged_us_states_c[~merged_us_states_c["STATEFP"].isin(['02', '15'])].to_crs(epsg=albers_epsg).plot(
    column='ame', ax=ax, cmap='viridis', 
    scheme="quantiles", legend=True,  legend_kwds={"loc": "upper center", "ncol": 3}
)
_ = ax.axis('off')    
# -
# ## In-sample validation

# +
# %%time

hier_ppc_train, out_df = predict(
    None, hard_df_train, dep_var, out_dict, ic_long_df=None, 
    n_samples=4000, verbose=False
)

# +
_, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.hist(hier_ppc_train.mean(axis=0), bins=19, alpha=0.5)
ax.axvline(out_df[dep_var].mean())
ax.set(xlabel='Deferment Pct.', ylabel='Frequency')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
pctile = np.percentile(hier_ppc_train.mean(axis=0), q=[5, 95])
ax.axvline(pctile[0], color="red", linestyle=":")
ax.axvline(pctile[1], color="red", linestyle=":")

ax.text(1.2 * out_df[dep_var].mean(), 0.85 * ax.get_ylim()[1], f'95% HPD: [{pctile[0]:.2%}, {pctile[1]:.2%}]');
# -

# ## Out-of-sample validation

# +
# %%time

hier_ppc_test, out_df =  predict(
    None, hard_df_test, dep_var,  out_dict, ic_long_df=None, 
    n_samples=1000, verbose=False
)
# +
_, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.hist(hier_ppc_test.mean(axis=0), bins=19, alpha=0.5)
ax.axvline(out_df[dep_var].mean())
ax.set(xlabel='Deferment Pct.', ylabel='Frequency')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
pctile = np.percentile(hier_ppc_test.mean(axis=0), q=[5, 95])
ax.axvline(pctile[0], color="red", linestyle=":")
ax.axvline(pctile[1], color="red", linestyle=":")

ax.text(1.2 * out_df[dep_var].mean(), 0.85 * ax.get_ylim()[1], f'95% HPD: [{pctile[0]:.2%}, {pctile[1]:.2%}]');

# +
pctile = np.percentile(hier_ppc_test, q=[5, 95], axis=0).T
    
zzz = pd.concat(
    [
        out_df, pd.DataFrame(
            np.hstack(
                (
                    hier_ppc_test.mean(axis=0).reshape(-1, 1), 
                    hier_ppc_test.std(axis=0).reshape(-1, 1),
                    pctile
                )
            ), 
            columns=["ymean", "ystd", "y5", "y95"], index=out_df.index
        )
    ], axis=1
)
# +
zzz["fhaz"] = zzz.groupby(level=0).agg(chaz=("ymean", np.cumsum))["chaz"].map(lambda x: 1 - np.exp(-x))
zzz_df = zzz.groupby("start").agg(
    y=(dep_var, np.mean), ymean=("ymean", np.mean), ystd=("ystd", np.mean),
    y5=("y5", np.mean), y95=("y95", np.mean)
).reset_index()

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.plot(zzz_df["start"], zzz_df["ymean"], label="Predicted")
ax.scatter(zzz_df["start"], zzz_df["y"], label="Actual")

ax.fill_between(
    zzz_df["start"], zzz_df["y5"], zzz_df["y95"], color="red", alpha=0.05, label="95% Interval"
)
ax.set(xlabel='Week', ylabel='Hazard')
ax.legend(loc="upper right");
# +
# %%time

zzz = forecast_hazard(
    hard_df_test, dep_var, out_dict, claims_dict, datetime.date(2020, 9, 30)
)

zzz_df = zzz.groupby(level=1).agg(
    y=(dep_var, np.mean), ymean=("ymean", np.mean),
    ystd=("ystd", np.mean), y5=("y5", np.mean), y95=("y95", np.mean)
)

# +
fig, ax = plt.subplots(1,1, figsize=(10, 5))
ax.plot(zzz_df.loc["2020-03-14":"2020-07-26", "ymean"], linewidth=5, color="blue", label="Average")

for i in zzz.index.get_level_values(0).to_series().sample(n=100, random_state=12345):
    zzz.loc[idx[i, "2020-03-14":"2020-07-26"], ["ymean"]].reset_index().plot(
        x="edate", y="ymean", ax=ax, legend=False, alpha=0.25
    )

_ = ax.set(xlabel='Week ending', ylabel='Hazard')
# -

# %watermark -a GyanSinha -n -u -v -iv -w


