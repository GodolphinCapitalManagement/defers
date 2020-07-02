# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: venv
#     language: python
#     name: venv
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
    ASOF_DATE = datetime.date(2020, 6, 24)

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

# +
# %%time

anonymize = True

df = []
for i in ["PR", "LC"]:
    df.append(make_covid_df(i, ASOF_DATE, anonymize))
hard_df = pd.concat(df, sort=False, ignore_index=True)
hard_df.drop_duplicates(subset=["loan_id"], keep="first", inplace=True, ignore_index=True)

# +
numeric_features = [
    "fico", "original_balance", "dti", "stated_monthly_income", "age", "pct_ic"
]
std_numeric_features = ["std_" + x for x in numeric_features]

categorical_features = [
    "grade", "purpose", "employment_status", "term", "home_ownership", "is_dq"
]

group_features = ["st_code", "originator"]
group_type = "nested"

dep_var = "distress"
# -

hard_df_train = hard_df.groupby(['state', 'originator', dep_var], group_keys=False).apply(
    lambda x: x.sample(frac=0.80, random_state=RANDOM_SEED)
).reset_index().copy()
hard_df_test = hard_df[~hard_df["loan_id"].isin(hard_df_train["loan_id"])].reset_index().copy()

hard_df_train.shape, hard_df_test.shape, hard_df.shape

# ## Transform

knots = hard_df[
    hard_df[dep_var]
]["dur"].quantile(q=[0.10, 0.25, 0.5, 0.75, 0.90]).values
knots

# +
# %%time

p_s_1 = Pipeline(
    steps=[
        ('select_originator', SelectOriginator(ORIGINATOR)),
        ('wide_to_long', WideToLong(dep_var)),
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
        ("interval", IntervalInterceptFeatures()),
    ]
)
s_2_df = p_s_2.fit_transform(s_1_df)
p_s_2
# -

s_2_df.reset_index().to_feather("s_2_df.feather")

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

# +
generic = True
if generic:
    obs_covars = [
        x for x in s_3_df.columns if search("T.", x) and not search("grade", x)
    ] + std_numeric_features[:-1]
else:
    obs_covars = [x for x in s_3_df.columns if search("T.", x)] + std_numeric_features[:-1]
    
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
    f'Observation-level features {X.shape[1]:>3,.0f}'
)

plt.figure(figsize=(12, 4))
corr = s_3_df[obs_covars + [pop_covars]].corr() #stage_three_df.iloc[:, :35].corr() 
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
    # X, E, y, st_originator_idx
    
    # globals for coefficients
    
    # global mean for a
    a_μ = pm.Normal("a_μ", mu=logit(y.mean()), sigma=1.0, shape=n_intervals)
    
    # global mean for b
    b_μ = pm.Normal("b_μ", mu=0.0, sigma=1.0, shape=X.shape[1])
                        
    a = hierarchical_normal("a", μ=a_μ, sigma=0.5, shape=(state_count, n_intervals))
    b = hierarchical_normal("b", μ=b_μ, sigma=0.5, shape=(state_count, X.shape[1]))
    
    phat = tinvcloglog(
         a[st_idx, A] + tt.sum(b[st_idx] * X, axis=1) + np.log(E))

    # likelihood
    yobs = pm.Bernoulli('yobs', p=phat, observed=y)

# ### Hierarchical

with model:
    prior = pm.sample_prior_predictive()
sns.distplot(prior["yobs"].mean(axis=0))

obs_df = s_3_df.groupby(["state", "start"]).agg(n=("note_id", "count"), y=(dep_var, np.mean))
sns.distplot(obs_df.y, kde=False)

model

model.check_test_point()

pm.model_to_graphviz(model)

# ## Fit

n_draws = 1000 # 5000 
n_tune = 1000 # 1000

with model:
    trace = pm.sample(
        draws=n_draws, tune=n_tune, random_seed=RANDOM_SEED, 
        target_accept=0.95,
    )

# ## Analyze

state_originator_indexes_df = p_s_2.named_steps.hier_index.grp_0_grp_1_indexes_df
state_originator_indexes_df = pd.merge(state_originator_indexes_df, states_df, on="st_code")
state_originator_index_map = state_originator_indexes_df[
    ["state", "originator", "level_0_01"]].set_index(["state", "originator"]
)

a_names = ["t_" + str(x) for x in np.arange(n_intervals)]
b_names =  obs_covars + [pop_covars]
st_orig_names = (
    state_originator_indexes_df["state"] + ":" + 
    state_originator_indexes_df["originator"]
).to_list()

index_0_to_st_code_df = state_originator_indexes_df.drop_duplicates(
    subset=["st_code"]
)[["level_0_0", "st_code", "state"]].set_index(["level_0_0"])

trace.varnames

 index_0_to_st_code_df.state.to_list(),

trace["b_σ"].shape

hier_data = az.from_pymc3(
    trace=trace, model=model,
    coords={
        "intercepts": a_names, "pop_covars": b_names,
        'st_code': index_0_to_st_code_df.state.to_list(),
    },
    dims={"b_μ": ["pop_covars"], "b_σ": ["st_code", "pop_covars"], 
          "b": ["st_code", "pop_covars"], "a_μ": ["intercepts"], 
          "a_σ": ["st_code", "intercepts"], "a": ["st_code", "intercepts"],           
        }
)

# ## Summary

# ### Hierarchical

az.plot_trace(hier_data, var_names=["b_μ"]);

az.plot_trace(hier_data, var_names=["a_μ"]);

a_μ_out = az.summary(hier_data, var_names=["a_μ"], round_to=3)
a_μ_out.index = [x + "(μ)" for x in a_names]
a_μ_out

az.summary(hier_data, var_names=["a_σ"], round_to=3)

a_out = az.summary(hier_data, var_names=["a"], round_to=3)
a_out_idx = pd.MultiIndex.from_tuples(
    [(x, y) for x in index_0_to_st_code_df.state.to_list() for y in a_names],
    names=["state", "param"]
)
a_out.index = a_out_idx
a_out

b_out = az.summary(hier_data, var_names=["b"], round_to=3)
b_out_idx = pd.MultiIndex.from_tuples(
    [(x, y) for x in index_0_to_st_code_df.state.to_list() for y in b_names],
    names=["state", "param"]
)
b_out.index = b_out_idx
b_out

st_η_samples_df = pd.DataFrame(trace["b"][:, :, -1], columns=index_0_to_st_code_df.state.to_list())
st_η_pos = []
for v in index_0_to_st_code_df.state.to_list():
    st_η_pos.append([v, 100 - stats.percentileofscore(st_η_samples_df[v], 0.0)])
st_η_pos_df = pd.DataFrame(st_η_pos, columns=["state", "P(η > 0 | X)"])
st_η_pos_df.set_index("state", inplace=True)

st_η_pos_df.loc[["NJ", "NY", "CA", "FL", "WA", "NV", "TX"]]

b_out.loc[idx[:, "std_pct_ic"], :].droplevel(level=1)

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
    hier_data, var_names=["b_μ"], coords={"pop_covars": ["std_fico", "std_pct_ic"]},
    divergences=True,
);

az.plot_rank(hier_data, var_names=("b_μ"));

# +
α = μ_a_out["mean"]
β = g_out["mean"]

dp_dx = pd.DataFrame(
    {"param": b_names, "dp_dx": 10000 * np.array([d_cinvloglog(X, A, α, β, i) for i, v in enumerate(b_names)])}
)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
sns.barplot(data=dp_dx, y="param", x="dp_dx", ax=ax)
ax.set_ylabel("Parameter")
ax.set_xlabel("dP/dX (bps)");
# -

dp_dx_df = dp_dx.set_index("param")
f'{dp_dx_df.loc["employment_status[T.Self-employed]", "dp_dx"]:.0f}'

hier_data.posterior.a.coords

# +
# %%time

plot_trace = True
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
if plot_trace:
    _ = az.plot_forest(
        hier_data, var_names=["a"],
        coords={"st_code": ["NJ"], "intercepts": a_names}, 
        combined=True, ax=ax[0]
    )
    _ = az.plot_forest(
        hier_data, var_names=["b"], 
        coords={"st_code": ["NJ"], "pop_covars": b_names},
        combined=True, ax=ax[1]
    );
plt.tight_layout()
# -
# ## Save Results

do_tests = True
if do_tests:
    loo = az.loo(hier_data, pointwise=True)
    az.plot_khat(loo, bin_format=True)

fff = loo.pareto_k.to_series()
s_3_df[fff>0.5]["state"].unique()

hard_df_train.groupby(hard_df_train["state"]).size()

loo

fname, out_dict = save_results(
    ORIGINATOR, "hier", ASOF_DATE, model, trace, 
    hard_df, hard_df_train, hard_df_test, s_3_df, 
    p_s_1, p_s_2, p_s_3, knots, loo, None
)

save_output = True
if save_output:
    with open("results/" + fname, "wb") as f:
        joblib.dump(out_dict, f)

us_states = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip")
# us_counties = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2019/shp/cb_2019_us_county_20m.zip")

# +
dff_η = b_out.loc[idx[[x for x in states_df.state], "std_pct_ic"], :].droplevel(
    level=1).reset_index().rename(columns={"mean": "value"}
)
merged_us_states_η = pd.merge(us_states, dff_η, left_on="STUSPS", right_on="state", how="right")

fig, ax = plt.subplots(1, figsize=(15, 18))

albers_epsg = 2163
ax = us_states[~us_states["STATEFP"].isin(['02', '15'])].to_crs(epsg=albers_epsg).plot(
    ax=ax, linewidth=0.25, edgecolor='white', color='grey'
)

ax = merged_us_states_η[~merged_us_states_η["STATEFP"].isin(['02', '15'])].to_crs(epsg=albers_epsg).plot(
    column='value', ax=ax, cmap='viridis', 
    scheme="quantiles", legend=True,  legend_kwds={"loc": "upper center", "ncol": 3}
)
_ = ax.axis('off')    
# -



# ## In-sample validation

# ### Posterior predictive distribution

posterior_predictive = pm.sample_posterior_predictive(trace, model=model)
y_hat = posterior_predictive["yobs"].mean(axis=0)

# +
_, ax = plt.subplots(figsize=(10, 5))

ax.hist(y_hat, bins=19, alpha=0.5)
ax.axvline(s_3_df[dep_var].mean())
ax.set(xlabel='Deferment Pct.', ylabel='Frequency')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

pctile = np.percentile(y_hat, q=[5, 95])
ax.axvline(pctile[0], color="red", linestyle=":")
ax.axvline(pctile[1], color="red", linestyle=":")

ax.text(1.2 * s_3_df[dep_var].mean(), 0.85 * ax.get_ylim()[1], f'95% HPD: [{pctile[0]:.2%}, {pctile[1]:.2%}]');

# +
# %%time

aaa, out_df =  predict(
    None, hard_df_train, dep_var, ic_long_df, ASOF_DATE, "hier", out_dict,
    numeric_features, generic, n_samples=1000, verbose=False
)

# +
_, ax = plt.subplots(figsize=(10, 5))

ax.hist(aaa.mean(axis=0), bins=19, alpha=0.5)
ax.axvline(out_df[dep_var].mean())
ax.set(xlabel='Deferment Pct.', ylabel='Frequency')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
pctile = np.percentile(aaa.mean(axis=0), q=[5, 95])
ax.axvline(pctile[0], color="red", linestyle=":")
ax.axvline(pctile[1], color="red", linestyle=":")

ax.text(1.2 * out_df[dep_var].mean(), 0.85 * ax.get_ylim()[1], f'95% HPD: [{pctile[0]:.2%}, {pctile[1]:.2%}]');
# -

# ## Out-of-sample validation

# +
# %%time

aaa, out_df =  predict(
    None, hard_df_test, dep_var, ic_long_df, ASOF_DATE, "hier", out_dict,
    numeric_features, generic, n_samples=1000, verbose=False
)
# +
_, ax = plt.subplots(figsize=(10, 5))

ax.hist(aaa.mean(axis=0), bins=19, alpha=0.5)
ax.axvline(out_df[dep_var].mean())
ax.set(xlabel='Deferment Pct.', ylabel='Frequency')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
pctile = np.percentile(aaa.mean(axis=0), q=[5, 95])
ax.axvline(pctile[0], color="red", linestyle=":")
ax.axvline(pctile[1], color="red", linestyle=":")

ax.text(1.2 * out_df[dep_var].mean(), 0.85 * ax.get_ylim()[1], f'95% HPD: [{pctile[0]:.2%}, {pctile[1]:.2%}]');

# +
pctile = np.percentile(aaa, q=[5, 95], axis=0).T
    
zzz = pd.concat(
    [
        out_df, pd.DataFrame(
            np.hstack(
                (
                    aaa.mean(axis=0).reshape(-1, 1), aaa.std(axis=0).reshape(-1, 1),
                    pctile
                )
            ), 
            columns=["ymean", "ystd", "y5", "y95"], index=out_df.index
        )
    ], axis=1
)
# +
zzz["fhaz"] = zzz.groupby(level=0).agg(chaz=("ymean", np.cumsum))["chaz"].map(lambda x: 1 - np.exp(-x))
zzz_df = zzz.groupby("stop").agg(
    y=(dep_var, np.mean), ymean=("ymean", np.mean), ystd=("ystd", np.mean),
    y5=("y5", np.mean), y95=("y95", np.mean)
).reset_index()

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.plot(zzz_df["stop"], zzz_df["ymean"], label="Predicted")
ax.scatter(zzz_df["stop"], zzz_df["y"], label="Actual")

ax.fill_between(
    zzz_df["stop"], zzz_df["y5"], zzz_df["y95"], color="red", alpha=0.05, label="95% Interval"
)
ax.set(xlabel='Week', ylabel='Hazard')
ax.legend(loc="upper right");
# +
# %%time

zzz = forecast_hazard(
    hard_df_test, dep_var, numeric_features, "hier", out_dict, claims_dict, ASOF_DATE, 
    datetime.date(2020, 9, 30)
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


