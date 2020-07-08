# -*- coding: utf-8 cspell: disable-*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# +
import pandas as pd
import numpy as np
import QuantLib
import patsy
import feather

from fastprogress.fastprogress import master_bar, progress_bar
# -

# %load_ext watermark

import lifelines
from lifelines import KaplanMeierFitter, NelsonAalenFitter


# +
import pymc3 as pm
import arviz as az

pm.__version__, az.__version__

# +
from IPython.display import IFrame
from highcharts import Highmap

import geopandas as gpd
# -

import statsmodels.api as sm
import statsmodels.formula.api as smf

# +
# %matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
# -

from sklearn import set_config
set_config(display='diagram')

sns.set()
plt.rcParams.update({
    "font.family": "Source Sans Pro",
    "font.serif": ["Source Sans Pro"],  # use latex default serif font
    "font.sans-serif": ["Source Sans Pro"],  # use a specific sans-serif font
    "font.size": 10,
})

external_files_dir = "/home/gsinha/admin/db/dev/Python/projects/models/data/"

# %run "/home/gsinha/admin/db/dev/Python/projects/models/defers/common.py"

RANDOM_SEED = 8112
np.random.seed(370)

states = states_df["state"].to_list()

fname = external_files_dir + "claims.pkl"
with open(fname, "rb") as f:
    claims_dict = joblib.load(f)

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
# -

# ### Definitions

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
dq_tbl = summary_by_group(
    ["originator", "loanstatus"], dep_var, hard_df
)
dq_tbl.index.names = ["Originator", "DQ Status"]

dq_tbl

# ### Nelson-Aalen Hazards

# +
T = hard_df.dur
E = hard_df[dep_var]

bandwidth = 1
naf = NelsonAalenFitter()
lc = hard_df["originator"].isin([omap["LC"]])

naf.fit(T[lc],event_observed=E[lc], label="Originator I")
ax = naf.plot_hazard(bandwidth=bandwidth, figsize=(10, 5))

naf.fit(T[~lc], event_observed=E[~lc], label="Originator II")
naf.plot_hazard(ax=ax, bandwidth=bandwidth)

naf.fit(T, event_observed=E, label="All")
naf.plot_hazard(ax=ax, bandwidth=bandwidth)

ax.set_xlabel("Weeks since March 14th, 2020")
ax.set_ylabel("Weekly hazard")

_  = plt.xlim(0, hard_df.dur.max() + 1)
# -

lt_df = lifelines.utils.survival_table_from_events(
    T, E, collapse=True
)
lt_df

# ### Training/Test samples

hard_df_train = hard_df.groupby(['state', 'originator', dep_var], group_keys=False).apply(
    lambda x: x.sample(frac=0.80, random_state=RANDOM_SEED)
).reset_index().copy()
hard_df_test = hard_df[~hard_df["loan_id"].isin(hard_df_train["loan_id"])].reset_index().copy()

# ### Transform

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
X = s_3_df[obs_covars + [pop_covars]].values
E = s_3_df[exp_covars].values

A = s_3_df["a_indx"].values
# -

print(
    f'Observation-level features {X.shape[1]:>3,.0f}'
)

fig, ax = plt.subplots(1,1, figsize=(10, 10))
corr = s_3_df[obs_covars].corr() #stage_three_df.iloc[:, :35].corr() 
mask = np.tri(*corr.shape).T 
sns.heatmap(corr.abs(), mask=mask, annot=False, cmap='viridis', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90);

with pm.Model() as model:
    μ_a = pm.Normal("μ_a", logit(y.mean()), sigma=0.35)
    a = hierarchical_normal("a", μ=μ_a, sigma=0.20, shape=n_intervals)
    
    μ_b = pm.Normal("μ_b", mu=0, sigma=0.35)
    b = hierarchical_normal("b", μ=μ_b, sigma=0.20, shape=X.shape[1])
    
    xbeta = pm.Deterministic("xbeta", a[A] + pm.math.dot(X, b))
    yobs = pm.Poisson("yobs", mu=pm.math.exp(xbeta) * E, observed=y)

# +
# %%time

yo = []
for i in np.arange(1):
    prior = pm.sample_prior_predictive(model=model)
    yo.append([prior["yobs"].min(), prior["yobs"].max()])
yo_df = pd.DataFrame(yo, columns=["ymin", "ymax"])
# -

sns.distplot(np.exp(prior["xbeta"]).mean(axis=0))

rate_df = pd.DataFrame(np.exp(prior["xbeta"]), columns=s_3_df.index.get_level_values(0)).T
(rate_df.apply(max, axis=1)).quantile(q=np.linspace(0,1,5))

obs_df = s_3_df.groupby(["state", "start"]).agg(n=("note_id", "count"), y=(dep_var, np.mean))
sns.distplot(obs_df.y, kde=False)

model

model.check_test_point()

pm.model_to_graphviz(model)

# ## Fit

n_draws = 1000
n_tune = 1000

with model:
    trace = pm.sample(
        n_draws, tune=n_tune, random_seed=RANDOM_SEED,
        target_accept=0.99,
        # init="advi+adapt_diag"
    )

# ### Diagnostics

# +
fig, ax = plt.subplots(3, 2, figsize=(10, 5))

ax[0,0].plot(trace['step_size_bar'])
ax[0,0].set_title("step_size_bar")

ax[0,1].plot(trace["step_size"], label="step_size")
ax[0,1].set_title("step_size")

sizes = trace.get_sampler_stats('depth', combine=True)
ax[1,0].plot(sizes, label="depth")
ax[1,0].set_title("depth")
ax[1,1].plot(trace["tree_size"], label="tree_size")
ax[1,1].set_title("tree_size")

accept = trace.get_sampler_stats('mean_tree_accept')
sns.distplot(accept, ax=ax[2,0])
ax[2,0].set_title("mean_tree_accept")

ax[2,1].plot(trace["model_logp"])
ax[2,1].set_title("model_logp")

plt.tight_layout()
# -

accept.mean(), trace["depth"].mean()


# +
def pairplot_divergence(trace, ax=None, divergence=True, color='C3', divergence_color='C2'):
    theta = trace.get_values(varname='b', combine=True)[:, 1]
    logtau = trace.get_values(varname='a_σ_log__', combine=True)
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(theta, logtau, 'o', color=color, alpha=.5)
    if divergence:
        divergent = trace['diverging']
        ax.plot(theta[divergent], logtau[divergent], 'o', color=divergence_color)
    ax.set_xlabel('μ_b')
    ax.set_ylabel('log(σ_b)')
    ax.set_title('scatter plot between log(σ_b) and μ_b');
    return ax

pairplot_divergence(trace);
# -

posterior_predictive = pm.sample_posterior_predictive(trace, model=model)
y_hat = posterior_predictive["yobs"].mean(axis=0)
sns.distplot(posterior_predictive["yobs"].mean(axis=0))

# ## Analyze

a_names = ["t_" + str(x) for x in np.arange(n_intervals)]
b_names = obs_covars + [pop_covars]

pooled_data = az.from_pymc3(
    trace=trace, prior=prior, model=model, 
    posterior_predictive=posterior_predictive,
    coords={'covars': b_names, 'intercepts': a_names}, 
    dims={'b': ['covars'], 'a': ['intercepts']}
)

az.plot_pair(
    pooled_data, var_names=["b"], coords={"covars": ["std_fico_0", "std_pct_ic"]},
    divergences=True,
);

az.summary(pooled_data, var_names=["a_σ", "b_σ"], round_to=3)

sns.pairplot(
    pd.DataFrame(trace["b"][:,:6], columns=b_names[:6]),
);

# ## Summary

# ### Pooled

pooled_out = az.summary(pooled_data, var_names=["a", "b"], round_to=3)
pooled_out.index = a_names + b_names
pooled_out

az.plot_forest(pooled_data, var_names=["a"], combined=True);

az.plot_forest(pooled_data, var_names=["b"], combined=True);

az.plot_trace(
    pooled_data, var_names=["b"], 
    coords={"covars": ["std_fico_0", "std_fico_1", "std_pct_ic", "std_age"]}
);

enc = OneHotEncoder()
enc.fit(s_3_df[["start"]])
XX = pd.concat(
    [
        pd.DataFrame(enc.transform(s_3_df[["start"]]).toarray(), columns=a_names),
        pd.DataFrame(X.copy(), columns=b_names)
    ], axis=1
)
pool_mle = sm.GLM(
    y, XX, family=sm.families.Binomial(link=sm.genmod.families.links.cloglog()), 
    offset=np.log(s_3_df["exposure"])
).fit()

print(pool_mle.summary())

# ## Save Results

do_tests = True
if do_tests:
    loo = az.loo(pooled_data, pointwise=True)
    az.plot_khat(loo, bin_format=True)

loo

fname, out_dict = save_results(
    ORIGINATOR, "pooled", ASOF_DATE, model, trace, 
    hard_df, hard_df_train, hard_df_test, s_3_df, 
    p_s_1, p_s_2, p_s_3, loo, None,
    numeric_features, categorical_features,
    group_features, group_type, dep_var, pop_covars,
    exp_covars, obs_covars,
)

save_output = True
if save_output:
    with open("results/" + fname, "wb") as f:
        joblib.dump(out_dict, f)

# ## Validation

pooled_ppc, out_df =  predict(
    None, hard_df_test, dep_var, out_dict, None, n_samples=4000, 
    verbose=False
)

# +
_, ax = plt.subplots(figsize=(10, 5))

ax.hist(pooled_ppc.mean(axis=0), bins=19, alpha=0.5)
ax.axvline(out_df[dep_var].mean())

ax.set(xlabel='Deferment Pct.', ylabel='Frequency')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

pctile = np.percentile(pooled_ppc.mean(axis=0), q=[5, 95])
ax.axvline(pctile[0], color="red", linestyle=":")
ax.axvline(pctile[1], color="red", linestyle=":")

_ = ax.text(
    1.65 * out_df[dep_var].mean(), 0.85 * ax.get_ylim()[1], 
    f'95% HPD: [{pctile[0]:.2%}, {pctile[1]:.2%}]'
)

# +
pctile = np.percentile(pooled_ppc, q=[5, 95], axis=0).T
    
zzz = pd.concat(
    [
        out_df, pd.DataFrame(
            np.hstack(
                (
                    pooled_ppc.mean(axis=0).reshape(-1, 1), pooled_ppc.std(axis=0).reshape(-1, 1),
                    pctile
                )
            ), 
            columns=["ymean", "ystd", "y5", "y95"], index=out_df.index
        )
    ], axis=1
)

zzz["fhaz"] = zzz.groupby(level=0).agg(chaz=("ymean", np.cumsum))["chaz"].map(lambda x: 1 - np.exp(-x))
zzz_df = zzz.groupby("start").agg(
    y=(dep_var, np.mean), ymean=("ymean", np.mean), ystd=("ystd", np.mean),
    y5=("y5", np.mean), y95=("y95", np.mean)
).reset_index()

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.plot(zzz_df["start"], zzz_df["ymean"], label="Predicted")
ax.scatter(zzz_df["start"], zzz_df["y"], label="Actual", color="red")

ax.fill_between(
    zzz_df["start"], zzz_df["y5"], zzz_df["y95"], color="red", alpha=0.05, label="95% Interval"
)
ax.set(xlabel='Week', ylabel='Hazard')
ax.legend(loc="upper right");

# +
zzz = pd.concat(
    [
        out_df, pd.DataFrame(
            np.hstack((pooled_ppc.mean(axis=0).reshape(-1, 1), pooled_ppc.std(axis=0).reshape(-1, 1))), 
            columns=["ymean", "ystd"], index=out_df.index)
    ], axis=1
)
zzz_df = zzz.groupby("start").agg(
    y=(dep_var, np.mean), ymean=("ymean", np.mean),
    ystd=("ystd", np.mean), n=("loan_id", "count"),
    k=(dep_var, np.sum)
).reset_index()

fig, ax = plt.subplots(1,1, figsize=(10, 5))

ax.scatter(zzz_df["start"], zzz_df["y"], label="observed", color="red")
ax.plot(zzz_df["start"], zzz_df["ymean"], label="predicted")
ax.fill_between(
    zzz_df["start"], zzz_df["ymean"] - 2*zzz_df["ystd"], 
    zzz_df["ymean"] + 2*zzz_df["ystd"], color="red", alpha=0.05, label="+/- 2 std"
)
ax.set(xlabel='Week', ylabel='Hazard')
ax.legend(loc="upper left");
# -
α = pooled_out.iloc[:n_intervals]["mean"]
β = pooled_out.iloc[n_intervals:]["mean"]
U = None
c = None
d = None
dp_dx = pd.DataFrame(
    {"param": b_names, 
     "dp_dx": 10000 * np.array([d_poisson(X, A, U, E, α, β, c, d, v, "pooled", False) for i, v in enumerate(b_names)])}
)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
sns.barplot(data=dp_dx, y="param", x="dp_dx", ax=ax)
ax.set_ylabel("Parameter")
ax.set_xlabel("dP/dX (bps)");

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


