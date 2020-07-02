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
#     display_name: dev
#     language: python
#     name: dev
# ---

# +
import os
import joblib
import datetime

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
# -

import pandas as pd
import numpy as np

# +
import pymc3 as pm
import arviz as az

pm.__version__, az.__version__
# -

# %load_ext watermark

# +
# %matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
# -

from sklearn.model_selection import train_test_split

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

override_asof_date = True
if override_asof_date:
    ASOF_DATE = datetime.date(2020, 6, 30)

print(f'As Of Date: {ASOF_DATE}')

# +
omap = {"LC": "I", "PR": "II", "ALL": None}

base_dir = "/home/gsinha/admin/db/dev/Python/projects/models/"
results_dir = {
  "LC": base_dir + "defers/pymc3/" + "originator_" + omap["LC"] + "/results",
  "PR": base_dir + "defers/pymc3/" + "originator_" + omap["PR"] + "/results",
  "ALL": base_dir + "defers/pymc3/" + "results/"
}
# -

out_dict = {}
for i in ["pooled", "hier"]:
    out_dict[i] = read_results(i, None, ASOF_DATE, results_dir["ALL"])

with open(external_files_dir + "claims.pkl", "rb") as f:
    claims_dict = joblib.load(f)

horizon_date = datetime.date(2020, 9, 30)
dep_var = "distress"

test_df = out_dict["hier"]["test"]

# +
data_scaler = (
    out_dict["hier"]["pipe"]["p_s_2"].named_steps["standardize"].numeric_transformer.named_steps["scaler"]
)

numeric_features = [
    "fico", "original_balance", "dti", "stated_monthly_income", "age", "pct_ic"
]

data_scaler_dict = {
    "mu" : dict(zip(numeric_features, data_scaler.mean_)),
    "sd":  dict(zip(numeric_features, data_scaler.scale_))
}
# -

sub_df = make_df(test_df, dep_var, ASOF_DATE, horizon_date)

# +
# %%time

aaa, zzz, _ = simulate(
    None, sub_df, dep_var, claims_dict["chg_df"], ASOF_DATE, 
    "hier", out_dict, numeric_features, True
)
zzz.set_index(["loan_id", "edate"], inplace=True)
zzz.sort_index(inplace=True)
# -

zzz["fhaz"] = zzz.groupby(level=0).agg(chaz=("ymean", np.cumsum))["chaz"].map(lambda x: 1 - np.exp(-x))
zzz_df = zzz.groupby("stop").agg(
    y=(dep_var, np.mean), ymean=("ymean", np.mean),
    ystd=("ystd", np.mean), y5=("y5", np.mean), y95=("y95", np.mean)
).reset_index()

ASOF_DATE.strftime("%Y-%m-%d"), ic_date

# +
fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [3, 5]})

# remaining deferments

first_weekend = (
    pd.to_datetime(ASOF_DATE) + pd.tseries.offsets.DateOffset(day=(7 - pd.to_datetime(ASOF_DATE).dayofweek))
).date()
last_weekend = pd.to_datetime("2020-09-26").date()
pred = zzz.loc[idx[:, [ic_date.strftime("%Y-%m-%d"), "2020-09-26"]], "fhaz"].groupby(level=0).diff().dropna()

sns.distplot(pred, ax=ax[0])
tot_pctile = np.percentile(pred, q=[5, 25, 50, 75, 95])

ymin, ymax = ax[0].get_ylim()
xmin, xmax = ax[0].get_xlim()
ax[0].text(0.95 * (xmin+xmax)/2, 0.8 * ymax,
  f'Median: {tot_pctile[2]:.2%}\n'
  f'95% interval: [{tot_pctile[0]:.2%}, {tot_pctile[-1]:.2%}]', alpha=1
)

ax[0].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax[0].set_xlabel("Deferment Pct.")

# hazards

ax[1].plot(zzz_df["stop"], zzz_df["ymean"], label="Predicted")
ax[1].fill_between(
    zzz_df["stop"], zzz_df["y5"], 
    zzz_df["y95"], color="red", alpha=0.05, label="95% Interval"
)

ax[1].set(xlabel='Week', ylabel='Hazard')
ax[1].legend();
plt.tight_layout();

# +
# %%time

hier_result = make_az_data("hier", out_dict)
hier_data = hier_result.az_data

pooled_result = make_az_data("pooled", out_dict)
pooled_data = pooled_result.az_data
# -

st_orig_out = hier_result.sum_out

# +
fig, axes = plt.subplots(7, 2, figsize=(20, 40), sharex=True)

for feature, ax in zip(hier_result.b_names, axes.flatten()):
    f_d = []
    for v in states_df.state:
        f_d.append([v, covar_diff(v, feature, st_orig_out)])
    
    f_df = pd.DataFrame(f_d, columns=["state", "β_diff"])
    sns.barplot(y="β_diff", x="state", data=f_df, ax=ax)
    ax.set_title(feature)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, size=8)
    
plt.tight_layout()

# +
states = states_df.state.to_list()
b_names = hier_result.b_names
    
X = hier_result.X
A = hier_result.A
E = hier_result.E

α = hier_result.μ_a_out["mean"]
    
dp = []
for state in states:    
    try:
        β = hier_result.st_out.loc[state, "mean"]
    except KeyError:
        continue
    
    dp.append(
        pd.DataFrame(
            {"state": [state] * len(b_names), "param": b_names,
             "dp_dx": 10000 * np.array([d_cinvloglog(X, A, α, β, i) for i, v in enumerate(b_names)])}
        )
    )
dp_df = pd.concat(dp)
# -

fig, ax = plt.subplots(1,1, figsize=(10, 6))
sns.boxplot(data=dp_df, y="param", x="dp_dx", ax=ax)
ax.set_yticklabels(ax.get_yticklabels(), rotation=30, size=9)
ax.set_ylabel("Parameter")
ax.set_xlabel("dP/dX (bps)");

1-np.exp(-(5/10000)*52)

s_3_df = out_dict["hier"]["s_3_df"]

# +
# %%time

aaa = ame_vec("FL", out_dict["hier"]["s_3_df"], out_dict["hier"]["trace"], "std_original_balance", b_names)
bbb = ame_vec("NY", out_dict["hier"]["s_3_df"], out_dict["hier"]["trace"], "std_original_balance", b_names)
ccc = abs(aaa["sim"] - bbb["sim"])
# -

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.distplot(10000 * ccc, kde=False, ax=ax)
ax.axvline((10000 * ccc).mean(), color="red", label="Mean")
ax.axvline((5), color="blue", label="Cutoff")
ax.set_xlabel("AME difference (bps)")
ax.set_ylabel("Frequency")
plt.legend(loc="upper right")
print(f'{(ccc > 5/10000).sum()/ccc.shape[0]:.2%}')

# +
# %%time

compare_dict = {"hierarchical": hier_data, "pooled": pooled_data}
az.compare(compare_dict)
# -

# ### Validation: pooled

horizon_date = ASOF_DATE
asof_date = CRISIS_START_DATE
n_samples = 4000

# +
# %%time

fff, t0 = predict_survival_function(
    out_dict["pooled"]["test"], dep_var, numeric_features, 
    "pooled", out_dict["pooled"], claims_dict, asof_date, horizon_date
)

# +
_na_cum_haz = fff.groupby("orig_grade", observed=True).apply(
        na_cum_haz, "dur", "distress"
    )
_na_cum_haz.index.set_names(["orig_grade", "mob"], inplace=True)

_obs_surv = pd.merge(
    fff.groupby("orig_grade").agg(poutcome=("poutcome", np.mean)),
    _na_cum_haz.groupby(level=0).last().apply(lambda x: 1 - np.exp(-x)), 
    left_index=True, right_index=True
)    
# -

obs_surv_df = _obs_surv.reset_index()
obs_surv_df = pd.concat(
    [
        obs_surv_df,
        obs_surv_df["orig_grade"].str.split(":", expand=True).rename(columns={0: "originator", 1: "grade"})
    ], axis=1
).drop(columns=["orig_grade"])

obs_surv_df.sort_values(by=["originator", "poutcome"])

# +
g = sns.FacetGrid(
    data=obs_surv_df.sort_values(by=["originator", "poutcome"]),
    hue="grade", col="originator",
    height=5, # aspect=1.2,
    margin_titles=True
)
g.map(sns.scatterplot, "poutcome", "obs_cumhaz").add_legend(title="Grade")
for ax in g.axes:
    for i in ax:
        i.plot(obs_surv_df.poutcome, obs_surv_df.poutcome, c="k", ls="--");
        
g.set_titles("{col_name}")  # use this argument literally
g.set_axis_labels(x_var="Predicted", y_var="Observed");

# +
# %%time

do_reps = False
if do_reps:
    alist = []
    blist = []
    for i in progress_bar(range(100)):
        a_df, b_df = calibration_plot(t0, fff, i)
        alist.append(a_df)
        blist.append(b_df)

    alist_df = pd.concat(alist)
    blist_df = pd.DataFrame(blist, columns=["sim", "pct_50", "pct_95"])
    print(blist_df.describe())
else:
    alist_df, blist_df = calibration_plot(t0, fff, None)
    print(f'E50: {blist_df[1]:.2%}, E95: {blist_df[2]:.2%}') 

# +
color = "tab:red"
sns.set_style("white")

_, ax = plt.subplots(1, 1, figsize=(8, 5))
    
ax.plot(alist_df.x, alist_df.y, label="smoothed calibration curve", color=color, alpha=0.5)
ax.set_xlabel("Predicted probability of \nt ≤ %.1f mortality" % t0)
ax.set_ylabel("Observed probability of \nt ≤ %.1f mortality" % t0, color=color)
ax.tick_params(axis="y", labelcolor=color)

# plot x=y line
ax.plot(alist_df.x, alist_df.x, c="k", ls="--", alpha=0.5)
ax.legend(loc = "lower right")

# plot histogram of our original predictions
color = "tab:blue"
twin_ax = ax.twinx()
twin_ax.set_ylabel("Count of \npredicted probabilities", color=color)  # we already handled the x-label with ax1
twin_ax.tick_params(axis="y", labelcolor=color)
twin_ax.hist(fff["poutcome"], alpha=0.3, bins="sqrt", color=color)

plt.tight_layout()
sns.set()
# -

# ### Validation: hierarchical

# +
# %%time

fff, t0 = predict_survival_function(
    out_dict["hier"]["test"], dep_var, numeric_features, 
    "hier", out_dict["hier"], claims_dict, asof_date, horizon_date
)

# +
_na_cum_haz = fff.groupby("orig_grade", observed=True).apply(
        na_cum_haz, "dur", "distress"
    )
_na_cum_haz.index.set_names(["orig_grade", "mob"], inplace=True)

_obs_surv = pd.merge(
    fff.groupby("orig_grade").agg(poutcome=("poutcome", np.mean)),
    _na_cum_haz.groupby(level=0).last().apply(lambda x: 1 - np.exp(-x)), 
    left_index=True, right_index=True
)    
# -

obs_surv_df = _obs_surv.reset_index()
obs_surv_df = pd.concat(
    [
        obs_surv_df,
        obs_surv_df["orig_grade"].str.split(":", expand=True).rename(columns={0: "originator", 1: "grade"})
    ], axis=1
).drop(columns=["orig_grade"])

obs_surv_df.sort_values(by=["originator", "poutcome"])

# +
g = sns.FacetGrid(
    data=obs_surv_df.sort_values(by=["originator", "poutcome"]),
    hue="grade", col="originator",
    height=5, # aspect=1.2,
    margin_titles=True
)
g.map(sns.scatterplot, "poutcome", "obs_cumhaz").add_legend(title="Grade")
for ax in g.axes:
    for i in ax:
        i.plot(obs_surv_df.poutcome, obs_surv_df.poutcome, c="k", ls="--");
        
g.set_titles("{col_name}")  # use this argument literally
g.set_axis_labels(x_var="Predicted", y_var="Observed");

# +
# %%time

do_reps = False
if do_reps:
    alist = []
    blist = []
    for i in progress_bar(range(100)):
        a_df, b_df = calibration_plot(t0, fff, i)
        alist.append(a_df)
        blist.append(b_df)

    alist_df = pd.concat(alist)
    blist_df = pd.DataFrame(blist, columns=["sim", "pct_50", "pct_95"])
    print(blist_df.describe())
else:
    alist_df, blist_df = calibration_plot(t0, fff, None)
    print(f'E50: {blist_df[1]:.2%}, E95: {blist_df[2]:.2%}') 

# +
color = "tab:red"
sns.set_style("white")

_, ax = plt.subplots(1, 1, figsize=(8, 5))
    
ax.plot(alist_df.x, alist_df.y, label="smoothed calibration curve", color=color, alpha=0.5)
ax.set_xlabel("Predicted probability of \nt ≤ %.1f mortality" % t0)
ax.set_ylabel("Observed probability of \nt ≤ %.1f mortality" % t0, color=color)
ax.tick_params(axis="y", labelcolor=color)

# plot x=y line
ax.plot(alist_df.x, alist_df.x, c="k", ls="--", alpha=0.5)
ax.legend(loc = "lower right")

# plot histogram of our original predictions
color = "tab:blue"
twin_ax = ax.twinx()
twin_ax.set_ylabel("Count of \npredicted probabilities", color=color)  # we already handled the x-label with ax1
twin_ax.tick_params(axis="y", labelcolor=color)
twin_ax.hist(fff["poutcome"], alpha=0.3, bins="sqrt", color=color)

plt.tight_layout()
sns.set()

# +
pooled_fff, t0 = predict_survival_function(
    out_dict["pooled"]["test"], dep_var, numeric_features, 
    "pooled", out_dict["pooled"], claims_dict, asof_date, horizon_date
)

pooled_alist_df, pooled_blist_df = calibration_plot(t0, pooled_fff, None)
#print(f'E50: {blist_df[1]:.2%}, E95: {blist_df[2]:.2%}') 

hier_fff, t0 = predict_survival_function(
    out_dict["hier"]["test"], dep_var, numeric_features, 
    "hier", out_dict["hier"], claims_dict, asof_date, horizon_date
)
hier_alist_df, hier_blist_df = calibration_plot(t0, hier_fff, None)

# +
color = "tab:red"
sns.set_style("white")

_, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    
ax[0].plot(pooled_alist_df.x, pooled_alist_df.y, label="Pooled", color=color, alpha=0.5)
ax[0].set_ylim(0, 1)

ax[0].set_xlabel("Predicted probability of \nt ≤ %.1f mortality" % t0)
ax[0].set_ylabel("Observed probability of \nt ≤ %.1f mortality" % t0, color=color)
ax[0].tick_params(axis="y", labelcolor=color)

# plot x=y line
ax[0].plot(np.linspace(0,1,10), np.linspace(0,1,10), c="k", ls="--", alpha=0.5)
ax[0].legend(loc = "upper center")

# plot histogram of our original predictions
color = "tab:blue"
twin_ax = ax[0].twinx()
twin_ax.set_ylabel("Count of \npredicted probabilities", color=color)  # we already handled the x-label with ax1
twin_ax.tick_params(axis="y", labelcolor=color)
twin_ax.hist(pooled_fff["poutcome"], alpha=0.2, bins="sqrt", color=color)

# now do hierarchical
color = "tab:red"
    
ax[1].plot(hier_alist_df.x, hier_alist_df.y, label="Hierarchical", color=color, alpha=0.5)
ax[0].set_ylim(0, 1)

ax[1].set_xlabel("Predicted probability of \nt ≤ %.1f mortality" % t0)
ax[1].set_ylabel("Observed probability of \nt ≤ %.1f mortality" % t0, color=color)
ax[1].tick_params(axis="y", labelcolor=color)

# plot x=y line
ax[1].plot(np.linspace(0,1,10), np.linspace(0,1,10), c="k", ls="--", alpha=0.5)
ax[1].legend(loc = "upper center")

# plot histogram of our original predictions
color = "tab:blue"
twin_ax = ax[1].twinx()
twin_ax.set_ylabel("Count of \npredicted probabilities", color=color)  # we already handled the x-label with ax1
twin_ax.tick_params(axis="y", labelcolor=color)
twin_ax.hist(hier_fff["poutcome"], alpha=0.2, bins="sqrt", color=color)

plt.tight_layout()
sns.set()
# -

ici_df = pd.DataFrame.from_dict(
    {
        "Pooled": pooled_blist_df[1:],
        "Hierarchical": hier_blist_df[1:]
    }
)
ici_df.index = ["E50", "E95", "Mean"]
ici_df

# %watermark -a GyanSinha -n -u -v -iv -w 

# +
# %%time

hard_df = out_dict["hier"]["hard_df"]
top_states = hard_df.groupby("state").agg(
    n=("loan_id", "count")
).sort_values(by=["n"], ascending=False).iloc[:12].index.to_list()

fig, ax = plt.subplots(4, 3, figsize=(10, 10), sharex=True, sharey=True)
naf = {}
for u, v in zip(top_states, ax.flatten()):
    naf[u] = fit_na(u, hard_df, "dur", "distress")
    naf[u].plot_hazard(bandwidth=1, ax=v)
    v.set_xlabel("Weeks")
    v.set_ylabel("Hazard")
    
plt.tight_layout()
# -


