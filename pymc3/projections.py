# -*- coding: utf-8 cspell: disable -*-
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

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
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
    ASOF_DATE = datetime.date(2020, 7, 12)

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
dep_var = "defer"

zzz = out_dict["hier"]["s_3_df"]

zzz = zzz[zzz["originator"] == omap["LC"]].copy()
a_df = zzz.groupby(["state"]).agg(
    n=("loan_id", "count"), k=(dep_var, np.sum), distress=(dep_var, np.mean),
    pct_ic=("pct_ic", np.mean)
  ).reset_index().rename(columns={"distress": dep_var})

# +
g = sns.FacetGrid(
    data=a_df.reset_index(),
)
g.map(sns.regplot, "pct_ic", dep_var, ci=True)
g.ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# add annotations one by one with a loop
for line in range(0, a_df.shape[0]):
    g.ax.text(
        a_df["pct_ic"][line]+0.001, a_df[dep_var][line], a_df["state"][line], 
        horizontalalignment='left', size='medium', color='red', 
        weight='semibold', alpha=0.20
    )

g.ax.figure.set_size_inches(10, 5)
g.ax.set_xlabel("Year-over-Year pct. change")
g.ax.set_ylabel("Distress hazard");
# -

test_df = out_dict["hier"]["test"]
sub_df = make_df(test_df, dep_var, horizon_date)

# +
# %%time

aaa, zzz, _ = simulate(
    None, sub_df, dep_var, "hier", out_dict, claims_dict["chg_df"]
)
zzz.set_index(["loan_id", "edate"], inplace=True)
zzz.sort_index(inplace=True)
# -

zzz["fhaz"] = zzz.groupby(level=0).agg(chaz=("ymean", np.cumsum))["chaz"].map(lambda x: 1 - np.exp(-x))
zzz_df = zzz.groupby("start").agg(
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

ax[1].plot(zzz_df["start"], zzz_df["ymean"], label="Predicted")
ax[1].fill_between(
    zzz_df["start"], zzz_df["y5"], 
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

# +
X = hier_result.X
A = hier_result.A
U = hier_result.U
E = hier_result.E 

a_names = hier_result.a_names
b_names = hier_result.b_names

a = hier_result.a_out["mean"]
b = hier_result.b_out["mean"]
c = hier_result.trace["c_μ"].mean()
d = hier_result.trace["γ_μ"].mean()
# -

p_s_2 = out_dict["hier"]["pipe"]["p_s_2"]
non_poly_vars =  list(
    set(b.index) - set(np.array(p_s_2.named_steps.poly.colnames).ravel().tolist())
)

pooled_ppc, pooled_out_df =  predict(
    None, out_dict["pooled"]["train"], dep_var, out_dict["pooled"], 
    None, n_samples=4000, verbose=False
)
hier_ppc, hier_out_df =  predict(
    None, out_dict["hier"]["train"], dep_var, out_dict["hier"], 
    None, n_samples=4000, verbose=False
)

plot_ame(out_dict, "pooled", pooled_ppc)

plot_ame(out_dict, "hier", hier_ppc)

# +
cbeta_d = []
for i, v in enumerate(hier_result.state_index_map.state):

    indx = hier_result.state_index_map.set_index("state").loc[v, "level_0"]
    trace = hier_result.trace["c"][:, indx]
    x = pd.Series(ame_obs_var(trace, hier_ppc), name=v)
    cbeta_d.append(
        pd.DataFrame(
            np.quantile(x, q=[0.025, 0.50, 0.975]).reshape(-1, 3),
            columns=["lo", "med", "hi"]
        )
    )
    
cbeta_df = pd.DataFrame(10000 * pd.concat(cbeta_d)).reset_index()
cbeta_df["state"] = hier_result.state_index_map.state
cbeta_df.sort_values("med", ascending=False, inplace=True)

# +
fig, ax = plt.subplots(1, 1, figsize=(12, 12/1.61))

ax.vlines(cbeta_df.state, cbeta_df.lo, cbeta_df.hi, "tab:red")
ax.scatter(cbeta_df.state, cbeta_df.med, color="tab:blue")
plt.setp(ax.get_xticklabels(), ha="right", size=9, rotation=30)
ax.set_xlabel("State")
ax.set_ylabel("dP/dX (bps)");

# +
b_names = hier_result.b_names
    
X = hier_result.X
A = hier_result.A
U = hier_result.U 
E = hier_result.E

a = hier_result.a_out["mean"]
b = hier_result.b_out["mean"]
d = hier_result.trace["γ_μ"].mean()

cbeta_d = []
for i, v in enumerate(hier_result.state_index_map.state):

    indx = hier_result.state_index_map.set_index("state").loc[v, "level_0"]
    c = hier_result.trace["c"][:, indx].mean()
    dp_dx =  10000 * np.array(d_poisson(X, A, U, E, a, b, c, d, "std_pct_ic", "hier", True))
    cbeta_d.append([v, dp_dx])

cbeta_df = pd.DataFrame(cbeta_d, columns=["state", "ame"])
cbeta_df.sort_values(by=["ame"], ascending=False, inplace=True)
# -

horizon_date = ASOF_DATE
n_samples = 4000
print(az.rcParams["stats.information_criterion"])

# +
# %%time

do_loo = True
if do_loo:
    compare_dict = {"hierarchical": hier_data, "pooled": pooled_data}
    compare_tbl = az.compare(compare_dict, ic="loo")
# -

compare_tbl.style.set_precision(2)

# ### Validation

# +
# %%time

pooled_fff, t0 = predict_survival_function(
    out_dict["pooled"]["test"], dep_var, out_dict["pooled"], 
    claims_dict, ASOF_DATE
)

pooled_alist_df, pooled_blist_df = glm_calibration_plot(pooled_fff, dep_var, None)

hier_fff, t0 = predict_survival_function(
    out_dict["hier"]["test"], dep_var, out_dict["hier"], 
    claims_dict, ASOF_DATE
)
hier_alist_df, hier_blist_df = glm_calibration_plot(hier_fff, dep_var, None)
# -
pool_raw, pool_calib = calibration_data(pooled_fff, dep_var)
hier_raw, hier_calib = calibration_data(hier_fff, dep_var)

# +
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
color = "tab:red"

calib_dict = {
    "Pooled": pool_calib,
    "Mixed": hier_calib
}

i = 0
for k, v in calib_dict.items():
    max_x = v["poutcome"].max() + 0.01
    ax[i].scatter(v["poutcome"], v["observed"], color=color, label=k)
    ax[i].plot(np.linspace(0, max_x,100), np.linspace(0, max_x,100), c="k", ls="--")

    ax[i].set_xlabel("Predicted probability of \nt ≤ %.1f deferment" % t0)
    ax[i].set_ylabel("Observed probability of \nt ≤ %.1f deferment" % t0, color=color)
    ax[i].tick_params(axis="y", labelcolor=color)
    
    for j in v.itertuples():
        ax[i].text(j.poutcome - 0.001, j.observed - 0.01, j[0])
    ax[i].legend(loc = "upper left")
    
    i += 1

plt.tight_layout()
# +
# sns.set_style("white")

_, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=False)

# plot histogram of our original predictions
color = "tab:blue"
twin_ax = ax[0].twinx()
twin_ax.set_ylabel("Count of \npredicted probabilities", color=color)  # we already handled the x-label with ax1
twin_ax.tick_params(axis="y", labelcolor=color)
twin_ax.hist(pooled_fff["poutcome"], bins="sqrt", color=color, alpha=0.2)
twin_ax.grid(None)

color = "tab:red"
ax[0].plot(pooled_alist_df.x, pooled_alist_df.y, label="Pooled", color=color)

ax[0].set_xlabel("Predicted probability of \nt ≤ %.1f mortality" % t0)
ax[0].set_ylabel("Observed probability of \nt ≤ %.1f mortality" % t0, color=color)
ax[0].tick_params(axis="y", labelcolor=color)

# plot x=y line
ax[0].plot(np.linspace(0, pooled_fff.poutcome.max() + 0.01, 100),
           np.linspace(0, pooled_fff.poutcome.max() + 0.01, 100),
           c="k", ls="--")
ax[0].legend(loc = "upper center")

# now do hierarchical

# plot histogram of our original predictions
color = "tab:blue"
twin_ax = ax[1].twinx()
twin_ax.set_ylabel("Count of \npredicted probabilities", color=color)  # we already handled the x-label with ax1
twin_ax.tick_params(axis="y", labelcolor=color)
twin_ax.hist(hier_fff["poutcome"], bins="sqrt", color=color, alpha=0.2)

color = "tab:red"
ax[1].plot(hier_alist_df.x, hier_alist_df.y, label="Hierarchical", color=color)
ax[1].set_xlabel("Predicted probability of \nt ≤ %.1f mortality" % t0)
ax[1].set_ylabel("Observed probability of \nt ≤ %.1f mortality" % t0, color=color)
ax[1].tick_params(axis="y", labelcolor=color)

# plot x=y line
ax[1].plot(np.linspace(0, hier_fff.poutcome.max() + 0.01, 100),
           np.linspace(0, hier_fff.poutcome.max() + 0.01, 100),
           c="k", ls="--")
ax[1].legend(loc = "upper center")
twin_ax.grid(None)

plt.tight_layout()
# -

ici_df = pd.DataFrame.from_dict(
    {
        "Pooled": pooled_blist_df[1:],
        "Hierarchical": hier_blist_df[1:]
    }
)
ici_df.index = ["E50", "E95", "Mean"]
ici_df.style.set_precision(4)

calibrate_by_grade(pooled_fff, dep_var)

calibrate_by_grade(hier_fff, dep_var)

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
# +
fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True)

pooled_prior = pm.sample_prior_predictive(model=out_dict["pooled"]["model"], random_seed=12345)
sns.distplot([x.mean() for x in (pooled_prior["yobs"].mean(axis=1))], ax=ax[0], label="Pooled")

hier_prior = pm.sample_prior_predictive(model=out_dict["hier"]["model"], random_seed=12345)
sns.distplot([x.mean() for x in (hier_prior["yobs"].mean(axis=1))], ax=ax[1], label="Hierarchical")
for i in ax:
    i.set_xlabel("Poisson λ")
    i.set_ylabel("Frequency")
    i.axvline(out_dict["pooled"]["s_3_df"][dep_var].mean(), color="tab:red", label="Sample μ")
    i.axvline(np.array([x.mean() for x in (hier_prior["yobs"].mean(axis=1))]).mean(), color="tab:blue", label="Prior μ")
    i.legend()
# -

fig = make_ppc_plot(pooled_ppc, pooled_out_df, dep_var)

fig = make_ppc_plot(hier_ppc, hier_out_df, dep_var)

# %watermark -a GyanSinha -n -u -v -iv -w 
