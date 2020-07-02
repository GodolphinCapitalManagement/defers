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

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# +
import pandas as pd
import numpy as np
import patsy
import tqdm

import sqlalchemy
# -

import lifelines
from lifelines import NelsonAalenFitter
from lifelines.utils import survival_table_from_events

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
sns.set()

plt.rcParams.update({
    "font.family": "Source Sans Pro",
    "font.serif": ["Source Sans Pro"], # use latex default serif font
    "font.sans-serif": ["Source Sans Pro"],  # use a specific sans-serif font
    "font.size": 10,
})

external_files_dir = "/home/gsinha/admin/db/dev/Python/projects/models/data/"

# +
# %%time

fred_fname = external_files_dir + "fred_data"
with open(fred_fname + ".pkl", "rb") as f:
    fred_dict = joblib.load(f)
    ic_df = fred_dict["ic_df"]
    fred_df = fred_dict["fred_df"]
    ic_date = fred_dict["ic_date"]
    w_52_pct_chg_df = fred_dict["w_52_pct_chg_df"]

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
    ASOF_DATE = datetime.date(2020, 6, 21)

print(f'As Of Date: {ASOF_DATE}')
# -

start_date = datetime.date(2020, 4, 20)
dt_range = pd.date_range(start=start_date, end=ASOF_DATE, freq="D")

# %run /home/gsinha/admin/src/dev/deferment/common.py

# +
# %%time 

backfit = False
if backfit:
    haz = []
    append = haz.append
    for v in tqdm.tqdm(dt_range):
        print(v)
        append(aalen_hazard(v)["smooth_df"])
    
    haz_df = pd.concat(haz, ignore_index=True)
    haz_normalized_df = normalize_aalen(haz_df)

# +
# Create your connection.

db_name = "/home/gsinha/admin/src/dev/deferment/data/smooth.db"
table_name = "smooth"

engine = sqlalchemy.create_engine("sqlite:///%s" % db_name, execution_options={"sqlite_raw_colnames": True})

write = False
if write:
    haz_normalized_df.to_sql(table_name, engine, index=False, if_exists="append")
else:
    haz_normalized_df = pd.read_sql_table("smooth", engine)
# -

haz_normalized_df.tail()

haz_normalized_df.head()

# +
# %%time

use_date = datetime.date(2020, 5, 13)
res = aalen_hazard(ASOF_DATE)
hard_df = res["hard_df"]
bandwidth = res["bandwidth"]
smooth_df = res["smooth_df"]
# -

hard_df.groupby(["dbucket"]).agg(defer=("defer", np.mean), n=("loan_id", "count"), k=("defer", sum))

normalize_aalen(smooth_df)


def plot_hazard(bandwidth=1):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    for i in ["I", "II", 'All']:
        ax = res["naf"][i].plot_hazard(bandwidth=bandwidth, ax=ax)

    ax.set_xlabel("Weeks since March 14th, 2020")
    ax.set_ylabel("Weekly hazard")
    plt.title(
        f'Hazard functions | bandwidth={bandwidth:.1f} | '
        f'asof date={res["asof_date"].strftime("%B %-d, %Y")}'
    );
    plt.xlim(0, hard_df.dur.max() + 1)
    
    return fig, ax


plot_hazard(1)

naf_dict = res["naf"]

print(f'Annualized CDR: {1-(1-1-(1-float(naf_dict["All"].smoothed_hazard_(1).iloc[-3:].mean()))**4)**12}')

1 - np.exp(-np.array([naf_dict["All"].smoothed_hazard_(1).iloc[-1]] * 52).sum())




