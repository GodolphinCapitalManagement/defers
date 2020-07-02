# -*- coding: utf-8 cspell:disable -*-
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
import re
import joblib
import itertools
import collections
import socket

from re import search

import datetime
from datetime import timedelta
# -

import pandas as pd
import numpy as np
import sqlalchemy
import patsy
from numba import jit

import pymc3 as pm
import arviz as az
import theano.tensor as tt

from scipy.special import expit
from scipy.special import logit
import scipy.interpolate as si

# +
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import add_dummy_feature, FunctionTransformer
from sklearn.compose import ColumnTransformer

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
# -

import lifelines
import us

# +
from lifelines import NelsonAalenFitter
from lifelines.fitters.crc_spline_fitter import CRCSplineFitter
from lifelines import CoxPHFitter

from lifelines.utils import survival_table_from_events

from analytics import initdb
# -

idx = pd.IndexSlice

START_DATE = datetime.date(1995, 1, 1)
CRISIS_START_DATE = datetime.date(2020, 3, 14)

states_df = pd.DataFrame([[x.abbr, x.fips, x.name] for x in us.STATES], columns=["state", "st_code", "st_name"])
states_df = states_df.append(pd.DataFrame([["DC", "11", "Washington, DC"]], columns=["state", "st_code", "st_name"]))
states_df.sort_values(by=["st_code"], inplace=True)

ZCTA_CBSA_FILE =  "/home/gsinha/admin/db/dev/Python/projects/models/data/ZIP_CBSA_032020.xlsx"
CBSA_NAMES_FILE = "/home/gsinha/admin/db/dev/Python/projects/models/data/cbsas.csv"
ZCTA_COUNTY_URL = "https://www2.census.gov/geo/docs/maps-data/data/rel/zcta_county_rel_10.txt?#"
COUNTY_NAME_URL = "https://www2.census.gov/programs-surveys/popest/geographies/2016/all-geocodes-v2016.xlsx"

prosper_dir = "/home/gsinha/admin/db/prod/Python/projects/deals/mpl/csv/prosper/"


def chaz(x, invf="poisson"):
    ''' generic bernoulli transformation function '''
    ret_dict = {
        "poisson": np.exp(x), "cloglog": invcloglog(x), "logit": expit(x)
    }
    
    return ret_dict[invf]


def invcloglog(x):
    ''' inverse complementary log-log numpy version '''
    lower = 1e-6
    upper = 1 - 1e-6
    return lower + (upper - lower) * (1 - np.exp(-np.exp(x)))


def tinvcloglog(x):
    ''' inverse complementary log-log theano version '''
    lower = 1e-6
    upper = 1 - 1e-6
    return lower + (upper - lower) * (1 - tt.exp(-tt.exp(x)))


def hierarchical_normal(name, shape, μ=0., sigma=1.0):
    ''' creates non-centered version for pymc3 param '''
    Δ = pm.Normal('{}_Δ'.format(name), 0., 1., shape=shape)
    σ = pm.HalfNormal('{}_σ'.format(name), sigma=sigma, shape=shape)

    return pm.Deterministic(name, μ + Δ * σ)


def listdir_fullpath(d):
    ''' list files in directory d '''
    return [os.path.join(d, f) for f in os.listdir(d)]


def read_last_pos(prosper_dir):
    ''' read prosper position files'''

    lof = listdir_fullpath(prosper_dir)
    ln_df = []
    for f in lof:
        if f.find("gz") != -1:
            print("Files {} is {} MegaBytes.".format(f, os.path.getsize(f) >> 20))
            f_df = pd.read_csv(
                    f,
                    parse_dates=True,
                    dtype={
                        'ThreeDigitZip': str,
                        'ListingNumber': str, 'LoanNumber': str,
                        'LoanNoteID': str,
                    }
                )
            ln_df.append(f_df)

    pos_df = pd.concat(ln_df, ignore_index=True, sort=True)
    pos_df = pos_df[["ListingNumber", "LoanNumber", "LoanNoteID", "ThreeDigitZip"]]
    pos_df.rename(columns={
        "LoanNumber": "loan_id", "ListingNumber": "listing_id", "LoanNoteID": "note_id",
        }, inplace=True
    )
    pos_df.drop_duplicates(subset=["note_id"], keep="last", inplace=True)
    
    return pos_df


def map_zcta5_to_cbsa(ZCTA_CBSA_FILE, ZCTA_COUNTY_URL):
    ''' maps zcta5 to CSA code'''
    
    zip_cbsa_df = pd.read_excel(
        ZCTA_CBSA_FILE, dtype={"ZIP": str, "CBSA": str}, usecols=["ZIP", "CBSA"]
    ).rename(columns={"ZIP": "zcta5", "CBSA": "cbsa_code"})

    zip_county_df = pd.read_csv(
        ZCTA_COUNTY_URL, dtype={"ZCTA5": str, "STATE": str, "COUNTY": str,"GEOID": str}
    ).rename(
        columns={
            "ZCTA5": "zcta5", "STATE": "st_code", "COUNTY": "county_code",
            "GEOID": "geo_id", "POPPT": "poppt"
        }
    )[["zcta5", "st_code", "county_code", "geo_id", "poppt"]]
    zip_county_df["fips_code"] = (
        zip_county_df["st_code"] + zip_county_df["county_code"]
    )

    zip_county_df["fips_code"] = (
        zip_county_df["st_code"] + zip_county_df["county_code"]
    )

    aaa = pd.merge(zip_county_df, zip_cbsa_df, on="zcta5", how="outer")
    aaa.sort_values(by=["zcta5", "poppt"], ascending=[True, False], inplace=True)
    aaa.drop_duplicates(subset=["zcta5"], keep="first", inplace=True)
    aaa.drop(columns=["geo_id"], inplace=True)
    aaa.reset_index(drop=True, inplace=True)
    aaa["prop_zip"] = aaa["zcta5"].map(lambda x: x[:3])

    aaa.sort_values(by=["prop_zip", "poppt"], ascending=[True, False], inplace=True)
    aaa.drop_duplicates(subset=["prop_zip"], keep="first", inplace=True)
    
    return aaa[["prop_zip", "cbsa_code"]]


# +
def post_covid_poff_or_dq(start_date, asof_date):
    ''' gets loans that went DQ after the epoch start date '''

    q_stmt = sqlalchemy.sql.text(
        "select distinct on(note_id) note_id, snaptime::date as event_date, "
        "loanstatus, dayspastdue "
        "from consumer.panels "
        "where note_id in ( "
        "    select note_id from consumer.csummary "
        "    where trade_date::date = :start_date "
        "    and loanstatus = 'Current' "
        ") "
        " "
        "and (snaptime::date >= :start_date and snaptime::date <= :asof_date "
        "and loanstatus = 'Fully Paid') "
        "or (snaptime::date > :start_date and snaptime::date <= :asof_date "
        "and dayspastdue > 12 and loanstatus = 'In Grace Period') "
        "or (snaptime::date >= :start_date and snaptime::date <= :asof_date "
        "and loanstatus = 'Charged Off') "
        "order by note_id, snaptime;"
    )

    dq_df = pd.read_sql(
        q_stmt, con=initdb.gcm_engine,
        params={"start_date": start_date, "asof_date": asof_date}
    )
    dq_df["event"] = dq_df["loanstatus"]

    return dq_df


def post_covid_defers(start_date, asof_date):
    ''' get loans that went into hardship deferment ever '''

    q_stmt = sqlalchemy.sql.text(
        "select distinct on(note_id) note_id, startdate as event_date "
        "from consumer.hardship "
        "where note_id in ( "
        "    select note_id from consumer.csummary "
        "    where trade_date::date = :start_date "
        "    and loanstatus = 'Current' "
        ") "
        "and startdate >= :start_date and startdate <= :asof_date "
        "order by note_id, snaptime;"
    )

    defer_df = pd.read_sql(
        q_stmt, con=initdb.gcm_engine,
        params={"start_date": start_date, "asof_date": asof_date}
    )
    defer_df["event"] = "Defer"

    return defer_df


# -

def make_covid_df(originator, asof_date, anonymize=True):
    ''' makes crisis data set'''
    
    start_df, _ = epoch_start_data(CRISIS_START_DATE, asof_date, anonymize=anonymize)
    poff_dq_df = post_covid_poff_or_dq(CRISIS_START_DATE, asof_date)
    defer_df = post_covid_defers(CRISIS_START_DATE, asof_date)
    
    aaa = pd.concat(
        [
            poff_dq_df[~poff_dq_df["note_id"].isin(defer_df["note_id"])][["note_id", "event_date", "event"]],
            defer_df
        ], ignore_index=True
    )
    hard_df = pd.merge(
        start_df, aaa, on="note_id", how="left"
    )
    hard_df = hard_df[hard_df["originator"] == originator].reset_index(drop=True).copy()
    
    hard_df["event_date"].fillna(asof_date, inplace=True)
    hard_df["event"].fillna("Current", inplace=True)
    hard_df["event_date"] = pd.to_datetime(hard_df["event_date"])
    
    hard_df["dur"] = (hard_df["event_date"] - pd.to_datetime(CRISIS_START_DATE)).dt.days/7
    
        # get zip code for prosper
    if originator in ["PR", "II"]:
        zip_df = get_due_day(originator, asof_date)[["note_id", "prop_zip"]]
        zip_df["prop_zip"] = zip_df["prop_zip"].astype(str)
        zip_df.rename(columns={"prop_zip": "prosper_zip"}, inplace=True)
        hard_df = pd.merge(hard_df, zip_df, on="note_id", how="left")
        hard_df["prop_zip"] = hard_df["prop_zip"].where(
            ~pd.isna(hard_df["prop_zip"]), hard_df["prosper_zip"]
        )
        hard_df.drop(columns=["prosper_zip"], inplace=True)
        
    # fix a bad zip code upfront
    hard_df["state"] = np.where(
        (hard_df["prop_zip"] == "026") & (hard_df["state"] == "CA"),
        "MA", hard_df["state"]
    )
    hard_df["state"] = np.where(
        (hard_df["prop_zip"] == "946") & (hard_df["state"] == "NM"),
        "CA", hard_df["state"]
    )
    hard_df["prop_zip"] = np.where(
        (hard_df["prop_zip"].isin([''])) & (hard_df["state"] == "NC"),
        "270", hard_df["prop_zip"]
    )
    hard_df = pd.merge(hard_df, states_df, on="state", how="left")
        
    # map zip to county
    county_name_map_df = pd.read_excel(COUNTY_NAME_URL, skiprows=4, dtype=str).rename(
        columns={
            "State Code (FIPS)": "st_code", "County Code (FIPS)": "county_code", 
            "Area Name (including legal/statistical area description)": "area_name",
            "Consolidtated City Code (FIPS)": "city_code", "Place Code (FIPS)": "place_code",
            "County Subdivision Code (FIPS)": "county_subdivision_code", 
            "Summary Level": "summary_level"
        }
    )
    county_name_map_df["fips_code"] = (
        county_name_map_df["st_code"] + county_name_map_df["county_code"]
    )
    county_name_map_df = county_name_map_df.drop_duplicates(subset=["fips_code"])
    
    zip_county_map_df = pd.read_csv(
        ZCTA_COUNTY_URL, dtype={"ZCTA5": str, "STATE": str, "COUNTY": str,"GEOID": str}
    )[["ZCTA5", "STATE", "COUNTY", "GEOID", "POPPT"]].rename(
        columns={
            "ZCTA5": "zcta5", "STATE": "st_code", "COUNTY": "county_code", "GEOID": "geo_id",
            "POPPT": "poppt"
        }
    )
    zip_county_map_df["fips_code"] = (
        zip_county_map_df["st_code"] + zip_county_map_df["county_code"]
    )
    zip_county_map_df["prop_zip"] = zip_county_map_df["zcta5"].map(lambda x: x[:3])
    zip_county_map_df.sort_values(by=["prop_zip", "poppt"], ascending=[True, False], inplace=True)
    zip_county_map_df.drop_duplicates(subset=["prop_zip"], keep="first", inplace=True)

    zip_county_map_df = pd.merge(
        zip_county_map_df, county_name_map_df, 
        on=["st_code", "fips_code", "county_code"], how="left"
    )
    
    
    zip_cbsa_df = pd.read_excel(
        ZCTA_CBSA_FILE, dtype={"ZIP": str, "CBSA": str}, usecols=["ZIP", "CBSA"]
    ).rename(columns={"ZIP": "zcta5", "CBSA": "cbsa_code"})
    
    hard_df = pd.merge(hard_df, zip_county_map_df, on=["st_code", "prop_zip"], how="left")
    
        # set categoricals
    hard_df["purpose"] = pd.Categorical(
        hard_df["purpose"],
        categories=["Debt Consolidation", "Acquisition", "LifeCycle", "Other"],
        ordered=True
    )

    hard_df["home_ownership"] = np.where(
        hard_df["home_ownership"].isin(["Mortgage"]),
        "Own", hard_df["home_ownership"]
        )
    hard_df["home_ownership"] = pd.Categorical(
        hard_df["home_ownership"],
        categories=["Own", "Rent"],
        ordered=True
    )
    hard_df["employment_status"] = pd.Categorical(
        hard_df["employment_status"],
        categories=["Employed", "Self-employed", "Other"],
        ordered=True
    )
    
    hard_df["term"] = hard_df["original_term"].map(
        lambda x: "Y3" if x <= 36 else "Y5"
    )
    hard_df["term"] = pd.Categorical(
        hard_df["term"], categories=["Y3", "Y5"],
        ordered=True
    )
        
    hard_df["defer"] = np.where(
        hard_df["event"] == "Defer", True, False
    )
    hard_df["distress"] = np.where(
        ~hard_df["event"].isin(["Current", "Fully Paid", "Charged Off"]), True, False
    )
    
    #
    hard_df["dq_grp"] = np.where(
        hard_df["defer"] == True, "Covid", 
        np.where(~hard_df["loanstatus"].isin(["Current"]), "DQ", "Current")
    )
    hard_df["dq_grp"] = pd.Categorical(
        hard_df["dq_grp"], categories=["Current", "Covid", "DQ"],
        ordered=True
    )
    
    hard_df["is_dq"] = np.where(
        hard_df["loanstatus"].isin(["Current"]), "No", "Yes"
    )
    hard_df["is_dq"] = pd.Categorical(
        hard_df["is_dq"], categories=["No", "Yes"], ordered=True
    )

    w_bins = np.arange(hard_df["dur"].max() + 2)
    w_lbls = [int(x) for x in w_bins]
    hard_df["dbucket"] = pd.cut(
        hard_df["dur"], bins=w_bins, labels=w_lbls[1:], include_lowest=True, right=True
    ).astype(int)
    
    if anonymize:
        lc_grade_map = {v: ("G" + str(i)) for  i, v in enumerate(["A", "B", "C", "D", "E"])}
        lc_grade_df = pd.DataFrame.from_dict(
            lc_grade_map, orient="index", columns=["syn_grade"]).reset_index().rename(
            columns={"index": "grade"}
        )
        lc_grade_df["originator"] = "LC"

        pr_grade_map = {v: ("G" + str(i)) for  i, v in enumerate(["AA", "A", "B", "C", "D", "E", "HR"])}
        pr_grade_df = pd.DataFrame.from_dict(
            pr_grade_map, orient="index", columns=["syn_grade"]).reset_index().rename(
            columns={"index": "grade"}
        )
        pr_grade_df["originator"] = "PR"

        grade_df = pd.concat([lc_grade_df, pr_grade_df], ignore_index=True)
        
        hard_df = pd.merge(hard_df, grade_df, on=["originator", "grade"], how="left")
        hard_df.drop(columns=["grade"], inplace=True)
        hard_df.rename(columns={"syn_grade": "grade"}, inplace=True)
        
        originator_map = {"LC": "I", "PR": "II"}
        hard_df["originator"] = hard_df["originator"].map(lambda x: originator_map[x])
        
    hard_df["current_balance"] = hard_df["original_balance"] * hard_df["cur_note_amount"]/hard_df["note_amount"]
    hard_df["defer_balance"] = hard_df["current_balance"] * hard_df["defer"]
    
    return hard_df


def terminations(start_date, end_date):
    ''' get terminated loans between 2 dates'''
    
    # stack prepaid loans here
    # get prepaid loans
    q_prep = sqlalchemy.sql.text(
        "select a.note_id, snaptime::date as trade_date, balance, loanstatus, "
        "age, b.loan_id, b.note_amount, b.note_issue_date "
        "from consumer.panels a "
        "join consumer.notes as b "
        "on b.note_id = a.note_id "
        "where snaptime::date >= :start_date "
        "and snaptime::date <= :end_date "
        "and (loanstatus = 'Charged Off' or balance <= 0);"
    )
    
    poff_df = pd.read_sql(
        q_prep, con=initdb.gcm_engine, params={
            "end_date": end_date.isoformat(), 
            "start_date": start_date.isoformat(),
        }
    )
    poff_df.drop_duplicates(subset=["note_id"], keep="first", inplace=True)
    
    return poff_df


def post_covid_dq(start_date):
    ''' gets loans that went DQ after the epoch start date '''
    
    q_stmt = sqlalchemy.sql.text(
        "select distinct on(note_id) note_id, snaptime::date as dq_date "
        "from consumer.panels "
        "where snaptime::date >= :start_date "
        "and note_id in ( "
        "    select note_id from consumer.panels "
        "    where snaptime::date = :start_date "
        "    and loanstatus = 'Current' "
        ") "
        "and loanstatus not in ('Current', 'Fully Paid', 'Charged Off') "
        "order by note_id, snaptime;"
    )
    
    dq_df = pd.read_sql(
        q_stmt, con=initdb.gcm_engine, params={"start_date": start_date}
    )
    
    return dq_df


def epoch_start_data(epoch_start_date, trade_date, anonymize=False):
    ''' grabs epoch start date data and then figures out what 
        happened to the loans
    '''
    
    q_stmt = sqlalchemy.sql.text(
        "select a.note_id, a.trade_date::date, a.cur_note_amount, "
        "a.loanstatus, a.asof_age as age, b.loan_id, "
        "b.note_amount, b.note_issue_date, c.original_balance, "
        "c.origination_date, c.commitment_date, d.originator,  "
        "d.grade, d.subgrade, d.purpose, d.employment_status, "
        "d.occupation, d.prop_zip, d.prop_state as state, "
        "d.home_ownership, d.months_employed, d.stated_monthly_income, "
        "d.fico, d.dti, d.original_rate, d.original_term,  "
        "d.application_type "
        "from consumer.csummary a  "
        "join consumer.notes b on b.note_id = a.note_id  "
        "join consumer.loans c on c.loan_id = b.loan_id  "
        "join consumer.listings d on d.listing_id = c.listing_id  "
        "where a.trade_date::date = :epoch_start_date;"
        
    )
    start_df = pd.read_sql(
        q_stmt, con=initdb.gcm_engine, params={
            "epoch_start_date": epoch_start_date.isoformat(),
        }
    )
    csum_df = pd.read_sql(
        sqlalchemy.sql.text(
            "select * from consumer.csummary where trade_date::date = :tdate;"
        ),
        con=initdb.gcm_engine, params={"tdate": trade_date}
    )
    poff_df = terminations(epoch_start_date, trade_date)
    
    aaa = set(start_df.note_id.to_list())
    bbb = set(csum_df.note_id.to_list())
    ccc = list(aaa - bbb)
    
    poff_df = poff_df[poff_df.note_id.isin(ccc)].copy()
    
    return start_df, poff_df


def summary_by_group(bvar, df):
    ''' summaries by categorical variables '''
    
    def wavg(x):
        return np.nansum(
            x * df.loc[x.index, "current_balance"], axis=0
        )/np.nansum(df.loc[x.index, "current_balance"])
    
    x = df.groupby(bvar).agg(
        n=('loan_id', "count"),
        original_balance=('original_balance', sum),
        current_balance=('current_balance', sum),
        wac=('original_rate', wavg),
        age=('age', wavg),
        fico=('fico', wavg),
        term=('original_term', wavg),
        dti=('dti', wavg),
        income=('stated_monthly_income', wavg),
        distress=('distress', wavg)
    )
    
    xy = x.groupby(level=0)["current_balance"].sum().to_frame().rename(
        columns={"current_balance": "total"}
    )
    x = pd.merge(x, xy, left_index=True, right_index=True)
    x["pct"] = x["current_balance"]/x["total"]
    x.drop(columns=["total"], inplace=True)
    
    return x


def make_dummy_names(name, data_df):
    return [name + "[T." + str(x) + "]" for x in data_df[name].cat.categories[1:]]


def make_one_hot_col_names(name, data_df):
    return ["C(" + name + ")[T." + str(x) + "]" for x in data_df[name].cat.categories[1:]]


def get_due_day(originator, asofdate, anonymize=True):
    ''' gets due day for originator '''
    
    pos_csv_dir = "/home/gsinha/admin/db/prod/Python/projects/deals/mpl/csv/"
    if originator in ["LC", "I"]:
        asofdate = (asofdate + timedelta(days=-1)).strftime("%Y%m%d")
        pos_fname = (
            pos_csv_dir + "lendingclub/" + "Lending_Club_Positions[Godolphin_BL]_" + 
            asofdate + ".csv.gz"
        )
        pos_df = pd.read_csv(pos_fname)
    else:
        asofdate = asofdate.strftime("%Y%m%d")
        pos_fname = (
            pos_csv_dir + "prosper/" + "4864439LenderPacketPositions_" + 
            asofdate + ".csv.gz"
        )
        pos_df = pd.read_csv(pos_fname, dtype={"ThreeDigitZip": str})
        
    if originator in ["LC", "I"]:
        pos_df = pos_df[
            [
                "LoanID", "InvestmentAssetID", "AsOfDate", "LastPaymentDueDate"
            ]
        ].copy()
        pos_df.rename(columns={
            "LoanID": "loan_id", "InvestmentAssetID": "note_id",
            "LastPaymentDueDate": "paymentdate", "AsOfDate": "asof_date"
            }, inplace=True
        )
        pos_df["loan_id"] = pos_df["loan_id"].astype(str)
        pos_df["note_id"] = pos_df["note_id"].astype(str)
        pos_df["pmt_day"] = pd.to_datetime(pos_df["paymentdate"]).dt.day
    else:
        pos_df = pos_df[
            [
                "LoanNumber", "LoanNoteID", "AsOf", "NextPaymentDueDate",
                "ThreeDigitZip"
            ]
        ]
        pos_df.rename(columns={
            "LoanNumber": "loan_id", "LoanNoteID": "note_id",
            "NextPaymentDueDate": "paymentdate", "AsOf": "asof_date",
            "ThreeDigitZip": "prop_zip"
            }, inplace=True
        )
        pos_df["loan_id"] = pos_df["loan_id"].astype(str)
        pos_df["note_id"] = pos_df["note_id"].astype(str)
        pos_df["pmt_day"] = pd.to_datetime(pos_df["paymentdate"]).dt.day
        
    if anonymize:
        pos_df["originator"] = "I" if originator in ["LC", "I"]  else "II"
    else:
        pos_df["originator"] = originator
    
    return pos_df


class WideToLong(TransformerMixin):
    ''' creates survival data from wide (one record per loan)
        to long (multiple records per loan)

        assumes common entry date for all observations if
        entry_date is not None which is stored in global
        variable CRISIS_START_DATE
    '''
    def __init__(self, id_col, duration_col, event_col, freq="W", entry_date_col=None):
        '''
            params:
                id_col: id column for observations
                duration_col: column with event/censored durations
                event_col: censoring indicator (1 if uncensored, 0 o/w)
                freq: frequency for event time measure (Weeks=W, Days=D, Month-End=M etc.)
                    same as pandas frequency strings
                entry_date_col: entry date from which durations are measured.
        '''

        if not entry_date_col:
            self.entry_date = pd.to_datetime(CRISIS_START_DATE)
 
        self.id_col = id_col
        self.duration_col = duration_col
        self.event_col = event_col
        self.freq = freq
        if self.freq in ["W"]:
            self.period_length = 7
        elif self.freq in ["M", "CBMS"]:
            self.period_length = 30
        self.entry_date_col = entry_date_col
        
    def transform(self, X):
        X_ = lifelines.utils.to_episodic_format(
            df=X, duration_col=self.duration_col, event_col=self.event_col,
            id_col=self.id_col
        )
        X_["stop"] = X_["stop"].astype(int)
        X_["start"] = X_["start"].astype(int)
        
        if self.entry_date_col:
            X_["sdate"] = X_[self.entry_date_col] + pd.TimedeltaIndex(X_["start"], unit=self.freq)
            X_["edate"] = X_[self.entry_date_col] + pd.TimedeltaIndex(X_["stop"], unit=self.freq)
        else:
            X_["sdate"] = self.entry_date + pd.TimedeltaIndex(X_["start"], unit=self.freq)
            X_["edate"] = self.entry_date + pd.TimedeltaIndex(X_["stop"], unit=self.freq)
        
        # add duration col since lifelines drops the 
        # raw event time column
        X_ = pd.merge(X_,  X[[self.id_col, self.duration_col]], on=self.id_col, how="left")
        X_["exposure"] =  np.where(
            X_["dur"] >= X_["stop"], 1, np.maximum(1/self.period_length, X_["dur"] - X_["start"])
        )
        
        return X_
    
    def fit(self, X, y=None):
        return self


class StandardizePopulationEffects(TransformerMixin):
    def __init__(self, groups):
        ''' initialize
        '''
        
        self.groups = groups
        self.numeric_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]
        )
        
    def fit(self, X, y=None):
        ''' fit 
            params:
                X: state level snapshot of pct in low-risk
                occupations
        '''
        X_ = X[X["state"].isin(self.groups)].copy()
        self.numeric_transformer.fit(X_[["pct_high_risk"]])
        
        return self
        
    def transform(self, X):
        ''' transform '''
        X_ = X[X["state"].isin(self.groups)].copy()
        X_ = self.numeric_transformer.transform(X_[["pct_high_risk"]])
        
        return add_dummy_feature(X_)


class AddStateMacroVars(TransformerMixin):
    def __init__(self, ic_long_df):
        ''' initialize with df containing macro information 
            params:
                ic_long_df: Iniial claim data in tidy format
                risk_df: state level snapshot of pct in low-risk
                    occupations and total labor force
            
        '''
        self.ic_long_df = ic_long_df
        
    def transform(self, X):
        X_ = pd.merge(X, self.ic_long_df, on=["state", "edate"], how="left")
        X_["pct_ic"] = X_["pct_ic"].interpolate(method="ffill")
        
        return X_
    
    def fit(self, X, y=None):
        return self


class SelectOriginator(TransformerMixin):
    def __init__(self, originator):
        ''' initialize with df containing macro information '''
        
        self.originator = originator
        
    def transform(self, X, verbose=False):

        if self.originator:
            X_ = X[X["originator"].isin([self.originator])].reset_index(drop=True)
        else:
            X_ = X.copy()
            
        if self.originator is None:
            X_["grade"] = X_["originator"] + ":" + X_["grade"]
            cat_list = X_["grade"].unique()
            cat_list.sort()
            self.grade_type = pd.api.types.CategoricalDtype(
                categories=cat_list, ordered=True
            )
        elif self.originator in ["II", "PR"]:
            self.grade_type = pd.api.types.CategoricalDtype(
                categories=["AA", "A", "B", "C", "D", "E", "HR"],
                ordered=True
            )
        elif self.originator in ["I", "LC"]:
            self.grade_type = pd.api.types.CategoricalDtype(
                categories=["A", "B", "C", "D", "E"], ordered=True
            )
        else:
            grades = list(X_.grade.unique())
            grades.sort()
            if verbose:
                print(f'setting grades: {grades}')
            self.grade_type = pd.api.types.CategoricalDtype(
                categories=grades, ordered=True
            )
            
        X_["grade"] = X_["grade"].astype(self.grade_type)
               
        return X_
    
    def fit(self, X, y=None):
        return self


class Standardize(TransformerMixin, BaseEstimator):
    ''' standardizes variables '''
    
    def __init__(self, numeric_features):
        ''' initializer '''
        self.numeric_features = numeric_features
        self.newcols = ["std_" + x for x in numeric_features]
        
    def fit(self, X, y=None):
        self.numeric_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]
        )
        self.numeric_transformer.fit(X[self.numeric_features])
                
        return self
    
    def transform(self, X):
        return pd.concat(
            [
                X, 
                pd.DataFrame(
                    self.numeric_transformer.transform(X[self.numeric_features]),
                    columns=self.newcols
                )
            ], axis=1
        )


class Dummy(TransformerMixin, BaseEstimator):
    ''' creates dummies '''
    def __init__(self, categorical_features):
        ''' initializer '''
        
        self.categorical_features = categorical_features
       
        self.num_grades = None
        self.col_names = None
    
    def fit(self, X, y=None):
        
        cat_list = []
        for i in self.categorical_features:
            cat_list.append(X[i].cat.categories.to_list())
        self.num_grades = X.grade.cat.categories.shape[0]
        
        col_names = [make_dummy_names(x, X) for x in self.categorical_features]
        col_names = [item for sublist in col_names for item in sublist]
        
        self.col_names = col_names
        
        self.categorical_transformer =  Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(categories=cat_list, drop="first", sparse=False))
            ]
        )            
        self.categorical_transformer.fit(X[self.categorical_features])
                
        return self
    
    def transform(self, X):
        ''' transformer'''
        
        return pd.concat(
            [
                X, pd.DataFrame(
                    self.categorical_transformer.transform(X[self.categorical_features]),
                    columns=self.col_names, index=X.index
                )
            ], axis=1
        )


class Interaction(TransformerMixin, BaseEstimator):
    ''' creates interactions between covariates'''
    
    def __init__(self, features):
        ''' initializer '''
        if isinstance(features, list):
            self.features = features
            self.combinations = list(itertools.combinations(self.features, 2))
        else:
            raise ValueError("Needs at least two features for interactions")
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_ = X.filter(regex="grade\[").multiply(X["std_fico"], axis="index")
        X_.columns = [x + ":" + "std_fico" for x in X_.columns]
        
        return pd.concat([X, X_], axis=1)


class MakeFeatureNames(TransformerMixin):
    def __init__(self, clf):
        combined_feature_names = clf.get_feature_names()
        self.feature_names = [
            v.split("__")[1] if any([v.startswith('standardize'), v.startswith('dummy')]) 
            else v for v in clf.get_feature_names()
        ]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return pd.DataFrame(X, columns=self.feature_names)


class IntervalInterceptFeatures(TransformerMixin, BaseEstimator):
    ''' standardizes and creates dummies'''
    
    def __init__(self):
        ''' initializer '''  
        
    def fit(self, X, y=None):
        ''' fits period-specific intercepts '''
        
        self.enc = OrdinalEncoder()
        self.enc.fit(X[["start"]])
        self.max_interval = self.enc.categories_[0].max()
        self.n_intervals = self.max_interval + 1
        
        return self
    
    def transform(self, X):
        # for new data which has duration intervals longer
        # than in the training data, we cap the interval
        # index at the maximum observed in the data.
        # this has the effect of extrapolating the 
        # hazard based on the interval estimate for the 
        # last interval
        x = np.minimum(self.max_interval, X[["start"]])
        return pd.concat(
            [
                X, pd.DataFrame(
                    self.enc.transform(x).astype(int),
                    columns=["a_indx"], index=X.index
                )
            ], axis=1
        )


class HierarchicalIndex(TransformerMixin, BaseEstimator):
    ''' standardizes and creates dummies'''
    
    def __init__(self, group_vars, group_type):
        ''' initializer 
            params:
                group_vars: list of cluster variables
                group_type: one of ["nested", "crossed"]
        '''    
        self.group_vars = group_vars
        self.group_type = group_type
    
    def fit(self, X, y=None):
        X_ = X.copy()
        
        if self.group_type == "nested":
            # create state cluster IDs
            self.grp_0_index = X.groupby(self.group_vars[0]).all().reset_index().reset_index()[
                ['level_0', self.group_vars[0]]
            ]
            self.grp_0_grp_1_index = X.groupby(self.group_vars).all().reset_index().reset_index()[
                ['level_0'] + self.group_vars
            ]

            self.grp_0_grp_1_indexes_df = pd.merge(
                self.grp_0_index, self.grp_0_grp_1_index, how='inner', on=self.group_vars[0], 
                suffixes=('_0', '_01')
            ).reset_index(drop=True)
        
            self.grp_0_indexes = self.grp_0_index['level_0'].values
            self.grp_0_count = len(self.grp_0_indexes)
        
            self.grp_0_grp_1_indexes = self.grp_0_grp_1_indexes_df['level_0_0'].values
            self.grp_0_grp_1_count = len(self.grp_0_grp_1_indexes)
        else:
            self.grp_0_index = X.groupby(self.group_vars[0]).all().reset_index().reset_index()[
                ['level_0', self.group_vars[0]]
            ]
            self.grp_1_index = X.groupby(self.group_vars[1]).all().reset_index().reset_index()[
                ["level_0", self.group_vars[1]]
            ].rename(columns={"level_0": "level_1"})
            
            self.grp_0_indexes = self.grp_0_index["level_0"].values
            self.grp_0_count = len(self.grp_0_indexes)
            
            self.grp_1_indexes = self.grp_1_index["level_1"].values
            self.grp_1_count = len(self.grp_1_indexes)
        
        return self
    
    def transform(self, X):
        X_ = X.copy()
        if self.group_type == "nested":
            X_ = pd.merge(
                X_, self.grp_0_grp_1_indexes_df, on=self.group_vars, how="left"
            )
        else:
            X_ = pd.merge(
                X_, self.grp_0_index, on=self.group_vars[0], how="left"
            )
            X_ = pd.merge(
                X_, self.grp_1_index, on=self.group_vars[1], how="left"
            )
        
        return X_


def gen_labor_risk_df(fname, external_files_dir):
    risk_df = pd.read_excel(
        external_files_dir + fname
    ).iloc[1:, [0, 1, 2, 3]].reset_index(drop=True)
    
    risk_df.rename(
        columns={
            "State": "state", "High Risk Rank": "risk_rank",
            "Total employment": "employment", "% Low Risk Employment": "pct_low_risk"
        }, inplace=True
    )

    risk_df["pct_low_risk"] = risk_df["pct_low_risk"].astype(float)/100
    risk_df["pct_high_risk"] = (1.0 - risk_df["pct_low_risk"])
    risk_df.reset_index().to_feather(external_files_dir + "risk_df.feather")
    
    return risk_df


def gen_open_table_df(fname, external_files_dir):
    open_df = pd.read_csv(
        external_files_dir + fname
    ).melt(
        id_vars=["Type", "Name"], var_name="obsdate", value_name="yoy"
    ).rename(columns={"Name": "name"})

    open_df["obsdate"] = open_df["obsdate"].map(lambda x: pd.to_datetime("2020/" + x).date())

    open_df = open_df[open_df["Type"] == "state"].drop(
        columns=["Type"]
    ).rename(
        columns={"name": "st_name"}
    ).reset_index(drop=True)
    open_df["yoy"] /= 100
    
    states_df = pd.DataFrame(
        [[x.abbr, x.fips, x.name] for x in us.STATES], 
        columns=["state", "st_code", "st_name"]
    )
    states_df = states_df.append(pd.DataFrame([["DC", "11"]], columns=["state", "st_code"]))
    states_df.sort_values(by=["st_code"], inplace=True)
    
    open_df = pd.merge(open_df, states_df, on="st_name", how="left")
    open_df = open_df[~pd.isnull(open_df["state"])].reset_index(drop=True)
    
    return open_df


def cubic_linear_spline(x, kn):
    z = np.minimum((x - kn) * 0.1, 0)

    return np.hstack(
        (x, np.power(z, 2), np.power(z, 3))
    )


@jit(nopython=True)
def rcs_spline(x, kn):
    """ restriced cubic spline for age
    divided by 10
    """
    kn_ = kn * 0.1
    x_ = 0.1 * x.ravel()
    
    t_k = len(kn_) - 1
    denom = kn_[t_k] - kn_[t_k - 1]
    X = np.zeros((x.shape[0], t_k - 1))

    for j in range(X.shape[1]):
        for i in range(X.shape[0]):
            X[i, j] = (np.power(max(0.0, x_[i] - kn_[j]), 3) -
                       np.power(max(0.0, x_[i] - kn_[t_k-1]), 3) *
                       (kn_[t_k] - kn_[j]) / denom +
                       np.power(max(0.0, x_[i] - kn_[t_k]), 3) *
                       (kn_[t_k-1] - kn_[j]) / denom)

    return np.hstack((x, X))


class CubicLinearSplineFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, knots):
        self.knots = knots
        self.colnames = ["t_" + str(i) for i in np.arange(3)]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_ = pd.DataFrame(
            rcs_spline(X[["stop"]].values, kn=self.knots),
            columns=self.colnames, index=X.index
        )
        return pd.concat([X, X_], axis=1)


class BasisSplineFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, df):
        self.df = df
    
    def fit(self, X, y=None):
        formula = f"bs(stop, df={self.df}, include_intercept=True) - 1"
        design_matrix = patsy.dmatrix(
            formula, data=X, return_type="dataframe"
        )
        self.design_info = design_matrix.design_info
        self.colnames = ["t_" + str(x) for x in np.arange(design_matrix.shape[1])]
        
        return self
    
    def transform(self, X):
        X_ = patsy.build_design_matrices(
            [self.design_info], X, return_type="dataframe"
        )[0]
        X_.columns = self.colnames
        
        return pd.concat([X, X_], axis=1)


# +
"""
Robust B-Spline regression with scikit-learn
"""

class BSplineFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, knots, degree=3, periodic=False):
        self.knots = knots
        self.bsplines = get_bspline_basis(knots, degree, periodic=periodic)
        self.nsplines = len(self.bsplines)
        self.colnames = ["t_" + str(i) for i, v in enumerate(self.knots)]

    def fit(self, X, y=None):
        self.max_X = float(X[["stop"]].max())
        return self

    def transform(self, X):
        X_ = X[["stop"]].astype(float).copy()
        X_ = np.minimum(X_, self.max_X)
        
        nsamples, nfeatures = X_.shape
        features = np.zeros((nsamples, nfeatures * self.nsplines))
        for ispline, spline in enumerate(self.bsplines):
            istart = ispline * nfeatures
            iend = (ispline + 1) * nfeatures
            features[:, istart:iend] = si.splev(X_, spline)
            
        X_spl = pd.DataFrame(features, columns=self.colnames, index=X_.index)
        X_ = pd.concat([X, X_spl], axis=1)
        
        X_.set_index(keys=["loan_id", "edate"], inplace=True)
                
        return X_

def get_bspline_basis(knots, degree=3, periodic=False):
    """Get spline coefficients for each basis spline."""
    nknots = len(knots)
    y_dummy = np.zeros(nknots)

    knots, coeffs, degree = si.splrep(knots, y_dummy, k=degree,
                                      per=periodic)
    ncoeffs = len(coeffs)
    bsplines = []
    for ispline in range(nknots):
        coeffs = [1.0 if ispl == ispline else 0.0 for ispl in range(ncoeffs)]
        bsplines.append((knots, coeffs, degree))
    return bsplines


# -

def predict(originator, df, dep_var, ic_long_df, asof_date, model_type, out_dict,
            n_samples=1000, verbose=False):
    
    ''' make predictions on test data '''
    
    trace = out_dict["trace"]
    knots = out_dict["knots"]
    if model_type == "hier":
        n_samples = min(n_samples, trace["st_orig_μ"].shape[0])
    else:
        n_samples = min(n_samples, trace["b"].shape[0])
        
    p_s_1 = Pipeline(
        steps=[
            ('select_originator', SelectOriginator(originator)),
            ('wide_to_long', WideToLong(id_col="note_id", duration_col="dur", event_col=dep_var)),
            ('add_state_macro_vars', AddStateMacroVars(ic_long_df)),
        ]
    )
    p_s_1 = out_dict["pipe"]["p_s_1"]
    p_s_2 = out_dict["pipe"]["p_s_2"]
    p_s_3 = out_dict["pipe"]["p_s_3"]
    
    s_1_df = p_s_1.fit_transform(df)
    # s_1_df = p_s_1.transform(df)
    
    s_2_df = p_s_2.transform(s_1_df)
    s_3_df = p_s_3.transform(s_2_df)

    numeric_features = out_dict["numeric_features"]
    generic = out_dict["generic"]
    pop_covars = out_dict["pop_covars"]
    exp_covars = out_dict["exp_covars"]
    
    std_numeric_features = ["std_" + x for x in numeric_features]
    if generic:
        obs_covars = [
            x for x in s_3_df.columns if search("T.", x) and not search("grade", x)
        ] + std_numeric_features[:-1]
    else:
        obs_covars = [x for x in s_3_df.columns if search("T.", x)] + std_numeric_features[:-1]
    
    X = s_3_df[obs_covars + [pop_covars]].values
    E = s_3_df[exp_covars].values
    A = s_3_df["a_indx"].values
            
    a = trace["a"]
    if model_type == "pooled":
        b = trace["b"]
        xbeta = (
            np.dot(X, b[:n_samples].T) + a[:n_samples, A].T + np.log(E[:, np.newaxis])
        ).T
    else:
        st_orig_idx =  s_3_df["level_0_01"].values
        st_orig_μ = trace["st_orig_μ"][:n_samples, st_orig_idx, :]
        xbeta = (
            (st_orig_μ * X[np.newaxis, :, :]).sum(axis=2) + 
            a[:n_samples, st_orig_idx, A] + np.log(E)
        )
        
    phat = invcloglog(xbeta)
    
    return phat, s_3_df


def make_df(df, dep_var, asof_date, horizon_date):
    ''' makes projection dataframe '''

    sub_df = df[
        [
            "loan_id", "note_id", "fico", "original_balance", "note_amount", "cur_note_amount", 
            "dti", "stated_monthly_income", "age", "grade", "purpose", "employment_status",
            "term", "home_ownership", "state", dep_var, "originator", "current_balance",
            "is_dq", "st_code",
        ]
    ].copy()
    
    sub_df["dur"] = (pd.to_datetime(horizon_date) - pd.to_datetime(CRISIS_START_DATE)).days/7
    
    w_bins = np.arange(sub_df["dur"].max() + 2)
    w_lbls = [int(x) for x in w_bins]
    sub_df["dbucket"] = pd.cut(
        sub_df["dur"], bins=w_bins, labels=w_lbls[1:], include_lowest=True, right=True
    ).astype(int)
    
    return sub_df


def simulate(originator, df, dep_var, ic_long_df, asof_date, model_type, out_dict,
             numeric_features, generic=False):
    ''' make predictions for originator '''

    odict = out_dict[model_type]
    aaa, out_df = predict(
        originator, df, dep_var, ic_long_df, asof_date, model_type, odict, numeric_features,
        generic
    )
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
    
    return aaa, zzz, out_df


def save_results(originator, model_type, asof_date, model, trace,
                 hard_df, hard_df_train, hard_df_test, s_3_df, p_s_1, 
                 p_s_2, p_s_3, knots, loo, waic, numeric_features,
                 categorical_features, group_features, group_type,
                 dep_var, pop_covars, exp_covars, generic):
    
    ''' pickles results for future use '''
    
    fname = "_".join(filter(None, ("defer", originator, model_type, asof_date.isoformat())))
    fname += ".pkl"
    print(fname)
    
    pipe = {"p_s_1": p_s_1, "p_s_2":  p_s_2, "p_s_3": p_s_3}

    out_dict = {
        "model": model, "trace": trace, "pipe": pipe, "hard_df": hard_df, 
        "train": hard_df_train, "test": hard_df_test, "s_3_df": s_3_df,
        "knots": knots, "loo": loo, "waic": waic,
        "numeric_features": numeric_features, 
        "categorical_features": categorical_features,
        "group_features": group_features, "group_type": group_type,
        "dep_var": dep_var, "pop_covars": pop_covars,
        "exp_covars": exp_covars, "generic": generic,
    }
        
    return fname, out_dict


def read_results(model_type, originator, asof_date, results_dir):
    ''' read pickled results '''

    fname = results_dir + "_".join(filter(None, ["defer", originator, model_type, asof_date.isoformat()]))
    fname += ".pkl"

    with open(fname, "rb") as f:
        out_dict = joblib.load(f)

    return out_dict


def make_az_data(model_type, out_dict):
    ''' make az data instance for model '''

    model = out_dict[model_type]["model"]
    trace = out_dict[model_type]["trace"]

    s_3_df = out_dict[model_type]["s_3_df"]
    p_s_2 = out_dict[model_type]["pipe"]["p_s_2"]

    numeric_features = out_dict[model_type]["numeric_features"]
    std_numeric_features = ["std_" + x for x in numeric_features]

    categorical_features = out_dict[model_type]["categorical_features"]
    pop_covars = out_dict[model_type]["pop_covars"]
    generic = out_dict[model_type]["generic"]

    if generic:
        obs_covars = [
            x for x in s_3_df.columns if search("T.", x) and not search("grade", x)
        ] + std_numeric_features[:-1]
    else:
        obs_covars = [x for x in s_3_df.columns if search("T.", x)] + std_numeric_features[:-1]
  
    n_intervals = p_s_2.named_steps.interval.n_intervals
    X = s_3_df[obs_covars + [pop_covars]].values
    A = s_3_df["a_indx"].values
    E = s_3_df["exposure"].values

    if model_type == "pooled":
        a_names = ["t_" + str(x) for x in np.arange(n_intervals)]
        b_names = obs_covars + [pop_covars]

        az_data = az.from_pymc3(
            trace=trace, model=model, coords={'covars': b_names, 'intercepts': a_names}, 
            dims={'b': ['covars'], 'a': ['intercepts']}
        )
        
        b_out = az.summary(az_data, round_to=3, var_names=["a", "b"])
        b_out.index = a_names + b_names
        
        Result = collections.namedtuple(
            'inference', 
            'trace az_data b_out numeric_features b_names X A E n_intervals'
        )
    
        return Result(
            trace, az_data, b_out, numeric_features, b_names, X, A, E, n_intervals
        )
    else:
        state_originator_indexes_df = p_s_2.named_steps.hier_index.grp_0_grp_1_indexes_df
        state_originator_indexes_df = pd.merge(state_originator_indexes_df, states_df, on="st_code")
        state_originator_index_map = state_originator_indexes_df[
            ["state", "originator", "level_0_01"]].set_index(
            ["state", "originator"]
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
        
        az_data = az.from_pymc3(
            trace=trace, model=model,
            coords={
                "intercepts": a_names, "pop_covars": b_names,
                'st_code': index_0_to_st_code_df.state.to_list(),
                "st_orig_code": st_orig_names
            },
            dims={"g_μ": ["pop_covars"], "g_σ": ["pop_covars"], "a": ["st_orig_code", "intercepts"],
                  "st_μ": ["st_code", "pop_covars"], "st_μ_σ": ["st_code", "pop_covars"],
                  "st_orig_μ": ["st_orig_code", "pop_covars"], "μ_a": ["intercepts"]
                }
        )

        g_out = az.summary(az_data, var_names=["g_μ"], round_to=3)
        g_out.index = b_names

        st_out = az.summary(az_data, var_names=["st_μ"], round_to=3)
        st_out_idx = pd.MultiIndex.from_tuples(
            [(x, y) for x in index_0_to_st_code_df.state.to_list() for y in b_names],
            names=["state", "param"]
        )
        st_out.index = st_out_idx

        sum_out = az.summary(az_data, round_to=3, var_names=["st_orig_μ"])
        sum_out_idx = pd.MultiIndex.from_tuples(
            [(x, y) for x in st_orig_names for y in b_names],
            names=["state:originator", "param"]
        )
        sum_out.index = sum_out_idx
        
        μ_a_out = az.summary(az_data, var_names=["μ_a"], round_to=3)
        μ_a_out.index = a_names
        
        a_out = az.summary(az_data, var_names=["a"], round_to=3)
        a_out_idx = pd.MultiIndex.from_tuples(
            [(x, y) for x in st_orig_names for y in a_names],
            names=["state", "param"]
        )
        a_out.index = a_out_idx
        
        state_originator_indexes_df = p_s_2.named_steps.hier_index.grp_0_grp_1_indexes_df
        state_originator_indexes_df = pd.merge(state_originator_indexes_df, states_df, on="st_code")
        state_originator_index_map = state_originator_indexes_df[
            ["state", "originator", "level_0_01"]].set_index(
            ["state", "originator"]
        )

        Result = collections.namedtuple(
            'inference', 
            'trace az_data g_out st_out sum_out μ_a_out a_out numeric_features '
            'a_names b_names X A E state_originator_index_map n_intervals'
        )
    
        return Result(
            trace, az_data, g_out, st_out, sum_out, μ_a_out, a_out, numeric_features,
            a_names, b_names, X, A, E, state_originator_index_map, n_intervals
        )


def d_cinvloglog(X, A, α, β, n):
    ''' average marginal effect for inverse cloglog function 
        params:
            X: matrix (nsamples x nfeatures)
            β: vector of coefficients for nfeatures
            n: position in nfeatures for which derivative is required
        returns:
            vector of prob derivatives of length nsamples
            
        Based on sympy derivative calculations:
            > α, β, γ, x0, x1 = sp.symbols('α β γ x0 x1')
            > expr = 1 - sp.exp(-sp.exp(α + β*x0 + γ*x1))
            > print(expr.diff(x0))
            β*exp(x0*β + x1*γ + α)*exp(-exp(x0*β + x1*γ + α))
    '''
    
    xbeta = np.dot(X, β) + α[A]
    return β[n] * (np.exp(xbeta) * np.exp(-np.exp(xbeta))).mean()


def map_claims(originator, sum_out, ax, us_states, states_df):
    ''' map state-by-state claims βs'''
    
    dff_η = sum_out.loc[idx[[x + ":" + originator for x in states_df.state], "std_pct_ic"], :].droplevel(
      level=1).reset_index().rename(
        columns={"state:originator": "state", "mean": "value"}
    )
    dff_η["state"] = dff_η["state"].map(lambda x: x.split(":")[0])
    merged_us_states_η = pd.merge(us_states, dff_η, left_on="STUSPS", right_on="state", how="right")

    albers_epsg = 2163
    ax = us_states[~us_states["STATEFP"].isin(['02', '15'])].to_crs(epsg=albers_epsg).plot(
        ax=ax, linewidth=0.25, edgecolor='white', color='grey'
    )

    ax = merged_us_states_η[~merged_us_states_η["STATEFP"].isin(['02', '15'])].to_crs(epsg=albers_epsg).plot(
        column='value', ax=ax, cmap='viridis', 
        scheme="quantiles", legend=True,  legend_kwds={"loc": "upper center", "ncol": 3}
    )
    _ = ax.axis('off')
    
    return ax


def claims_diff(state, trace, state_originator_index_map):
    try:
        indx = state_originator_index_map.loc[state, "level_0_01"]
    except KeyError:
        return None
    else:
        aaa = trace["st_orig_μ"][:,indx, -1]
        if aaa.shape[1] > 1:
            return (aaa[:, 1] - aaa[:, 0]).mean()
        else:
            return None


def covar_diff(state, covar, st_orig_out):
    try:
        out = (
            abs(
                st_orig_out.loc[[state + ":I", state + ":II"], [covar], :].droplevel(
                    level=1
                )["mean"]
            ).diff().iloc[-1,]
        )
        return out
    except IndexError:
        return None


def aalen_hazard(asof_date, bandwidth=1, anonymize=True, verbose=False,
                 read_from_db=True):
    ''' calculates aalen hazards for snapshot taken
        on asof_date
        params:
            asof_date: datetime
        returns:
            dict of results
    '''

    if read_from_db:
        df = []
        for i in ["PR", "LC"]:
            df.append(make_covid_df(i, asof_date, anonymize))

        hard_df = pd.concat(df, sort=False, ignore_index=True)
        hard_df.to_feather("/home/gsinha/admin/src/dev/deferment/data/hard_df.feather")
    else:
        hard_df = pd.read_feather("/home/gsinha/admin/src/dev/deferment/data/hard_df.feather")

    # pick on note of many on the same loan
    hard_df.drop_duplicates(
        subset=["loan_id"], keep="first", inplace=True, ignore_index=True
    )
    if verbose:
        print(f'Obsdate: {asof_date}, Number of loans: {hard_df.shape[0]}')

    naf_dict = {}
    surv_tbl_dict = {}
    for i in ["I", "II", 'All']:
        if i not in ["I", "II"]:
            T = hard_df["dur"]
            E = hard_df["defer"]
        else:
            T = hard_df[hard_df["originator"].isin([i])]["dur"]
            E = hard_df[hard_df["originator"].isin([i])]["defer"]

        surv_tbl_dict[i] = survival_table_from_events(
                T, E, collapse=True
            ).reset_index()
        surv_tbl_dict[i]["event_at"] = surv_tbl_dict[i]["event_at"].astype(str)

        naf_dict[i] = NelsonAalenFitter()
        naf_dict[i].fit(T, event_observed=E, label=f'Originator {i}')
        
    smooth_df = pd.concat(
        [
            naf_dict["I"].smoothed_hazard_(bandwidth),
            naf_dict["I"].smoothed_hazard_confidence_intervals_(bandwidth),
            naf_dict["II"].smoothed_hazard_(bandwidth),
            naf_dict["II"].smoothed_hazard_confidence_intervals_(bandwidth),
            naf_dict["All"].smoothed_hazard_(bandwidth),
            naf_dict["All"].smoothed_hazard_confidence_intervals_(bandwidth),
        ], axis=1
    ).rename(
        columns={
            "differenced-Originator I": "Originator I",
            "differenced-Originator II": "Originator II",
            "differenced-Originator All": "All"
        }
    ).reset_index().rename(columns={"index": "dur"})
    smooth_df["asof_date"] = asof_date

    return {
        "asof_date": asof_date, "surv_tbl": surv_tbl_dict,
        "hard_df": hard_df, "bandwidth": bandwidth,
        "naf": naf_dict, "smooth_df": smooth_df,
    }


def normalize_aalen(df):
    ''' normalizes aalen smooth hazard df '''

    haz_normalized_df = pd.melt(df, id_vars=["asof_date", "dur"])

    haz_normalized_df[["originator", "ci"]] = haz_normalized_df.variable.str.split("_", expand=True)[[0, 1]]
    haz_normalized_df["ci"] = haz_normalized_df["ci"].where(~pd.isna(haz_normalized_df.ci), "median")
    haz_normalized_df.drop(columns=["variable"], inplace=True)

    return haz_normalized_df


# +
def ortho_poly_fit(x, degree = 1):
    n = degree + 1
    x = np.asarray(x).flatten()
    if(degree >= len(np.unique(x))):
            raise ValueError("'degree' must be less than number of unique points")
    xbar = np.mean(x)
    x = x - xbar
    X = np.fliplr(np.vander(x, n))
    q,r = np.linalg.qr(X)

    z = np.diag(np.diag(r))
    raw = np.dot(q, z)

    norm2 = np.sum(raw**2, axis=0)
    alpha = (np.sum((raw**2)*np.reshape(x,(-1,1)), axis=0)/norm2 + xbar)[:degree]
    Z = raw / np.sqrt(norm2)
    return Z, norm2, alpha

def ortho_poly_predict(x, alpha, norm2, degree = 1):
    x = np.asarray(x).flatten()
    n = degree + 1
    Z = np.empty((len(x), n))
    Z[:,0] = 1
    if degree > 0:
        Z[:, 1] = x - alpha[0]
    if degree > 1:
        for i in np.arange(1,degree):
            Z[:, i+1] = (x - alpha[i]) * Z[:, i] - (norm2[i] / norm2[i-1]) * Z[:, i-1]
    Z /= np.sqrt(norm2)
    return Z


# -

class OrthoPolyFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, degree=3):
        self.degree = degree
        self.colnames = ["t_" + str(i) for i, v in enumerate(np.arange(self.degree+1))]
        
    def fit(self, X, y=None):
        _, self.norm2, self.alpha = ortho_poly_fit(X["stop"].values, self.degree)
        return self
    
    def transform(self, X):
        X_ = X[["stop"]].astype(float).copy()
        Z = ortho_poly_predict(X_, self.alpha, self.norm2, self.degree)
            
        X_spl = pd.DataFrame(Z, columns=self.colnames, index=X_.index)
        X_ = pd.concat([X, X_spl], axis=1)
        
        X_.set_index(keys=["loan_id", "edate"], inplace=True)
                
        return X_


def forecast_hazard(df, dep_var, model_type, out_dict, claims_dict, asof_date, 
                    horizon_date, n_samples=4000):
    ''' generates hazard predictions '''

    sub_df = make_df(df, dep_var, asof_date, horizon_date)
    aaa, out_df =  predict(
        None, sub_df, dep_var, claims_dict["chg_df"], asof_date, 
        model_type, out_dict, n_samples=n_samples, verbose=False
    )
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
    zzz.set_index(["loan_id", "edate"], inplace=True)
    zzz.sort_index(inplace=True)

    return zzz


def ame_vec(state, s_3_df, trace, feature, b_names, n_samples=1000):
    ''' average marginal effect for inverse cloglog function 
        params:
            state: string
            s_3_df: dataframe
            trace: pymc3 trace
            n: position in nfeatures for which derivative is required
            n_samples: int
        returns:
            vector of prob derivatives of length nsamples
            
        Based on sympy derivative calculations:
            > α, β, γ, x0, x1 = sp.symbols('α β γ x0 x1')
            > expr = 1 - sp.exp(-sp.exp(α + β*x0 + γ*x1))
            > print(expr.diff(x0))
            β*exp(x0*β + x1*γ + α)*exp(-exp(x0*β + x1*γ + α))
    '''
    
    numeric_features = [
        "fico", "original_balance", "dti", "stated_monthly_income", "age", "pct_ic"
    ]
    std_numeric_features = ["std_" + x for x in numeric_features]
    pop_covars = "std_pct_ic"

    generic = True
    if generic:
            obs_covars = [
        x for x in s_3_df.columns if search("T.", x) and not search("grade", x)
        ] + std_numeric_features[:-1]
    else:
        obs_covars = [x for x in s_3_df.columns if search("T.", x)] + std_numeric_features[:-1]
        
    def input_mats(state, s_3_df):
        X_ = s_3_df[
            s_3_df.state == state
        ]
        X = X_[obs_covars + [pop_covars]].values
        A = X_["a_indx"].values
        E = X_["exposure"].values
    
        return X_, X, A, E

    X_, X, A, E = input_mats(state, s_3_df)
    
    st_orig_idx =  X_["level_0_01"].values
    st_orig_μ = trace["st_orig_μ"]
    a = trace["a"]
    
    xbeta = (
        (st_orig_μ[:n_samples, st_orig_idx, :] * X[np.newaxis, :, :]).sum(axis=2) + 
        a[:n_samples, st_orig_idx, A] + np.log(E)
    )
    n = b_names.index(feature)
    ame = pd.DataFrame(
        (st_orig_μ[:n_samples, st_orig_idx, n] * 
        (np.exp(xbeta) * np.exp(-np.exp(xbeta)))).mean(axis=1),
        columns=["sim"]
    )
    ame["state"] = state
    ame["feature"] = feature
    
    return ame


def predict_survival_function(df, dep_var, numeric_features, model_type, out_dict, 
             claims_dict, asof_date, horizon_date, generic=True, 
             n_samples=1000):
    
    ''' generates cum prob of event at horizon date '''
    
    def Surv(x):
        return np.exp(-np.cumsum(x))
    
    def ccl(p):
        return np.log(-np.log(1 - p))
    
    sub_df = make_df(df, dep_var, asof_date, horizon_date)
    t0 = sub_df["dur"].max()
    
    fff, ggg =  predict(
        None, sub_df, dep_var, claims_dict["chg_df"], asof_date, 
        model_type, out_dict, numeric_features, generic=generic, 
        n_samples=n_samples, verbose=False
    )
    fff_df = pd.DataFrame(
        fff.T, columns=["s_" + str(x) for x in np.arange(fff.shape[0])],
        index = pd.MultiIndex.from_tuples(list(zip(ggg["loan_id"], ggg["edate"])))
    )
    
    hhh = np.exp(-fff_df.groupby(level=0).apply(sum))
    hhh.index.name = "loan_id"
    
    hhh_df = pd.melt(hhh.reset_index(), id_vars="loan_id", var_name="sim", value_name="surv")
    hhh_df["sim"] = hhh_df["sim"].map(lambda x: x.split("_")[-1])
    hhh_df["poutcome"] = 1 - hhh_df["surv"] 
    hhh_df["ccl"] = ccl(hhh_df["poutcome"])

    hhh_df = pd.merge(
        hhh_df, df[["loan_id", "dur", "distress", "originator", "grade"]],
        on="loan_id", how="left"
    ).set_index(["loan_id", "sim"])
    hhh_df["distress"] = hhh_df["distress"].astype(int)
    hhh_df["orig_grade"] = hhh_df["originator"] + ":" + hhh_df["grade"]
    
    return hhh_df, t0


def calibration_plot(t0, df, sim=None):
    ''' generates calibration plots '''

    if sim:
        df_sub = df.loc[idx[:, str(sim)], :]
    else:
        df_sub = df.groupby("loan_id").agg(
            surv=("surv", np.mean), poutcome=("poutcome", np.mean),
            ccl=("ccl", np.mean), dur=("dur", np.mean),
            distress=("distress", np.mean)
        )
    
    cph = CoxPHFitter(baseline_estimation_method="spline", n_baseline_knots=5)
    cph.fit(
        df_sub[["ccl", "dur", "distress"]], 
        duration_col="dur", event_col="distress"
    )
    predictions_at_t0 = np.clip(df_sub["poutcome"], 1e-10, 1 - 1e-10)
    x = np.linspace(
        np.clip(predictions_at_t0.min() - 0.01, 0, 1), 
        np.clip(predictions_at_t0.max() + 0.01, 0, 1), 100
    )
    y = 1 - cph.predict_survival_function(
        pd.DataFrame({"ccl": np.log(-np.log(1-x))}), times=[t0]
    ).T.squeeze()
    
    fin_df = pd.DataFrame.from_dict(
        {
            "predicted": df_sub["poutcome"], 
            "observed": (1 - cph.predict_survival_function(df_sub, times=[t0])).T.squeeze()
        }
    )
    pctile = np.percentile((fin_df["predicted"] - fin_df["observed"]).abs(), q=[50, 95])
    calib_df = pd.DataFrame.from_dict({"x": x, "y": y})
    
    return calib_df, [sim, pctile[0], pctile[1], (fin_df["predicted"] - fin_df["observed"]).abs().mean()]


def na_cum_haz(df, duration_col, event_col, label="obs_cumhaz"):
    ''' returns Nelson-Aalen fitter '''
    
    T = df[duration_col]
    E = df[event_col]
    naf = NelsonAalenFitter(nelson_aalen_smoothing=True)
    naf.fit(durations=T, event_observed=E, label=label)
    
    return naf.cumulative_hazard_


def fit_na(state, df, duration_col, event_col):
    ''' returns Nelson-Aalen fitter '''
    
    if df[df.state == state].empty:
        return None
    
    T = df[df.state == state][duration_col]
    E = df[df.state == state][event_col]

    naf = NelsonAalenFitter(nelson_aalen_smoothing=True)
    naf.fit(durations=T, event_observed=E, label=state)
    
    return naf
