#=============================================================================#
#=============================== Acknowledgement =============================#
#=============================================================================#

# The codes are developed based on both the Lab's codes and the book 
# 'Tidy Finance with Python' by Christoph Frey, Christoph Scheuch, Stefan Voigt,
# and Patrick Weiss.
# Combining both, we want to employ the power of SQL to access the WRDS database
# directly using SQL queries, which are rewarding because it gives us a sustainable
# way to manipulate, store, and use the data with minimal chance of errors.

# for marking convenience, we save our SQLite Database on the cloud:
# https://livelancsac-my.sharepoint.com/:f:/g/personal/hoangv_lancaster_ac_uk/EvLq47Y6jM9LjYxBt2SpR7ABZcPbc78p2a0gX5F5jS3tTg?e=lkjEKv

# References:
# Bali, T.G. et al. (2016) Empirical asset pricing: the cross section of stock returns. 1st ed. New York: Wiley.
# Fama, E.F. & French, K.R. (1993) Common risk factors in the returns on stocks and bonds. Journal of Financial Economics, 33(1), 3–56.
# Fama, E.F. & French, K.R. (2015) A five-factor asset pricing model. Journal of Financial Economics, 116(1), 1–22.
# French, K.R. (2024) Current Research Returns. Available at: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html [Accessed: 25 February 2024].
# Frey, C. & Scheuch, C. (2024) Tidy Finance with Python. Available at: https://www.tidy-finance.org/python/ [Accessed: 10 April 2024].
# Novy-Marx, R. (2013) The other side of value: The gross profitability premium. Journal of financial economics, 108(1), 1–28. Elsevier B.V.
# WRDS (2024) Wharton Research Data Services. Available at: https://wrds-www.wharton.upenn.edu/pages/ [Accessed: 10 April 2024].

#=============================================================================#
#============================Task 1. Factor Modelling ========================#
#=============================================================================#

#=================================== 1.1 =====================================#
#Construct the Fama-French five-factor model following Fama and French (2015)
#and compare your replicated results
#with constructed factors (i.e., test the correlation) from the French Data Library
#The sample period is from July 1963 to December 2023.
#=================================== 1.1 =====================================#

#=============================================================================#
#============== 1.1.1: Import packages & 3 sets of DATA ======================#
#=============================================================================#

# 1.1.1 is housework, PLEASE GO STRAIGHT TO SECTION 1.1.3

# import packages here
import os
import pandas as pd
import numpy as np
import pandas_datareader as pddataread
import sqlite3
import datetime as datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import urllib.parse
import statsmodels.formula.api as smf
import scipy
from plotnine import *
from mizani.formatters import comma_format, percent_format, date_format
from mizani.breaks import date_breaks
from datetime import datetime
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
from pandas import to_datetime
from sqlalchemy import create_engine
from regtabletotext import prettify_result
from itertools import product
from statsmodels.regression.rolling import RollingOLS
from joblib import Parallel, delayed, cpu_count


# Set display options
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Auto-detect display width


print(os.getcwd())
os.chdir("/yourdirectory")

# We use SQL Web Queries to get the 3 sets data
# The motivation is to ensure consistency without conflicting type of variables


# 1st data set: Kenneth Library 

start_date_FFweb = "1963-07-01"
end_date_FFweb = "2023-12-31"


FF5_data_raw = pddataread.DataReader(
  name="F-F_Research_Data_5_Factors_2x3",
  data_source="famafrench", 
  start=start_date_FFweb, 
  end=end_date_FFweb)[0] #specify [0] to get the data structure of a list

factors_ff5_monthly = (FF5_data_raw
  .divide(100)
  .reset_index(names="month")
  .assign(month=lambda x: pd.to_datetime(x["month"].astype(str))) #create a new column with new format
  .rename(str.lower, axis="columns") #lowercase the columns' names
  .rename(columns={"mkt-rf": "mkt_excess"})
)

# set a database to store data for future access
cw602_database = sqlite3.connect(database="/cw602_database.sqlite") # create the SQLite database

(factors_ff5_monthly
  .to_sql(name="factors_ff5_monthly", 
          con=cw602_database, 
          if_exists="replace",
          index=False)
)

data_dict = {"factors_ff5_monthly": factors_ff5_monthly}

for key, value in data_dict.items():
    value.to_sql(name=key,
                 con=cw602_database, 
                 if_exists="replace",
                 index=False)

# Establish SQL_query connection to accesss WRDS to get CRSP & Compustat Data
# get this data by establishing the remote connection to WRDS 
start_date_wrds_crsp = "07/01/1963"
end_date_wrds_crsp = "12/31/2023"

password = 'yourpassword'
url_encoded_password = urllib.parse.quote_plus(password)
connection_string = (
    "postgresql+psycopg2://"
    f"your_user_name:{url_encoded_password}"
    "@wrds-pgdata.wharton.upenn.edu:9737/wrds"
)
wrds = create_engine(connection_string, pool_pre_ping=True)


# 2nd data set: CRSP Data
#three tables to get the desired data: 
    #(i) the CRSP monthly security file (msf), 
    #(ii) the identifying information (msenames), 
    #(iii) the delisting information (msedelist).
# filter to get data of interest: 
    #(i) the time windows of interest, 
    #(ii) US-listed stocks as identified via share codes (shrcd) 10 and 11, 
    #(iii) months within permno-specific start dates (namedt) and end dates (nameendt). 
    # (iv) add delisting codes and returns

# we consult this web: https://wrds-www.wharton.upenn.edu/data-dictionary/crsp_a_stock/

crsp_monthly_query = ( # this is how we interact with the query web on WRDS
  "SELECT msf.permno, msf.date, " # access Monthly Stock Files
         "date_trunc('month', msf.date)::date as month, " # filter and select columns
         "msf.ret, msf.shrout, msf.altprc, "
         "msenames.exchcd, msenames.siccd, "
         "msedelist.dlret, msedelist.dlstcd "
    "FROM crsp.msf AS msf " #main table is msf before joining
    "LEFT JOIN crsp.msenames as msenames " #left join and match columns from different tables
    "ON msf.permno = msenames.permno AND "
       "msenames.namedt <= msf.date AND "
       "msf.date <= msenames.nameendt "
    "LEFT JOIN crsp.msedelist as msedelist "
    "ON msf.permno = msedelist.permno AND "
       "date_trunc('month', msf.date)::date = "
       "date_trunc('month', msedelist.dlstdt)::date "
   f"WHERE msf.date BETWEEN '{start_date_wrds_crsp}' AND '{end_date_wrds_crsp}' "
          "AND msenames.shrcd IN (10, 11)"
)

crsp_monthly = (pd.read_sql_query( # import the query to a table
    sql=crsp_monthly_query,
    con=wrds,
    dtype={"permno": int, "exchcd": int, "siccd": int},
    parse_dates={"date", "month"})
  .assign(shrout=lambda x: x["shrout"]*1000)
)

# additional variable of market cap
crsp_monthly = (crsp_monthly
  .assign(mktcap=lambda x: abs(x["shrout"]*x["altprc"]/1000000))
  .assign(mktcap=lambda x: x["mktcap"].replace(0, np.nan))
)

# create lagged market capitalisation
# to compute value-weighted portfolio returns
# create a column of lagged market cap values by adding one month to each obs
mktcap_lag = (crsp_monthly
  .assign(
    month=lambda x: x["month"]+pd.DateOffset(months=1),
    mktcap_lag=lambda x: x["mktcap"]
  )
  .get(["permno", "month", "mktcap_lag"])
)

crsp_monthly = (crsp_monthly
  .merge(mktcap_lag, how="left", on=["permno", "month"])
)

# change exchanges codes to exhanges names
def assign_exchange(exchcd):
    if exchcd in [1, 31]: #1 or 31
        return "NYSE"
    elif exchcd in [2, 32]: #2 or 32
        return "AMEX"
    elif exchcd in [3, 33]: #3 or 33
        return "NASDAQ"
    else:
        return "Other"

crsp_monthly["exchange"] = (crsp_monthly["exchcd"]
  .apply(assign_exchange)
)

# change industry codes to industry names
def assign_industry(siccd):
    if 1 <= siccd <= 999:
        return "Agriculture"
    elif 1000 <= siccd <= 1499:
        return "Mining"
    elif 1500 <= siccd <= 1799:
        return "Construction"
    elif 2000 <= siccd <= 3999:
        return "Manufacturing"
    elif 4000 <= siccd <= 4899:
        return "Transportation"
    elif 4900 <= siccd <= 4999:
        return "Utilities"
    elif 5000 <= siccd <= 5199:
        return "Wholesale"
    elif 5200 <= siccd <= 5999:
        return "Retail"
    elif 6000 <= siccd <= 6799:
        return "Finance"
    elif 7000 <= siccd <= 8999:
        return "Services"
    elif 9000 <= siccd <= 9999:
        return "Public"
    else:
        return "Missing"

crsp_monthly["industry"] = (crsp_monthly["siccd"]
  .apply(assign_industry)
)

# construct delisting-adjusted returns and then drop them
conditions_delisting = [
    crsp_monthly["dlstcd"].isna(), 
    (~crsp_monthly["dlstcd"].isna()) & (~crsp_monthly["dlret"].isna()), #'~': NOT operator for the boolean
    crsp_monthly["dlstcd"].isin([500, 520, 580, 584]) | #'|' is the EITHER operator
        ((crsp_monthly["dlstcd"] >= 551) & (crsp_monthly["dlstcd"] <= 574)),
    crsp_monthly["dlstcd"] == 100
]
# 4 conditions result in 4 choices
choices_delisting = [
    crsp_monthly["ret"],
    crsp_monthly["dlret"],
    -0.30,
    crsp_monthly["ret"]
]

crsp_monthly = (crsp_monthly
  .assign(
    ret_adj=np.select(conditions_delisting, choices_delisting, default=-1)
  )
  .drop(columns=["dlret", "dlstcd"]) # drop after use them to form the new column 'ret_adj'
)

# compute excess returns by extracting risk-free rate from FF-library
# merge it to crsp_data to compute and then drop the elemenatary components

factors_ff5_monthly = pd.read_sql_query(
  sql="SELECT month, rf FROM factors_ff5_monthly",
  con=cw602_database,
  parse_dates={"month"}
)
  
crsp_monthly = (crsp_monthly
  .merge(factors_ff5_monthly, how="left", on="month")
  .assign(ret_excess=lambda x: x["ret_adj"]-x["rf"])
  .assign(ret_excess=lambda x: x["ret_excess"].clip(lower=-1))
  .drop(columns=["ret_adj", "rf"])
)

# drop missing returns and market caps
crsp_monthly = (crsp_monthly
  .dropna(subset=["ret_excess", "mktcap", "mktcap_lag"])
)
 
# save and store crsp to the database
(crsp_monthly
  .to_sql(name="crsp_monthly", 
          con=cw602_database, 
          if_exists="replace",
          index=False)
)

# 3rd data set: Compustat Data (firm accounting data)
# tap WRDS database's 'funda' table of annual firm-level information
# we apply these conventional filters
    #(i) we get only records in industrial data format;
    #(ii) in the standard format (i.e., consolidated information);
    #(iii) data in the desired time window.
compustat_monthly_query = (
  "SELECT gvkey, datadate, seq, ceq, at, lt, txditc, txdb, itcb,  pstkrv, "
         "pstkl, pstk, capx, oancf, sale, cogs, xint, xsga "
    "FROM comp.funda "
    "WHERE indfmt = 'INDL' "
          "AND datafmt = 'STD' "
          "AND consol = 'C' "
         f"AND datadate BETWEEN '{start_date_wrds_crsp}' AND '{end_date_wrds_crsp}'"
)

compustat_monthly = pd.read_sql_query(
  sql=compustat_monthly_query,
  con=wrds,
  dtype={"gvkey": str},
  parse_dates={"datadate"}
)

# compute book value of (1) preferred stock and equity BE and
#   the (2) operating profitability OP
#   the (3) investment ratio
# follow definitions in Kenneth French’s data library.
# set negative or zero equity to missing (see Fama and French 1992)
compustat_monthly = (compustat_monthly
  .assign(
    BE=lambda x: 
      (x["seq"].combine_first(x["ceq"]+x["pstk"])
       .combine_first(x["at"]-x["lt"])+
       x["txditc"].combine_first(x["txdb"]+x["itcb"]).fillna(0)-
       x["pstkrv"].combine_first(x["pstkl"])
       .combine_first(x["pstk"]).fillna(0))
  )
  .assign(
    BE=lambda x: x["BE"].apply(lambda y: np.nan if y <= 0 else y)
  )
  .assign(
    OP=lambda x: 
      ((x["sale"]-x["cogs"].fillna(0)- 
        x["xsga"].fillna(0)-x["xint"].fillna(0))/x["BE"])
  )
)

# keep only the last available information for each firm-year group
# using the 'tail(1)'
# NOTE: datedate refers to the date 31 December of annual report
compustat_monthly = (compustat_monthly
  .assign(year=lambda x: pd.DatetimeIndex(x["datadate"]).year)
  .sort_values("datadate")
  .groupby(["gvkey", "year"])
  .tail(1) #keep the last available information
  .reset_index()
)

# compute investment ratio INV as change in total asset between 2 consecutive fiscal years
# this follows same logic for lagged market capitalisation above
compustat_lag = (compustat_monthly
  .get(["gvkey", "year", "at"])
  .assign(year=lambda x: x["year"]+1)
  .rename(columns={"at": "at_lag"})
)

compustat_monthly = (compustat_monthly
  .merge(compustat_lag, how="left", on=["gvkey", "year"])
  .assign(inv=lambda x: x["at"]/x["at_lag"]-1)
  .assign(inv=lambda x: np.where(x["at_lag"] <= 0, np.nan, x["inv"]))
)

#save and store to the pre-established SQLITE database
(compustat_monthly
  .to_sql(name="compustat_monthly", 
          con=cw602_database, 
          if_exists="replace",
          index=False)
)

#=============================================================================#
#=========== 1.1.2: Merging the CRSP and Compustat together ==================#
#=============================================================================#

# we follow the description outlined in Bali, Engle, and Murray (2016)
# CRSP uses 'permno' for stocks
# Compustat uses 'gvkey' to identify firms. 
# WRDS database has its CRSP-Compustat Merged table (provided by CRSP). 
# keep only relevant and correct links:
    #(1) LU - Unresearched link to issue by CUSIP
    #(2) LC -  Link research complete. Standard connection between databases.
    #(3) “P” represents the primary security issue identified by Compustat, 
    #(4) “C” represents the primary security issue identified by CRSP.
    #In most applications, we only need the primary security.
# LINKDT and LINKENDDT mark the startdate and enddate during which the link is valid.
# 'CCMXPF_LNKHIST' is the primary table used for WRDS CCM web queries

# Note: currently active links have no end date, so we just enter the current date via the SQL verb CURRENT_DATE.
        # for details, consult this web: https://www.kaichen.work/?p=138
                        # and this web: https://wrds-www.wharton.upenn.edu/data-dictionary/crsp_a_ccm/ccmxpf_linktable/

ccmxpf_query = (
  "SELECT lpermno AS permno, gvkey, linkdt, "
         "COALESCE(linkenddt, CURRENT_DATE) AS linkenddt " #COALESCE to get only non-null value
    "FROM crsp.ccmxpf_linktable "
    "WHERE linktype IN ('LU', 'LC') "
          "AND linkprim IN ('P', 'C') "
          "AND usedflag = 1"
)

ccmxpf_linktable = pd.read_sql_query(
  sql=ccmxpf_query,
  con=wrds,
  dtype={"permno": int, "gvkey": str},
  parse_dates={"linkdt", "linkenddt"}
)

# create a table mapping stock identifier, firm identifier, and months
ccm_mapping = (crsp_monthly
  .merge(ccmxpf_linktable, how="inner", on="permno")
  .query("~gvkey.isnull() & (date >= linkdt) & (date <= linkenddt)")
  .get(["permno", "gvkey", "date"])
)

# merge the mapping table to the crsp_monthly
crsp_monthly = (crsp_monthly
  .merge(ccm_mapping, how="left", on=["permno", "date"])
)

# then, update the crsp_monthly dataset in our SQLite database
(crsp_monthly
  .to_sql(name="crsp_monthly", 
          con=cw602_database, 
          if_exists="replace",
          index=False)
)
# FINISHED, we are now ready to replicate FF 5-factor model


#=============================================================================#
#=========== 1.1.3: Construction of Fama-French 5-factor model ===============#
#=============================================================================#

# since we have the SQLite database
# it is very convenient, just extract the 3 datasets again

cw602_database = sqlite3.connect(
  database = "/cw602_database.sqlite"
)

crsp_monthly = (pd.read_sql_query(
    sql=("SELECT permno, gvkey, month, ret_excess, mktcap, "
         "mktcap_lag, exchange FROM crsp_monthly"),
    con=cw602_database,
    parse_dates={"month"})
  .dropna()
)

compustat_monthly = (pd.read_sql_query(
    sql="SELECT gvkey, datadate, BE, OP, inv FROM compustat_monthly",
    con=cw602_database,
    parse_dates={"datadate"})
  .dropna()
)

factors_ff5_monthly = pd.read_sql_query(
  sql="SELECT month, smb, hml, rmw, cma FROM factors_ff5_monthly",
  con=cw602_database,
  parse_dates={"month"}
)

# Fama's and French's protocols
    # (1) they form their portfolios in June of year T --> first monthly return in July,T
    # (2) they use Market Cap in June,T and shift to June,T+1
    # (3) they compute book-to-market ratio using Book Equity (BE) in Dec,T-1
    # and Market Equity (ME) WITHIN T-1 since different fiscal timeframes among firms
    # NOTE: ME not neccessarily match the same timepoint of BE

# Sorting:  (1) Size & Book-to-market ratio = 6 portfolios
          # (2) Size & Operating Profitability = 6 portfolios
          # (3) Size & Investment = 6 portfolios

size = (crsp_monthly
  .query("month.dt.month == 6") # select June and shift to July
  .assign(sorting_date=lambda x: (x["month"]+pd.DateOffset(months=1)))
  .get(["permno", "exchange", "sorting_date", "mktcap"])
  .rename(columns={"mktcap": "size"})
)

market_equity = (crsp_monthly
  .query("month.dt.month == 12") # select December and shift to July T+1
  .assign(sorting_date=lambda x: (x["month"]+pd.DateOffset(months=7)))
  .get(["permno", "gvkey", "sorting_date", "mktcap"])
  .rename(columns={"mktcap": "me"})
)

other_variables = (compustat_monthly
  .assign( # use column 'datadate' and shift to July T+1 
    sorting_date=lambda x: (pd.to_datetime(
      (x["datadate"].dt.year+1).astype(str)+"0701")
    )
  ) # after bring BE and ME to the same timepoint, calculate B-t-M ratio
  .merge(market_equity, how="inner", on=["gvkey", "sorting_date"]) # merge to one table
  .assign(bm=lambda x: x["BE"]/x["me"])
  .get(["permno", "sorting_date", "me", "bm", "OP", "inv"])
)

# merge Size to other variables' table
sortings = (size
  .merge(other_variables, how="inner", on=["permno", "sorting_date"])
  .dropna()
  .drop_duplicates(subset=["permno", "sorting_date"])
 )


# we define a function called assign_portfolios() to independently sort
# NYSE-specific breakpoints, 1 breakpoint (median) for Size
# 2 breakpoints (30th and 70th percentile) for the other three factors

def assign_portfolios(data, sorting_variable, percentiles):
    
    breakpoints = (data
      .query("exchange == 'NYSE'")
      .get(sorting_variable)
      .quantile(percentiles, interpolation="linear")
    )
    breakpoints.iloc[0] = -np.Inf #less than or equal to the first percentile will be included in the 1st portfolio
    breakpoints.iloc[breakpoints.size-1] = np.Inf #greater than or equal to the last percentile will be included in the 3rd portfolio
    
    assigned_portfolios = pd.cut( #assign labels to data sorted by breakpoints
      data[sorting_variable],
      bins=breakpoints, # bin range
      labels=pd.Series(range(1, breakpoints.size)),
      include_lowest=True,
      right=False
    )
    
    return assigned_portfolios

# get the assigned portfolios
portfolios = (sortings
  .groupby("sorting_date")
  .apply(lambda x: x
    .assign(
      portfolio_size=assign_portfolios(x, "size", [0, 0.5, 1])
    )
  )
  .reset_index(drop=True)
  .groupby(["sorting_date", "portfolio_size"])
  .apply(lambda x: x
    .assign(
      portfolio_bm=assign_portfolios(x, "bm", [0, 0.3, 0.7, 1]),
      portfolio_op=assign_portfolios(x, "OP", [0, 0.3, 0.7, 1]),
      portfolio_inv=assign_portfolios(x, "inv", [0, 0.3, 0.7, 1])
    )
  )
  .reset_index(drop=True)
  .get(["permno", "sorting_date", 
        "portfolio_size", "portfolio_bm",
        "portfolio_op", "portfolio_inv"])
)

# merge sorted portfolios table to stock-return data table
portfolios = (crsp_monthly
  .assign(
    sorting_date=lambda x: (pd.to_datetime(
      x["month"].apply(lambda x: str(x.year-1)+ # sort the return data to July,T-1 or July,T depending on the month
        "0701" if x.month <= 6 else str(x.year)+"0701")))
  )
  .merge(portfolios, how="inner", on=["permno", "sorting_date"])
)

# start to construct the factor
SMB_BM = (portfolios
  .groupby(["portfolio_size", "portfolio_bm", "month"])
  .apply(lambda x: pd.Series({
      "ret": np.average(x["ret_excess"], weights=x["mktcap_lag"])
    })
  )
  .reset_index()
)

SMB_OP = (portfolios
  .groupby(["portfolio_size", "portfolio_op", "month"])
  .apply(lambda x: pd.Series({
      "ret": np.average(x["ret_excess"], weights=x["mktcap_lag"])
    })
  )
  .reset_index()
)

SMB_INV = (portfolios
  .groupby(["portfolio_size", "portfolio_inv", "month"])
  .apply(lambda x: pd.Series({
      "ret": np.average(x["ret_excess"], weights=x["mktcap_lag"])
    })
  )
  .reset_index()
)

HML_factor = (SMB_BM
  .groupby("month")
  .apply(lambda x: pd.Series({
    "hml_replicated": (
      x["ret"][x["portfolio_bm"] == 3].mean() - 
        x["ret"][x["portfolio_bm"] == 1].mean())})
  )
  .reset_index()
)

RMW_factor = (SMB_OP
  .groupby("month")
  .apply(lambda x: pd.Series({
    "rmw_replicated": (
      x["ret"][x["portfolio_op"] == 3].mean() - 
        x["ret"][x["portfolio_op"] == 1].mean())})
  )
  .reset_index()
)

INV_factor = (SMB_INV
  .groupby("month")
  .apply(lambda x: pd.Series({
    "cma_replicated": (
      x["ret"][x["portfolio_inv"] == 1].mean() - 
        x["ret"][x["portfolio_inv"] == 3].mean())})
  )
  .reset_index()
)

SMB_factor = (
  pd.concat(
    [SMB_BM, SMB_OP, SMB_INV], 
    ignore_index=True
  )
  .groupby("month")
  .apply(lambda x: pd.Series({
    "smb_replicated": (
      x["ret"][x["portfolio_size"] == 1].mean() - 
        x["ret"][x["portfolio_size"] == 2].mean())})
  )
  .reset_index()
)

# put all replicated factors into 1 table
all_factors = (SMB_factor
  .merge(HML_factor, how="outer", on="month")
  .merge(RMW_factor, how="outer", on="month")
  .merge(INV_factor, how="outer", on="month")
)

all_factors = (all_factors
  .merge(factors_ff5_monthly, how="inner", on="month")
  .round(4)
)

# TEST REPLICATED FACTORS, first extract the all_factors AT THE END

SMB_corr = all_factors['smb'].corr(all_factors['smb_replicated'])
HML_corr = all_factors['hml'].corr(all_factors['hml_replicated'])
RMW_corr = all_factors['rmw'].corr(all_factors['rmw_replicated'])
INV_corr = all_factors['cma'].corr(all_factors['cma_replicated'])

print(SMB_corr, HML_corr, RMW_corr, INV_corr)  # 0.99 for size factor and 0.97 for value factor


# SUCCEEDED, save to the database

(all_factors
  .to_sql(name="all_factors", 
          con=cw602_database, 
          if_exists="replace",
          index=False)
)

all_factors = pd.read_sql_query(
    sql="SELECT * "
        "FROM all_factors ",
    con=cw602_database,
    parse_dates={"caldt"}
)


#=================================== 1.2 ====================================#
#Each group should select two US domestic equity mutual funds, 
#including one large value and one small growth fund. 
#You should obtain monthly returns for these two funds from Jan 2010 to Dec 2023 from CRSP.
#Then regress the monthly returns of you selected funds (as dependent variable) 
#on Fama-French five factors (as independent variables) and interpret your regression results.
#Note that you can find fund categories from website such as “US News Money”. [15 marks]
#=================================== 1.2 ====================================#

# https://wrds-www.wharton.upenn.edu/documents/410/CRSP_MFDB_Guide.pdf
# https://wrds-www.wharton.upenn.edu/data-dictionary/crsp_q_mutualfunds/


#=============================================================================#
#=========== 1.2.1: Set-up: Fama-French Regression on Mutual Funds ===========#
#=============================================================================#


#FMILX is listed as: #1 in Large Value Mutual Fund
#NEAGX is listed as: #2 in Small Growth Mutual Fund

start_date_wrds_crsp2 = "2010/01/01/"
end_date_wrds_crsp2 = "2023/12/31"

# run 82th-89th code first to access the WRDS
# import the data of crsp_q_mutualfunds and crsp_fundno into the database
crsp_fundmonthly_query = (
    "SELECT mfinfor.crsp_fundno, mfinfor.fund_name, "
    "mfret.caldt, mfret.mret "
    "FROM crsp_q_mutualfunds.fund_hdr AS mfinfor "
    "INNER JOIN crsp_q_mutualfunds.monthly_returns AS mfret "
    "ON mfinfor.crsp_fundno = mfret.crsp_fundno "
    f"WHERE mfret.caldt BETWEEN '{start_date_wrds_crsp2}' AND '{end_date_wrds_crsp2}' "
    "AND mfinfor.ticker IN ('FMILX', 'NEAGX')"
)

crsp_fundmonthly = (pd.read_sql_query( # import the query to a table
    sql=crsp_fundmonthly_query,
    con=wrds,
    dtype={"crsp_fundno": int},
    parse_dates={"caldt"})
)

# connect to the database to save
cw602_database = sqlite3.connect(
  database = "/cw602_database.sqlite"
)

# save mutual funds data to the database
(crsp_fundmonthly
  .to_sql(name="crsp_fundmonthly", 
          con=cw602_database, 
          if_exists="replace",
          index=False)
)

#=============================================================================#
#============== 1.2.2: Fama-French Regression on Mutual Funds ================#
#=============================================================================#

crsp_fundmonthly = pd.read_sql_query(
    sql="SELECT * "
        "FROM crsp_fundmonthly ",
    con=cw602_database,
    parse_dates={"caldt"}
)

# Convert "caldt" to datetime format
crsp_fundmonthly["caldt"] = pd.to_datetime(crsp_fundmonthly["caldt"])

# Format "caldt" to the desired format with day set to '01'
crsp_fundmonthly["caldt"] = crsp_fundmonthly["caldt"].dt.strftime('%Y-%m-01')

# Separate data for crsp_fundno 11948 (large value)
fundmonthly_largevalue = crsp_fundmonthly[crsp_fundmonthly['crsp_fundno'] == 11948].copy()
# Separate data for crsp_fundno 22000 (small growth)
fundmonthly_smallgrowth = crsp_fundmonthly[crsp_fundmonthly['crsp_fundno'] == 22000].copy()

rep_factors = pd.read_sql_query(
    sql="SELECT month, smb_replicated, hml_replicated, rmw_replicated, cma_replicated "
        "FROM all_factors "
        "WHERE all_factors.month >= '2009/12/01' ", #get same 168 observations
    con=cw602_database,
    parse_dates={"month"}
)

mkt_excess = pd.read_sql_query(
    sql="SELECT month, mkt_excess "
        "FROM factors_ff5_monthly "
        "WHERE factors_ff5_monthly.month >= '2009/12/01' ",
    con=cw602_database,
    parse_dates={"month"}
)

fundmonthly_largevalue["caldt"] = pd.to_datetime(fundmonthly_largevalue['caldt'])
fundmonthly_smallgrowth["caldt"] = pd.to_datetime(fundmonthly_smallgrowth['caldt'])
rep_factors['month'] = pd.to_datetime(rep_factors['month'])
mkt_excess['month'] = pd.to_datetime(mkt_excess['month'])

# Merge the DataFrames
ff5_regression_largevalue = pd.merge(
    pd.merge(fundmonthly_largevalue, rep_factors, left_on='caldt', right_on='month', how='inner'),
    mkt_excess, left_on='caldt', right_on='month', how='inner'
)

ff5_regression_smallgrowth = pd.merge(
    pd.merge(fundmonthly_smallgrowth, rep_factors, left_on='caldt', right_on='month', how='inner'),
    mkt_excess, left_on='caldt', right_on='month', how='inner'
)

beta_est_largevalue = (smf.ols(
    formula="mret ~ mkt_excess + smb_replicated + hml_replicated + rmw_replicated + cma_replicated",
    data=ff5_regression_largevalue)
  .fit()
)

beta_est_smallgrowth = (smf.ols(
    formula="mret ~ mkt_excess + smb_replicated + hml_replicated + rmw_replicated + cma_replicated",
    data=ff5_regression_smallgrowth)
  .fit()
)

prettify_result(beta_est_largevalue)
prettify_result(beta_est_smallgrowth)


# save to the database
(ff5_regression_largevalue
  .to_sql(name="ff5_regression_largevalue", 
          con=cw602_database, 
          if_exists="replace",
          index=False)
)

(ff5_regression_smallgrowth
  .to_sql(name="ff5_regression_smallgrowth", 
          con=cw602_database, 
          if_exists="replace",
          index=False)
)

beta_est_largevalue = pd.DataFrame(beta_est_largevalue)

(beta_est_largevalue
  .to_sql(name="beta_est_largevalue", 
          con=cw602_database, 
          if_exists="replace",
          index=False)
)

(beta_est_smallgrowth
  .to_sql(name="beta_est_smallgrowth", 
          con=cw602_database, 
          if_exists="replace",
          index=False)
)


#=================================== 1.3 ====================================#
#Perform Fama-MacBeth regression on Fama-French five factors 
#using the universe of stocks listed on NYSE, AMEX and NASDAQ from 1963 to 2023
#and briefly interpret the results. [15 marks]
#=================================== 1.3 ====================================#

# connect to the database again

cw602_database = sqlite3.connect(
  database = "/cw602_database.sqlite"
)

crsp_monthly = pd.read_sql_query(
  sql="SELECT permno, month, industry, ret_excess FROM crsp_monthly",
  con=cw602_database,
  parse_dates={"month"}
)

factors_ff5_monthly = pd.read_sql_query(
  sql="SELECT month, mkt_excess FROM factors_ff5_monthly",
  con=cw602_database,
  parse_dates={"month"}
)

# merge the factors_ff5_monthly to the crsp_monthly
crsp_monthly = (crsp_monthly
  .merge(factors_ff5_monthly, how="left", on="month")
)

#=============================================================================#
#========================= 1.3.1: Beta Initialisation ========================#
#=============================================================================#

WE = 60 # we follow Fama-Macbeth to use 5-year rolling
min_WE = 48

# customise a function based on Joblib package and tidy-finance : https://github.com/tidy-finance/website/blob/main/python/beta-estimation.qmd
def rolling_capm_beta_setup(crsp_monthly, factors_ff5_monthly, WE):
        
    """Prepare environment for Rolling CAPM Beta Estimation"""
    
    rolling_permo = (crsp_monthly
      .dropna()
      .groupby("permno")["permno"]
      .count()
      .reset_index(name="counts")
      .query(f"counts > {WE}+1")
    )

    unique_permno = crsp_monthly["permno"].unique()
    unique_month = factors_ff5_monthly["month"].unique()

    all_permos = pd.DataFrame(
      product(unique_permno, unique_month),  #we use the 'product' to produce the tuple (permo, 'Jan')
      columns=["permno", "month"] #then use that tuple to create 2 columns
    )

    # firms are sometimes delisted and listed again so we need to filter out the missing observations
    permno_filter = (crsp_monthly
      .merge(rolling_permo, how="inner", on="permno")
      .groupby(["permno"])
      .aggregate(first_month=("month", "min"),
                 last_month=("month", "max"))
      .reset_index()
    )
    
    # combine all above variables into a new table
    clean_beta_setup = (all_permos
      .merge(crsp_monthly.get(["permno", "month", "ret_excess"]), 
             how="left", on=["permno", "month"])
      .merge(permno_filter, how="left", on="permno")
      .query("(month >= first_month) & (month <= last_month)")
      .drop(columns=["first_month", "last_month"])
      .merge(crsp_monthly.get(["permno", "month", "industry"]),
             how="left", on=["permno", "month"])
      .merge(factors_ff5_monthly, how="left", on="month")
    )
    
    return clean_beta_setup, rolling_permo
    
clean_beta_setup, rolling_permo = rolling_capm_beta_setup(crsp_monthly, factors_ff5_monthly, WE)


# use function from tidy-finance (Github) to estimate rolling betas
def rolling_capm_beta_estimation(permno, group):
    
    """Calculate rolling CAPM estimation"""
    
    if "date" in group.columns:
      group = group.sort_values(by="date")
    else:
      group = group.sort_values(by="month")

    beta_values = (RollingOLS.from_formula(
        formula="ret_excess ~ mkt_excess",
        data=group,
        window=WE,
        min_nobs=min_WE,
        missing="drop"
      )
      .fit()
      .params.get("mkt_excess")
    )
    
    result = pd.DataFrame(beta_values)
    result.columns = ["beta"]
    result["month"] = group["month"].values
    result["permno"] = permno
    
    try:
      result["date"] = group["date"].values
      result = result[
        (result.groupby("month")["date"].transform("max")) == result["date"]
      ]
    except(KeyError):
      pass

    result = result.dropna()
    result = result.rename(columns={"beta":"beta_monthly"})

    return result


n_cores = cpu_count()-1

permno_groups = (clean_beta_setup
  .merge(rolling_permo, how="inner", on="permno")
  .groupby("permno", group_keys=False)
)
    

# 'bottleneck' library must be 1.3.6 or newer to run smoothly the 'joblib' package
# pip install --upgrade bottleneck

initial_beta_monthly = (
  pd.concat(
    Parallel(n_jobs=n_cores)
    (delayed(rolling_capm_beta_estimation)(name, group)
    for name, group in permno_groups)
  )
  .dropna()
  .rename(columns={"beta": "beta_monthly"})
)


# SUCCEEDED, save to the cw602_database

(initial_beta_monthly.to_sql(
    name="initial_beta_monthly",
    con=cw602_database,
    if_exists="replace",
    index=False
    )
)

#=============================================================================#
#========================= 1.3.2: Fama-Macbeth Setup  ========================#
#=============================================================================#

cw602_database = sqlite3.connect(
  database = "/cw602_database.sqlite"
)

crsp_monthly = pd.read_sql_query(
  sql="SELECT * FROM crsp_monthly",
  con=cw602_database,
  parse_dates={"month"}
)

compustat_monthly = pd.read_sql_query(
  sql="SELECT * FROM compustat_monthly",
  con=cw602_database,
  parse_dates={"datadate"}
)

beta_monthly = pd.read_sql_query(
  sql="SELECT * FROM initial_beta_monthly",
  con=cw602_database,
  parse_dates={"month"}
)

#=============================================================================#
#========================= 1.3.3: Fama-Macbeth Setup  ========================#
#=============================================================================#

def fama_macbeth_setup(crsp_monthly, compustat_monthly, beta_monthly):
    
    """Prepare environment for Fama-Macbeth Regression"""
    
    firm_characteristics = (compustat_monthly
      .assign(month=lambda x: x["datadate"].dt.to_period("M").dt.to_timestamp()) # transform 'datadate' to Timestamp before merge
      .merge(crsp_monthly, how="left", on=["gvkey", "month"], )
      .merge(beta_monthly, how="left", on=["permno", "month"])
      .assign( # after combine into 1 table, then compute the "bm", "log_mktcap", "and "sorting_date"
        bm=lambda x: x["BE"]/x["mktcap"],
        log_mktcap=lambda x: np.log(x["mktcap"]),
        op=lambda x: x["OP"],
        inv=lambda x: x["inv"],
        sorting_date=lambda x: x["month"]+pd.DateOffset(months=6)
      )
      .get(["gvkey", "bm", "log_mktcap", "op", "inv", "beta_monthly", "sorting_date"])
      .rename(columns={"beta_monthly": "beta"})
    )
    
    fama_macbeth_setup = (crsp_monthly
      .merge(firm_characteristics, 
             how="left",
             left_on=["gvkey", "month"], right_on=["gvkey", "sorting_date"])
      .sort_values(["month", "permno"])
      .groupby("permno")
      .apply(lambda x: x.assign(
          beta=x["beta"].fillna(method="ffill"), # fill NaN with last observed non-null value
          bm=x["bm"].fillna(method="ffill"),
          log_mktcap=x["log_mktcap"].fillna(method="ffill"),
          op=x["op"].fillna(method="ffill"),
          inv=x["inv"].fillna(method="ffill")
        )
      )
      .reset_index(drop=True)  
    )
    
    fama_macbeth_setup_lagged = (fama_macbeth_setup
      .assign(month=lambda x: x["month"]-pd.DateOffset(months=1))
      .get(["permno", "month", "ret_excess"])
      .rename(columns={"ret_excess": "ret_excess_lead"})
    )
    
    fama_macbeth_setup = (fama_macbeth_setup
      .merge(fama_macbeth_setup_lagged, how="left", on=["permno", "month"])
      .get(["permno", "month", "ret_excess_lead", "beta", "log_mktcap", "bm", "op", "inv"])
      .dropna()
    )
    return fama_macbeth_setup

# get the fama_macbeth_table
fama_macbeth_table = fama_macbeth_setup(crsp_monthly, compustat_monthly, beta_monthly)


#=============================================================================#
#============= 1.3.4: Fama-Macbeth Cross-sectional Regression  ===============#
#=============================================================================#

def fama_macbeth_cross_reg(fama_macbeth_table):
    
    """Run Fama-Macbeth Cross-sectional Regression"""
    
    # Multivariate Fama-Macbeth regression
    risk_factor_coefs = (fama_macbeth_table
      .groupby("month")
      .apply(lambda x: smf.ols( #run cross-factor regression to get risk-factor coefficients
          formula="ret_excess_lead ~ beta + log_mktcap + bm + op + inv", 
          data=x
        ).fit()
        .params
      )
      .reset_index()
    )
    
    # Univariate Fama-Macbeth regression
    mkt_risk_coef = (fama_macbeth_table
      .groupby("month")
      .apply(lambda x: smf.ols( #run cross-factor regression to get risk-factor coefficients
          formula="ret_excess_lead ~ beta", 
          data=x
        ).fit()
        .params
      )
      .reset_index()
    )
    
    size_premium_coef = (fama_macbeth_table
      .groupby("month")
      .apply(lambda x: smf.ols( #run cross-factor regression to get risk-factor coefficients
          formula="ret_excess_lead ~ log_mktcap", 
          data=x
        ).fit()
        .params
      )
      .reset_index()
    )
    
    value_premium_coef = (fama_macbeth_table
      .groupby("month")
      .apply(lambda x: smf.ols( #run cross-factor regression to get risk-factor coefficients
          formula="ret_excess_lead ~ bm", 
          data=x
        ).fit()
        .params
      )
      .reset_index()
    )
    
    op_premium_coef = (fama_macbeth_table
      .groupby("month")
      .apply(lambda x: smf.ols( #run cross-factor regression to get risk-factor coefficients
          formula="ret_excess_lead ~ op", 
          data=x
        ).fit()
        .params
      )
      .reset_index()
    )
    
    inv_premium_coef = (fama_macbeth_table
      .groupby("month")
      .apply(lambda x: smf.ols( #run cross-factor regression to get risk-factor coefficients
          formula="ret_excess_lead ~ inv", 
          data=x
        ).fit()
        .params
      )
      .reset_index()
    )

    return risk_factor_coefs, mkt_risk_coef, size_premium_coef, value_premium_coef, op_premium_coef, inv_premium_coef

# get the results of both multivariate and univariate Fama-Macbeth Regression
risk_factor_coefs, mkt_risk_coef, size_premium_coef, value_premium_coef, op_premium_coef, inv_premium_coef = fama_macbeth_cross_reg(fama_macbeth_table)

#=============================================================================#
#=============== 1.3.5: Fama-Macbeth Time Series Averaging ===================#
#=============================================================================#

def fama_macbeth_final_reg(risk_factor_coefs, 
                           mkt_risk_coef, 
                           size_premium_coef, 
                           value_premium_coef,
                           op_premium_coef,
                           inv_premium_coef):
    
    """Run Fama-Macbeth Final Time-series Regression"""
    
    # Multivariate Fama-Macbeth Time Series Averaging
    
    risk_premiums = (risk_factor_coefs
       #create a melted Dataframe with 'month' column; 'factor' column to indicate which factor; 'estimate' column to show the coefficient
      .melt(id_vars="month", var_name="factor", value_name="estimate")
      .groupby("factor")["estimate"] #group by same factor and select the estimates to be applied the following calculation
      .apply(lambda x: pd.Series({
          "risk_premium": 100*x.mean(), #average across the time dimension to get expected value
          "t_statistic": x.mean()/x.std()*np.sqrt(len(x)),
          "standard_error": x.std() / np.sqrt(len(x))  # Calculate standard error
        })
      )
      .reset_index()
      .pivot(index="factor", columns="level_1", values="estimate") #pivot the table
      .reset_index()
    )
    
    p_values_risk_premiums = (risk_factor_coefs
      .melt(id_vars="month", var_name="factor", value_name="estimate")
      .groupby("factor")
      .apply(lambda x: (
          smf.ols("estimate ~ 1", x)
          .fit()
          .pvalues["Intercept"]
        )
      )
      .reset_index()
    )
    
    Newey_West_risk_premiums = (risk_factor_coefs
      .melt(id_vars="month", var_name="factor", value_name="estimate")
      .groupby("factor")
      .apply(lambda x: (
          x["estimate"].mean()/ 
            smf.ols("estimate ~ 1", x)
            .fit(cov_type="HAC", cov_kwds={"maxlags": 6}).bse
        )
      )
      .reset_index()
      .rename(columns={"Intercept": "t_statistic_newey_west"})
    )
    
    multi_fama_macbeth = (risk_premiums
                          .merge(p_values_risk_premiums, on="factor")
                          .merge(Newey_West_risk_premiums, on="factor")
                          .round(3)
                          )

    
    # Univariate Fama-Macbeth Time Series Averaging
    
    # MARKET RISK
    mkt_risk_prem = (mkt_risk_coef
       #create a melted Dataframe with 'month' column; 'factor' column to indicate which factor; 'estimate' column to show the coefficient
      .melt(id_vars="month", var_name="factor", value_name="estimate")
      .groupby("factor")["estimate"] #group by same factor and select the estimates to be applied the following calculation
      .apply(lambda x: pd.Series({
          "risk_premium": 100*x.mean(), #average across the time dimension to get expected value
          "t_statistic": x.mean()/x.std()*np.sqrt(len(x)),
          "standard_error": x.std() / np.sqrt(len(x))  # Calculate standard error
        })
      )
      .reset_index()
      .pivot(index="factor", columns="level_1", values="estimate") #pivot the table
      .reset_index()
    )
    
    p_values_mkt = (mkt_risk_coef
      .melt(id_vars="month", var_name="factor", value_name="estimate")
      .groupby("factor")
      .apply(lambda x: (
          smf.ols("estimate ~ 1", x)
          .fit()
          .pvalues["Intercept"]
        )
      )
      .reset_index()
    )
    
    Newey_West_mkt = (mkt_risk_coef
      .melt(id_vars="month", var_name="factor", value_name="estimate")
      .groupby("factor")
      .apply(lambda x: (
          x["estimate"].mean()/ 
            smf.ols("estimate ~ 1", x)
            .fit(cov_type="HAC", cov_kwds={"maxlags": 6}).bse
        )
      )
      .reset_index()
      .rename(columns={"Intercept": "t_statistic_newey_west"})
    )
        
    mkt_risk_prem = (mkt_risk_prem
                     .merge(p_values_mkt, on="factor")
                     .merge(Newey_West_mkt, on="factor")
                     .round(3)
                    )

    # SIZE
    size_prem = (size_premium_coef
       #create a melted Dataframe with 'month' column; 'factor' column to indicate which factor; 'estimate' column to show the coefficient
      .melt(id_vars="month", var_name="factor", value_name="estimate")
      .groupby("factor")["estimate"] #group by same factor and select the estimates to be applied the following calculation
      .apply(lambda x: pd.Series({
          "risk_premium": 100*x.mean(), #average across the time dimension to get expected value
          "t_statistic": x.mean()/x.std()*np.sqrt(len(x)),
          "standard_error": x.std() / np.sqrt(len(x))  # Calculate standard error
        })
      )
      .reset_index()
      .pivot(index="factor", columns="level_1", values="estimate") #pivot the table
      .reset_index()
    )
    
    p_values_size = (size_premium_coef
      .melt(id_vars="month", var_name="factor", value_name="estimate")
      .groupby("factor")
      .apply(lambda x: (
          smf.ols("estimate ~ 1", x)
          .fit()
          .pvalues["Intercept"]
        )
      )
      .reset_index()
    )
    
    Newey_West_size = (size_premium_coef
      .melt(id_vars="month", var_name="factor", value_name="estimate")
      .groupby("factor")
      .apply(lambda x: (
          x["estimate"].mean()/ 
            smf.ols("estimate ~ 1", x)
            .fit(cov_type="HAC", cov_kwds={"maxlags": 6}).bse
        )
      )
      .reset_index()
      .rename(columns={"Intercept": "t_statistic_newey_west"})
    )
        
    size_prem = (size_prem
                 .merge(p_values_size, on="factor")
                 .merge(Newey_West_size, on="factor")
                 .round(3)
                )

    # VALUE
    value_prem = (value_premium_coef
       #create a melted Dataframe with 'month' column; 'factor' column to indicate which factor; 'estimate' column to show the coefficient
      .melt(id_vars="month", var_name="factor", value_name="estimate")
      .groupby("factor")["estimate"] #group by same factor and select the estimates to be applied the following calculation
      .apply(lambda x: pd.Series({
          "risk_premium": 100*x.mean(), #average across the time dimension to get expected value
          "t_statistic": x.mean()/x.std()*np.sqrt(len(x)),
          "standard_error": x.std() / np.sqrt(len(x))  # Calculate standard error
        })
      )
      .reset_index()
      .pivot(index="factor", columns="level_1", values="estimate") #pivot the table
      .reset_index()
    )
    
    p_values_value = (value_premium_coef
      .melt(id_vars="month", var_name="factor", value_name="estimate")
      .groupby("factor")
      .apply(lambda x: (
          smf.ols("estimate ~ 1", x)
          .fit()
          .pvalues["Intercept"]
        )
      )
      .reset_index()
    )
    
    Newey_West_value = (value_premium_coef
      .melt(id_vars="month", var_name="factor", value_name="estimate")
      .groupby("factor")
      .apply(lambda x: (
          x["estimate"].mean()/ 
            smf.ols("estimate ~ 1", x)
            .fit(cov_type="HAC", cov_kwds={"maxlags": 6}).bse
        )
      )
      .reset_index()
      .rename(columns={"Intercept": "t_statistic_newey_west"})
    )
        
    value_prem = (value_prem
                 .merge(p_values_value, on="factor")
                 .merge(Newey_West_value, on="factor")
                 .round(3)
                )
    
    
    # PROFITABILITY
    op_prem = (op_premium_coef
       #create a melted Dataframe with 'month' column; 'factor' column to indicate which factor; 'estimate' column to show the coefficient
      .melt(id_vars="month", var_name="factor", value_name="estimate")
      .groupby("factor")["estimate"] #group by same factor and select the estimates to be applied the following calculation
      .apply(lambda x: pd.Series({
          "risk_premium": 100*x.mean(), #average across the time dimension to get expected value
          "t_statistic": x.mean()/x.std()*np.sqrt(len(x)),
          "standard_error": x.std() / np.sqrt(len(x))  # Calculate standard error
        })
      )
      .reset_index()
      .pivot(index="factor", columns="level_1", values="estimate") #pivot the table
      .reset_index()
    )
    
    p_values_op = (op_premium_coef
      .melt(id_vars="month", var_name="factor", value_name="estimate")
      .groupby("factor")
      .apply(lambda x: (
          smf.ols("estimate ~ 1", x)
          .fit()
          .pvalues["Intercept"]
        )
      )
      .reset_index()
    )
    
    Newey_West_op = (op_premium_coef
      .melt(id_vars="month", var_name="factor", value_name="estimate")
      .groupby("factor")
      .apply(lambda x: (
          x["estimate"].mean()/ 
            smf.ols("estimate ~ 1", x)
            .fit(cov_type="HAC", cov_kwds={"maxlags": 6}).bse
        )
      )
      .reset_index()
      .rename(columns={"Intercept": "t_statistic_newey_west"})
    )
        
    op_prem = (op_prem
                 .merge(p_values_op, on="factor")
                 .merge(Newey_West_op, on="factor")
                 .round(3)
                )
    
    # INVESTMENT LEVEL
    inv_prem = (inv_premium_coef
       #create a melted Dataframe with 'month' column; 'factor' column to indicate which factor; 'estimate' column to show the coefficient
      .melt(id_vars="month", var_name="factor", value_name="estimate")
      .groupby("factor")["estimate"] #group by same factor and select the estimates to be applied the following calculation
      .apply(lambda x: pd.Series({
          "risk_premium": 100*x.mean(), #average across the time dimension to get expected value
          "t_statistic": x.mean()/x.std()*np.sqrt(len(x)),
          "standard_error": x.std() / np.sqrt(len(x))  # Calculate standard error
        })
      )
      .reset_index()
      .pivot(index="factor", columns="level_1", values="estimate") #pivot the table
      .reset_index()
    )
    
    p_values_inv = (inv_premium_coef
      .melt(id_vars="month", var_name="factor", value_name="estimate")
      .groupby("factor")
      .apply(lambda x: (
          smf.ols("estimate ~ 1", x)
          .fit()
          .pvalues["Intercept"]
        )
      )
      .reset_index()
    )
    
    Newey_West_inv = (inv_premium_coef
      .melt(id_vars="month", var_name="factor", value_name="estimate")
      .groupby("factor")
      .apply(lambda x: (
          x["estimate"].mean()/ 
            smf.ols("estimate ~ 1", x)
            .fit(cov_type="HAC", cov_kwds={"maxlags": 6}).bse
        )
      )
      .reset_index()
      .rename(columns={"Intercept": "t_statistic_newey_west"})
    )
        
    inv_prem = (inv_prem
                 .merge(p_values_inv, on="factor")
                 .merge(Newey_West_inv, on="factor")
                 .round(3)
                )
    
    # Combine the three DataFrames row-wise
    uni_fama_macbeth = pd.concat([mkt_risk_prem, 
                                  size_prem, 
                                  value_prem, 
                                  op_prem, 
                                  inv_prem])
    
    # Reset index after concatenation
    uni_fama_macbeth.reset_index(drop=True, inplace=True)
    
    uni_fama_macbeth = uni_fama_macbeth.round(3)
    
    return multi_fama_macbeth, uni_fama_macbeth

multi_fama_macbeth, uni_fama_macbeth = fama_macbeth_final_reg(risk_factor_coefs, 
                                                                mkt_risk_coef, 
                                                                size_premium_coef, 
                                                                value_premium_coef,
                                                                op_premium_coef,
                                                                inv_premium_coef
                                                                )


# Display the DataFrame
print(multi_fama_macbeth)
print(uni_fama_macbeth)


# save to the database
(fama_macbeth_table.to_sql(
    name="fama_macbeth_table",
    con=cw602_database,
    if_exists="replace",
    index=False
    )
)

(multi_fama_macbeth.to_sql(
    name="multi_fama_macbeth",
    con=cw602_database,
    if_exists="replace",
    index=False
    )
)

(uni_fama_macbeth.to_sql(
    name="uni_fama_macbeth",
    con=cw602_database,
    if_exists="replace",
    index=False
    )
)


#=============================================================================#
#============================= 1.4: TASK 1 PLOTS =============================#
#=============================================================================#

######################### Access cw602_database.sqlite ########################

# Task 1.1's variables
cw602_database = sqlite3.connect(
  database = "/cw602_database.sqlite"
)

crsp_monthly = pd.read_sql_query(
  sql="SELECT * FROM crsp_monthly",
  con=cw602_database
)

compustat_monthly = pd.read_sql_query(
  sql="SELECT * FROM compustat_monthly",
  con=cw602_database
)

all_factors = pd.read_sql_query(
  sql="SELECT * FROM all_factors",
  con=cw602_database
)

# Task 1.2's variables


crsp_fundmonthly = pd.read_sql_query(
  sql="SELECT * FROM crsp_fundmonthly",
  con=cw602_database
)

ff5_regression_largevalue = pd.read_sql_query(
  sql="SELECT * FROM ff5_regression_largevalue",
  con=cw602_database
)


ff5_regression_smallgrowth = pd.read_sql_query(
  sql="SELECT * FROM ff5_regression_smallgrowth",
  con=cw602_database
)

# Task 1.3's variables

beta_monthly = pd.read_sql_query(
  sql="SELECT * FROM initial_beta_monthly",
  con=cw602_database
)

multi_fama_macbeth = pd.read_sql_query(
  sql="SELECT * FROM multi_fama_macbeth",
  con=cw602_database
)

uni_fama_macbeth = pd.read_sql_query(
  sql="SELECT * FROM uni_fama_macbeth",
  con=cw602_database
)


################################ Task 1.1: Plots ##############################

# PLOT NUMBER OF SECURITIES of EACH EXCHANGE
filtered_exchange = crsp_monthly[crsp_monthly['exchange'].isin(['AMEX', 'NASDAQ', 'NYSE'])]
# Group by exchange and date, then count the occurrences
securities_per_exch = (filtered_exchange
                      .groupby(["exchange", "date"])
                      .size()
                      .reset_index(name="n")
                      )
securities_per_exch = (
  ggplot(securities_per_exch, 
         aes(x="date", y="n", group="exchange")) +
  geom_line(aes(color="exchange", linetype="exchange")) + 
  labs(x="", y="", color="Exchange", linetype="Exchange",
       title="Monthly Number of Securities by Exchanges") +
  scale_x_datetime(date_breaks="10 years", date_labels="%Y") +
  scale_y_continuous(labels=comma_format())
)
securities_per_exch.draw()


amex_count_31122023 = filtered_exchange.loc[(filtered_exchange['exchange'] == 'AMEX') 
                                            & (filtered_exchange['month'] == '2023-12-01 00:00:00')].shape[0]

filtered_exchange['month'] = pd.to_datetime(filtered_exchange['month'])

# Count no. of securities in specific month for AMEX
amex_monthly_counts = (
    filtered_exchange[filtered_exchange['exchange'] == 'AMEX']
    .groupby(filtered_exchange['month'].dt.to_period('M'))
    .size()
    .to_frame(name='Number of Securities')
)

amex_122023_counts = amex_monthly_counts.loc[(amex_monthly_counts.index == pd.Period('2023-12'))]
amex_max_index, amex_max_count = amex_monthly_counts.idxmax(), amex_monthly_counts.max()


# Count no. of securities in specific month for NASDAQ
nasdaq_monthly_counts = (
    filtered_exchange[filtered_exchange['exchange'] == 'NASDAQ']
    .groupby(filtered_exchange['month'].dt.to_period('M'))
    .size()
    .to_frame(name='Number of Securities')
)
nasdaq_122023_counts = nasdaq_monthly_counts.loc[(nasdaq_monthly_counts.index == pd.Period('2023-12'))]
nasdaq_max_index, nasdaq_max_count = nasdaq_monthly_counts.idxmax(), nasdaq_monthly_counts.max()


# Count no. of securities in specific month for NYSE
nyse_monthly_counts = (
    filtered_exchange[filtered_exchange['exchange'] == 'NYSE']
    .groupby(filtered_exchange['month'].dt.to_period('M'))
    .size()
    .to_frame(name='Number of Securities')
)
nyse_122023_counts = nyse_monthly_counts.loc[(nyse_monthly_counts.index == pd.Period('2023-12'))]
nyse_max_index, nyse_max_count = nyse_monthly_counts.idxmax(), nyse_monthly_counts.max()



# 5 factors REPLICATION PLOT
# Convert 'month' column to datetime
all_factors['month'] = pd.to_datetime(all_factors['month'])

# Factors to plot
factors = ['smb', 'hml', 'rmw', 'cma']

# Plotting
for factor in factors:
    plt.figure(figsize=(8, 6))
    plt.plot(all_factors['month'], all_factors[f'{factor}_replicated'], color='red', alpha=0.5, label=f'Replicated {factor.upper()}')
    plt.plot(all_factors['month'], all_factors[f'{factor}'], color='black', alpha=1, label=f'Fama-French {factor.upper()}')
    plt.title(f'{factor.upper()} Factor')
    plt.xlabel('Month')
    plt.ylabel('Factor Value')
    plt.legend()

    # Set x-axis major locator and formatter
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=10))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.show()

# Descriptive Statistics Table of Replicated Factors
from scipy.stats import skew, kurtosis

rep_factors = all_factors.filter(regex='_replicated$')
desc_replicated = pd.DataFrame(rep_factors).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).transpose()
desc_replicated['Skewness'] = pd.DataFrame(rep_factors).apply(skew)
desc_replicated['Excess Kurtosis'] = pd.DataFrame(rep_factors).apply(kurtosis) - 3
desc_replicated['Number of Observations'] = pd.DataFrame(rep_factors).count()

desc_replicated.to_csv('desc_replicated.csv', index_label='Factor')


################################ Task 1.2: Plots ##############################

# 1. PLOT mutual fund return of 2 funds and the market excess return

# Convert date columns to proper datetime format
ff5_regression_largevalue['caldt'] = to_datetime(ff5_regression_largevalue['caldt'])
ff5_regression_smallgrowth['caldt'] = to_datetime(ff5_regression_smallgrowth['caldt'])

plt.figure(figsize=(12, 6))
# Plot market excess return
plt.plot(ff5_regression_largevalue['caldt'], ff5_regression_largevalue['mkt_excess'], label='Market Excess Return', linestyle='dashed')
# Plot mutual fund returns
plt.plot(ff5_regression_largevalue['caldt'], ff5_regression_largevalue['mret'], label='Large Value Fund Return (FMILX)', linestyle='dashed')
plt.plot(ff5_regression_smallgrowth['caldt'], ff5_regression_smallgrowth['mret'], label='Small Growth Fund Return (NEAGX)', linestyle='dashed')
# Adding labels and title
plt.title('Monthly Market Excess Return and Mutual Fund Returns')
plt.xlabel('Date')
plt.ylabel('Value')
# Set x-axis to show every 1 year
plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# Adding legend and grid
plt.legend()
# Show plot
plt.tight_layout()
plt.show()

# PLOT graph for each mutual fund and its significant factors
# Define elegant and professional colors
elegant_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# Plotting for large value fund
plt.figure(figsize=(12, 6))
# Plot mutual fund returns for large value fund
plt.plot(ff5_regression_largevalue['caldt'], ff5_regression_largevalue['mret'], label='Large Value Fund Return', color=elegant_colors[0])
# Plot HML replicated factor for large value fund
plt.plot(ff5_regression_largevalue['caldt'], ff5_regression_largevalue['hml_replicated'], label='HML Replicated', color=elegant_colors[1])
# Adding labels and title for large value fund
plt.title('Monthly Returns and Factors for Large Value Fund (FMILX)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# Adding legend and grid for large value fund
plt.legend()
# Show plot for large value fund
plt.tight_layout()
plt.show()


# PLOT for small growth fund
plt.figure(figsize=(12, 6))
# Plot mutual fund returns for small growth fund
plt.plot(ff5_regression_smallgrowth['caldt'], ff5_regression_smallgrowth['mret'], label='Small Growth Fund Return', color=elegant_colors[2])
# Plot market excess return for small growth fund
plt.plot(ff5_regression_smallgrowth['caldt'], ff5_regression_smallgrowth['hml_replicated'], label='HML Replicated', color=elegant_colors[3])
# Plot SMB replicated factor for small growth fund
plt.plot(ff5_regression_smallgrowth['caldt'], ff5_regression_smallgrowth['smb_replicated'], label='SMB Replicated', color=elegant_colors[4])
# Adding labels and title for small growth fund
plt.title('Monthly Returns and Factors for Small Growth Fund (NEAGX)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# Adding legend and grid for small growth fund
plt.legend()
# Show plot for small growth fund
plt.tight_layout()
plt.show()


################################ Task 1.3: Plots ##############################

# Descriptive statistics for Fama-Macbeth data
famac_factors = fama_macbeth_table[['ret_excess_lead', 
                                   'beta',
                                   'log_mktcap',
                                   'bm',
                                   'op',
                                   'inv']]

desc_famac = pd.DataFrame(famac_factors).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).transpose()
desc_famac['Skewness'] = pd.DataFrame(famac_factors).apply(skew)
desc_famac['Excess Kurtosis'] = pd.DataFrame(famac_factors).apply(kurtosis) - 3
desc_famac['Number of Observations'] = pd.DataFrame(famac_factors).count()

desc_famac.to_csv('desc_famac.csv', index_label='Factor')



# Plot Boxplot of BETA
beta_industries = (beta_monthly
  .merge(crsp_monthly, how="inner", on=["permno", "month"])
  .groupby(["industry","permno"])["beta_monthly"]
  .aggregate("mean")
  .reset_index()
)

beta_industries_order = (beta_industries
  .groupby("industry")["beta_monthly"]
  .aggregate("median")
  .sort_values()
  .index.tolist()
)

plot_beta_industries = (
  ggplot(beta_industries, 
         aes(x="industry", y="beta_monthly")) +
  geom_boxplot() +
  coord_flip() +
  labs(x="", y="", 
       title="Initial Beta Boxplots by Industry") +
  scale_x_discrete(limits=beta_industries_order)
)
plot_beta_industries.draw()

# PLOT LINE OF BETA DECILES
beta_quantiles = (beta_monthly
  .groupby("month")["beta_monthly"]
  .quantile(q=np.arange(0.1, 1.0, 0.1))
  .reset_index()
  .rename(columns={"level_1": "quantile"})
  .assign(quantile=lambda x: (x["quantile"]*100).astype(int))
  .dropna()
)

linetypes = ["-", "--", "-.", ":"]
n_quantiles = beta_quantiles["quantile"].nunique()

plot_beta_quantiles = (
  ggplot(beta_quantiles, 
         aes(x="month", y="beta_monthly", 
         color="factor(quantile)", linetype="factor(quantile)")) +
  geom_line() +
  labs(x="", y="", color="", linetype="",
       title="Monthly Deciles of Estimated Beta") +
  scale_x_datetime(breaks=date_breaks("5 year"), labels=date_format("%Y")) +
  scale_linetype_manual(
    values=[linetypes[l % len(linetypes)] for l in range(n_quantiles)]
  ) 
)
plot_beta_quantiles.draw()
