#%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import backtest_pkg as bt 
import importlib
import pandas as pd 
from pandas_datareader import data
# import matplotlib.pyplot as plt


#%%%%%%%%%%%%%%%%   Code for first download    %%%%%%
start_date = '2010-01-01'
end_date = '2018-12-31'

ticker_list = ['AAPL','GOOG','FB', 'MSFT']
for ticker in ticker_list:
    price_date = data.DataReader(ticker, 'yahoo', start_date, end_date)
    price_date.to_csv(f'pkg_test/Technical Data/{ticker}.csv')


#%%%%%%%%%%%%%%%%%%%%%%%%%%
importlib.reload(bt)
