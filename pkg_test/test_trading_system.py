#%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import backtest_pkg.backtest_trading_system as bt 
import importlib
import pandas as pd 
import matplotlib.pyplot as plt

os.chdir(r'M:\Share\Colleagues\Andy\Python Project\Backtest Module')
price_data = pd.read_csv('pkg_test/Adjusted_Price.csv', index_col=0, parse_dates=True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%
importlib.reload(bt)
