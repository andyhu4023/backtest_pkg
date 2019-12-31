#%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

os.chdir(r'M:\Share\Colleagues\Andy\Python Project\Backtest Module')
price_data = pd.read_csv('pkg_test/Adjusted_Price.csv', index_col=0, parse_dates=True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%
# import importlib
# spec = importlib.util.spec_from_file_location("bt", r"backtest_pkg\backtest_portfolio.py")
# bt = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(bt)
import backtest_pkg.backtest_signal as bt 
import importlib
importlib.reload(bt)
