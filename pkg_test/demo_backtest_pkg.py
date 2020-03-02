#%%%%%%%%%%%%%%%%%
import backtest_pkg as bt 
from pandas_datareader import DataReader
import pandas as pd 

# Getting adjusted price from yahoo:
universe = ['MMM', 'ABBV', 'FB', 'T', 'GOOGL']
start = pd.datetime(2018,12,31)
end = pd.datetime(2019, 12, 31)

price_data = DataReader(universe, 'yahoo', start=start, end=end)['Adj Close']

# Construct portfolio weight: (From strategies like MVO)
port_weight = pd.DataFrame(columns= universe)
port_weight.loc[pd.to_datetime('2018-12-31'), :] = pd.Series([3, 5, 2, 3, 4], index= universe)
port_weight.loc[pd.to_datetime('2019-06-28'), :] = pd.Series([1, 2, 5, 3, 4], index=universe)

# Equal weight benchmark:
benchmark = pd.DataFrame(1, index= port_weight.index, columns=port_weight.columns)

# Backtest process:
portfolio = bt.portfolio(port_weight, benchmark=benchmark, end_date=end)
portfolio.set_price(price_data)
bt_result = portfolio.backtest(plot=True)
print(bt_result.tail())