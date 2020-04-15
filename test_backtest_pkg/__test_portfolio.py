#%%%%%%%%%%%%%%%%%%%%%%%  Prepare for testing   %%%%%%%%%%%%%%
import os
import backtest_pkg.backtest_portfolio as bt
import pandas as pd 
from IPython.display import display
import importlib

os.chdir(r'M:\Share\Colleagues\Andy\Python Project\Backtest Module')
price_data = pd.read_csv('pkg_test/Adjusted_Price.csv', index_col=0, parse_dates=True)

#########################################################################
########                 Portfolio construction           ###############
#########################################################################
#%%%%%%%%%%%%%%%%%%%%%%%  Portfolio from weight   %%%%%%%%%%%%%%%
importlib.reload(bt)

# Small testing date:
small_price_data = price_data.iloc[:10, :5]
small_weight = pd.DataFrame(data=[[1, 2, 3]], index=[price_data.index[0]], columns=price_data.columns[:3])
small_share = pd.DataFrame(data=1, index=[price_data.index[0]], columns=price_data.columns[:3])

# Initiate a portfolio by weight:
small_port = bt.portfolio(small_weight, name='Small Portfolio')
# Better way to initialize a portfolio:
# small_port = bt.portfolio(weight = small_weight, name='Small Portfolio')
small_port.set_price(small_price_data)
display(small_port.weight)

##### Check some properties:
# End date:
small_port.end_date = None   # Default: last date of price data
print(f'End date (default): {small_port.end_date:%Y-%m-%d}')
small_port.end_date = pd.datetime(2013, 1, 10)  # Manually set
print(f'End date (set): {small_port.end_date:%Y-%m-%d}')

# Trading_status:
small_port.trading_status = None   # Default: TRUE if price is available
display(f'Trading Status (default):')
display(small_port.trading_status)
    
trading_status = small_price_data.notna()
trading_status.loc[:'2013-01-10', 'ALB SQ Equity'] = False
small_port.trading_status = trading_status  # Set for specific requirement
display(f'Trading Status (set):')
display(small_port.trading_status)

# Daily return: calculate from price, the first available date has return 1.
display(f'Daily Return:')
display(small_port.daily_ret)

#%%%%%%%%%%%%%%%%%%   Portfolio from share   %%%%%%%%%%%%%%%%%%%%%
# Initiate a portfolio by share:
small_port = bt.portfolio(share = small_share, name='Small share Portfolio')
small_port.set_price(small_price_data)
display(small_port.weight)

##### Check some properties:
# End date:
small_port.end_date = None   # Default: last date of price data
print(f'End date (default): {small_port.end_date:%Y-%m-%d}')
small_port.end_date = pd.datetime(2013, 1, 10)  # Manually set
print(f'End date (set): {small_port.end_date:%Y-%m-%d}')

# Trading_status:
small_port.trading_status = None   # Default: TRUE if price is available
display(f'Trading Status (default):')
display(small_port.trading_status)
    
trading_status = small_price_data.notna()
trading_status.loc[:'2013-01-10', 'ALB SQ Equity'] = False
small_port.trading_status = trading_status  # Set for specific requirement
display(f'Trading Status (set):')
display(small_port.trading_status)

# Daily return: calculate from price, the first available date has return 1.
display(f'Daily Return:')
display(small_port.daily_ret)

#####################    Error handling   ##############################
importlib.reload(bt)
#%%%%%%%%%%%%%%%%%%%%   Weight in untradable security  %%%%%%%%%%%%%%%%%
small_weight = pd.DataFrame(data=[[1, 2, 3]], index=[small_price_data.index[0]], columns= small_price_data.columns[-3:])
small_share = pd.DataFrame(data=1, index=[small_price_data.index[0]], columns=small_price_data.columns[-3:])

small_port = bt.portfolio(small_weight, name='Small error Portfolio')
small_port.set_price(small_price_data)
display(small_weight)
display(small_port.weight)

#%%%%%%%%%%%%%%%%%%%   Share in untradable security   %%%%%%%%%%%%%%%%%%%%%
small_port = bt.portfolio(share=small_share, name='Small error Portfolio from share')
small_port.set_price(small_price_data)
display(small_share)
display(small_port.weight)

#%%%%%%%%%%%%%%%%%%%%%%  Unknown tickers   %%%%%%%%%%%%%%%%%
small_weight = pd.DataFrame(data=[[1, 2, 3]], index=[small_price_data.index[0]], columns= small_price_data.columns[:3])
small_weight['Strange Equity']=1
small_port = bt.portfolio(small_weight, name='Small Portfolio with unknown ticker')
small_port.set_price(small_price_data)
display('Input weight:')
display(small_weight)
display('Output weight:')
display(small_port.weight)
#%%%%%%%%%%%%%%%%%%%%%%%    Outrange date   %%%%%%%%%%%%%%%%%%%%%%%%
small_weight = pd.DataFrame(data=[[1, 2, 3]], index=[small_price_data.index[0]], columns= small_price_data.columns[:3])
small_weight.loc[pd.to_datetime('2019-01-01'), :] = 1
small_port = bt.portfolio(small_weight, name='Small Portfolio with unknown ticker')
small_port.set_price(small_price_data)
display('Input weight:')
display(small_weight)
display('Final weight:')
display(small_port.weight)
#%%%%%%%%%%%%%%%%%%%%%%  Unknown tickers in share   %%%%%%%%%%%%%%%%%
small_share = pd.DataFrame(data=1, index=[small_price_data.index[0]], columns= small_price_data.columns[:3])
small_share['Strange Equity']=1
small_port = bt.portfolio(share = small_share, name='Small Portfolio with unknown ticker')
small_port.set_price(small_price_data)
display('Input weight:')
display(small_share)
display('Final weight:')
display(small_port.weight)
#%%%%%%%%%%%%%%%%%%%%%%  Outrange date in share   %%%%%%%%%%%%%%%%%
small_share = pd.DataFrame(data=1, index=[small_price_data.index[0]], columns= small_price_data.columns[:3])
small_share.loc[pd.to_datetime('2019-01-01'), :]=1
small_port = bt.portfolio(share = small_share, name='Small Portfolio with unknown ticker')
small_port.set_price(small_price_data)
display('Input weight:')
display(small_share)
display('Final weight:')
display(small_port.weight)

########################################################################
##################         Backtesting                   ###############
########################################################################

#%%%%%%%%%%%%%%   Construct a small portfolio   %%%%%%%%%%%%%%
importlib.reload(bt)
small_price_data = price_data.iloc[:10, :5]

# Initiate an equal weight portfolio by weight:
small_weight = pd.DataFrame(data=1, index=[small_price_data.index[0]], columns=small_price_data.columns)
small_port = bt.portfolio(small_weight, name='Small Portfolio')
small_port.set_price(small_price_data)

#%%%%%%%%%%%%%%%%%%%%%   Simple drift   %%%%%%%%%%%%%%%%%%%%
drifting_weight = small_port._drift_weight(small_port.weight)
display('Simple drift')
display(drifting_weight)

#%%%%%%%%%%%%%%%%%%%%    Rebalanced drift   %%%%%%%%%%%%%%%%%
rebalance_date = pd.datetime(2013, 1, 10)
rebalanced_weight = pd.DataFrame(data=1, index=[rebalance_date], columns=small_price_data.columns)
drifting_weight_rebalanced = drifting_weight.copy()
new_weights = small_port._drift_weight(drifting_weight.loc[[rebalance_date], :], rebalanced_weight=rebalanced_weight)
drifting_weight_rebalanced.loc[new_weights.index, :] = new_weights

display(f'Rebalance on {rebalance_date: %y-%m-%d} to equal weight')
display(drifting_weight_rebalanced)

#%%%%%%%%%%%%%%%%%%%   Rebalance drift with untradable stocks  %%%%%%%%%%%%%%%
trading_status = small_price_data.notna()
trading_status.loc[rebalance_date, :] = [False]*2 + [True]*3
small_port.trading_status = trading_status
drifting_weight_rebalanced = drifting_weight.copy()
new_weights = small_port._drift_weight(drifting_weight.loc[[rebalance_date], :], rebalanced_weight=rebalanced_weight)
drifting_weight_rebalanced.loc[new_weights.index, :] = new_weights

display(f'Rebalance on {rebalance_date: %y-%m-%d} to equal weight (untradable on SZG GY, KER FP)')
display(drifting_weight_rebalanced)

#%%%%%%%%%%%%%%%%%%%%    Extend weight %%%%%%%%%%%%%%%%%%%%%%%
importlib.reload(bt)
small_price_data = price_data.iloc[:10, :5]

# Initiate an equal weight portfolio by weight:
small_weight = pd.DataFrame(data=1, index=small_price_data.index[0::5], columns=small_price_data.columns)
small_port = bt.portfolio(small_weight, name='Small Portfolio')
small_port.set_price(small_price_data)

display('Portfolio weight:')
display(small_port.weight)
display('Extended portfolio weight:')
display(small_port.ex_weight)

#%%%%%%%%%%%%%%%%%%%%%%%%   Daily return and total return    %%%%%%%%%%%%%%%%%
display('Daily return:')
display(small_port.port_daily_ret)
display('Total return:')
display(small_port.port_total_ret)
display(small_port.port_total_ret.plot())

#%%%%%%%%%%%%%%%%%%%%%%   Extend weight with untradable   %%%%%%%%%%%%%%
# Prepare weight:
small_weight = pd.DataFrame(data=1, index=small_price_data.index[0::5], columns=small_price_data.columns)
# Prepare trading status:
rebalance_date = small_price_data.index[5]
trading_status = small_price_data.notna()
trading_status.loc[rebalance_date, :] = [False]*2 + [True]*3
# Construct portfolio
small_port = bt.portfolio(small_weight, name='Small Portfolio')
small_port.set_price(small_price_data, trading_status=trading_status)

display('Portfolio weight:')
display(small_port.weight)
display('Extended portfolio weight:')
display(small_port.ex_weight)

#%%%%%%%%%%%%%%%%%%%%%%%%%   Daily and total return   %%%%%%%%%%%%%%%%%%
display('Daily return:')
display(small_port.port_daily_ret)
display('Total return:')
display(small_port.port_total_ret)
display(small_port.port_total_ret.plot())

#%%%%%%%%%%%%%%%%%%    Simple backtest (no benchmark)   %%%%%%%%%%%%%%%%%
importlib.reload(bt)
port_weight = pd.DataFrame(data=0, index=small_price_data.index[[0, 5]], columns=small_price_data.columns) 
port_weight.iloc[0, [0, 1]] = 1
port_weight.iloc[1, [2, 3]] = 1
# Construct portfolio
small_port = bt.portfolio(weight=port_weight, name='Selection Portfolio')
small_port.set_price(small_price_data)

small_port.backtest(plot=True)

#%%%%%%%%%%%%%%%%%%    Backtest with benchmark   %%%%%%%%%%%%%%%%%%%%%%%
# Prepare weight:
benchmark_weight = pd.DataFrame(data=1, index=small_price_data.index[[0, 5]], columns=small_price_data.columns)
port_weight = pd.DataFrame(data=0, index=small_price_data.index[[0, 5]], columns=small_price_data.columns) 
port_weight.iloc[0, [0, 1]] = 1
port_weight.iloc[1, [2, 3]] = 1
# Construct portfolio
small_port = bt.portfolio(
    weight=port_weight, benchmark=benchmark_weight,
    name='Selection Portfolio', benchmark_name='Equal weight Benchmark')
small_port.set_price(small_price_data)

display(small_port.backtest())
display(small_port.performance_plot())

#%%%%%%%%%%%%%%%%%%    Backtest with benchmark (Large portfolio) %%%%%%%%%%%%%%%%%%%%%%%
seed = 56

# Prepare end date, weights for portfolio and benchmark:
testing_length = 300
end_date = price_data.index[testing_length]
benchmark_weight = pd.DataFrame(data=1, index=price_data.index[0:testing_length:20], columns=price_data.columns)
port_weight = pd.DataFrame(data=0, index=price_data.index[0:testing_length:20], columns=price_data.columns)
for i in range(port_weight.shape[0]):
    stock_selection = port_weight.columns.to_series().sample(100, random_state = seed*(i+1))
    port_weight.iloc[i, :][stock_selection] = 1
# Construct portfolio
large_port= bt.portfolio(
    weight=port_weight, benchmark=benchmark_weight, end_date=end_date,
    name='Selection Portfolio', benchmark_name='Equal weight Benchmark')
large_port.set_price(price_data)

display(large_port.backtest())
display(large_port.performance_plot())
display(large_port.performance_summary())



###########################################################################
#################         Performance                 #####################
###########################################################################
#%%%%%%%%%%%%%%%%%%%%%%%
small_port.performance_plot()
small_port.performance_summary()















































#%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
period_ts = pd.Series(bt_result.index.map(lambda s: (s>weight.index).sum()+(s>end_date)), index=bt_result.index)
period_ts.name = 'Period'
# bt_result['Active Return'].groupby(period_ts).agg({'Active Return':lambda s:s.cumprod().tail(1),'Tracking Error': 'std'})

port_ret= select_port.port_daily_ret
bm_ret = select_port.benchmark.port_daily_ret
daily_active_ret = port_ret - bm_ret
port_period_ret = port_ret.groupby(period_ts).agg(Portfolio_Return = 'prod') -1
bm_period_ret = bm_ret.groupby(period_ts).agg(Benchmark_Return = 'prod') -1
agg_df = pd.concat([port_period_ret, bm_period_ret], axis=1)
agg_df['Active_Return'] = agg_df.iloc[:, 0] - agg_df.iloc[:, 1]
agg_df['Tracking_Error'] = daily_active_ret.groupby(period_ts).agg('std')
agg_df['Sharpe_Ratio'] = agg_df.Active_Return/agg_df.Tracking_Error
agg_df = agg_df.drop(index=0)
agg_df

#%%
