import unittest
from pandas.util.testing import assert_frame_equal, assert_series_equal

import backtest_pkg as bt 
import pandas as pd 
import numpy as np
from math import sqrt, log, sin, pi

def cal_std(data):
    if len(data)<=1:
        return np.nan
    data_mean = sum(data)/len(data)
    data_var = sum((i-data_mean)**2 for i in data)/(len(data)-1)
    return sqrt(data_var)
def cal_mean(data):
    return sum(data)/len(data)

class TestPortfolio(unittest.TestCase):
    def setUp(self):
        n = 10   # length of the period
        price_dict = {
            'Up trend': list(range(1, n+1)),
            'Down trend': list(range(n, 0, -1)),
            'Convex': list(1+(n/2)**2+ i*(i-n+1) for i in range(n)),
            'Concave': list(1+i*(n-1-i) for i in range(n)),
            'Sin': list(1+n*(1+sin(i/(n-1)*2*pi)) for i in range(n)),
        }
        adj_price_df = pd.DataFrame(price_dict, index=pd.date_range('2020-01-01', periods=n,freq='D'))
        self.ticker = adj_price_df.columns 
        self.index = adj_price_df.index
        self.price = adj_price_df
        ''' Price in values:
        'Up trend': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Down trend': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'Convex': [26, 18, 12, 8, 6, 6, 8, 12, 18, 26],
        'Concave': [1, 9, 15, 19, 21, 21, 19, 15, 9, 1],
        'Sin': [11, 17.42, 20.84, 19.66, 14.42, 7.57, 2.33, 1.15, 4.57, 11]
        '''

        self.trading_status = pd.DataFrame(True, index=self.index, columns=self.ticker)
        self.trading_status.iloc[:3, 0]=False
        self.trading_status.iloc[3:6, 1]=False
        self.trading_status.iloc[6:, 2]=False

        # Equal weight portfolio:
        self.weight = pd.DataFrame(1, index=self.index[[0, 5]], columns=self.ticker)
        # Equal weight asset values:
        self.asset_values_no_rebal = self.price.copy()
        self.asset_values_no_rebal = self.asset_values_no_rebal.apply(lambda ts: ts/ts[0], axis=0)
        self.asset_values_1_rebal = self.asset_values_no_rebal.copy()
        rebal_value = self.asset_values_1_rebal.iloc[5, :].mean()
        self.asset_values_1_rebal.iloc[5:,:] = self.asset_values_1_rebal.iloc[5:,:].apply(lambda ts: ts/ts[0]*rebal_value, axis=0)
        # With trading status:
        self.asset_values_no_rebal_tst = self.asset_values_no_rebal.copy()
        self.asset_values_no_rebal_tst.iloc[:, 0] = 0
        asset_values = self.asset_values_no_rebal.copy()
        asset_values.iloc[:5, 0] = 0
        rebal_value = asset_values.iloc[5, [2, 3, 4]].sum()/4
        apply_col = [0, 2, 3, 4]
        asset_values.iloc[5:,apply_col] = asset_values.iloc[5:,apply_col].apply(lambda ts: ts/ts[0]*rebal_value, axis=0)
        self.asset_values_1_rebal_tst = asset_values
        
        # Price weight portfolio: 
        self.share = pd.DataFrame(1, index=self.index[[0, 5]], columns=self.ticker)
        # Price weight asset values:
        self.asset_values_no_rebal_share = self.price.copy()
        self.asset_values_1_rebal_share = self.price.copy()
        # With trading status: 
        self.asset_values_no_rebal_share_tst = self.asset_values_no_rebal_share.copy()
        self.asset_values_no_rebal_share_tst.iloc[:, 0] = 0
        asset_values = self.asset_values_no_rebal_share.copy()
        asset_values.iloc[:5, 0] = 0
        adjust_factor = asset_values.iloc[5, [2, 3, 4]].sum()/asset_values.iloc[5, [0, 2, 3, 4]].sum()
        apply_col = [0, 2, 3, 4]
        asset_values.iloc[5:,apply_col] = asset_values.iloc[5:,apply_col].apply(lambda ts: ts*adjust_factor, axis=0)
        self.asset_values_1_rebal_share_tst = asset_values




#######################    Portfolio Construction   ########################
    def test_portfolio_set_price(self):
        # Normal setting:
        port = bt.portfolio(weight=self.weight)
        port.set_price(price=self.price)
        expect_status = pd.DataFrame(True, index=self.index, columns=self.ticker)
        assert_frame_equal(port.price, self.price)
        assert_frame_equal(port.trading_status, expect_status)

        # Setting at initiation:
        port = bt.portfolio(weight=self.weight, price=self.price)
        expect_status = pd.DataFrame(True, index=self.index, columns=self.ticker)
        assert_frame_equal(port.price, self.price)
        assert_frame_equal(port.trading_status, expect_status)

        # Price and trading status cannot be set directly
        with self.assertRaises(AttributeError):
            port.price = self.price  
        with self.assertRaises(AttributeError):
            port.trading_status = self.trading_status  

        # Try masking out untradable prices:
        price = self.price.where(self.trading_status, other=np.nan)
        port = bt.portfolio(weight=self.weight)
        port.set_price(price=price)
        assert_frame_equal(port.price, price)
        assert_frame_equal(port.trading_status, self.trading_status)
    def test_portfolio_set_price_and_trading_status(self):
        # Normal setting price and trading status:
        port = bt.portfolio(weight=self.weight)
        port.set_price(price=self.price, trading_status=self.trading_status)
        assert_frame_equal(port.price, self.price)
        assert_frame_equal(port.trading_status, self.trading_status)

        # Setting at initiation:
        port = bt.portfolio(weight=self.weight, price=self.price, trading_status=self.trading_status)
        assert_frame_equal(port.price, self.price)
        assert_frame_equal(port.trading_status, self.trading_status)

        # Independent NA prices and trading status:
        price = self.price.copy()
        price.iloc[:5, 4] = np.nan
        expect_status = self.trading_status.copy()
        expect_status.iloc[:5, 4]=False
        port = bt.portfolio(weight=self.weight)
        port.set_price(price=price, trading_status=self.trading_status)
        assert_frame_equal(port.price, price)
        assert_frame_equal(port.trading_status, expect_status)

        # Out range trading status:
        out_range_status = self.trading_status.copy()
        out_range_status['Extra Ticker'] = True
        out_range_status.loc[pd.to_datetime('2020-01-20'), :]=False
        expect_status = self.trading_status
        port = bt.portfolio(weight=self.weight)
        port.set_price(price=self.price, trading_status=out_range_status)
        assert_frame_equal(port.trading_status, expect_status)

    def test_portfolio_weight(self):
        # Noraml equal weigt:
        port = bt.portfolio(weight=self.weight, price=self.price)
        expect = pd.DataFrame(0.2, index=self.index[[0, 5]], columns=self.ticker)
        assert_frame_equal(port.weight, expect) 

        # Weights of row sum==zeros:
        weight = pd.DataFrame(0.2, index=self.index[[0, 5]], columns=self.ticker)
        weight.iloc[1, :] = 0
        port = bt.portfolio(weight=weight, price=self.price)
        expect = weight.iloc[[0], :]
        assert_frame_equal(port.weight, expect)
        weight = pd.DataFrame(0.2, index=self.index[[0, 5]], columns=self.ticker)
        weight.iloc[0, :] = 0
        port = bt.portfolio(weight=weight, price=self.price)
        expect = weight.iloc[[1], :]
        assert_frame_equal(port.weight, expect)

        # Out range weight:
        out_range_weight=self.weight.copy()
        out_range_weight['Extra Ticker']=1
        out_range_weight.loc[pd.to_datetime('2020-01-20'), :]=1
        port = bt.portfolio(weight=out_range_weight, price=self.price)
        expect = pd.DataFrame(0.2, index=self.index[[0, 5]], columns=self.ticker)
        assert_frame_equal(port.weight, expect) 

        # Weight cannot be set after initiation:
        with self.assertRaises(AttributeError):
            port.weight = self.weight
    def test_portfolio_weight_with_trading_status(self):
        # Weights on untradables:
        port = bt.portfolio(weight=self.weight, price=self.price, trading_status=self.trading_status)
        expect = pd.DataFrame(0.25, index=self.index[[0, 5]], columns=self.ticker)
        expect.iloc[0, 0]=0
        expect.iloc[1, 1]=0
        assert_frame_equal(port.weight, expect) 

        # Weights sum 0 from untradables:
        weight = pd.DataFrame(0.25, index=self.index[[0, 5]], columns=self.ticker)
        weight.iloc[0, 0]=0
        weight.iloc[1, :] = [0, 1, 0, 0, 0]
        port = bt.portfolio(weight=weight, price=self.price, trading_status=self.trading_status)
        expect = weight.iloc[[0], :]
        assert_frame_equal(port.weight, expect) 
        
    def test_portfolio_from_share(self):
        # No trading status:
        price_1 = [1, 10, 26, 1, 11]
        price_2 = [6, 5, 6, 21, 1+10*(1+sin(5/9*2*pi))]
        weight_value=[[i/sum(price_1) for i in price_1], [j/sum(price_2) for j in price_2]]
        expect = pd.DataFrame(weight_value, index=self.index[[0, 5]], columns=self.ticker)
        port = bt.portfolio(share=self.share, price=self.price)
        assert_frame_equal(port.weight,expect) 

        # With trading status:
        price_1 = [0, 10, 26, 1, 11]
        price_2 = [6, 0, 6, 21, 1+10*(1+sin(5/9*2*pi))]
        weight_value=[[i/sum(price_1) for i in price_1], [j/sum(price_2) for j in price_2]]
        port = bt.portfolio(share=self.share, price=self.price, trading_status=self.trading_status)
        expect = pd.DataFrame(weight_value, index=self.index[[0, 5]], columns=self.ticker)
        assert_frame_equal(port.weight,expect) 
    
    def test_portfolio_end_date(self):
        # Defulat end date: last date of price
        port = bt.portfolio(weight=self.weight, price=self.price)
        self.assertEqual(port.end_date, self.index[-1])

        # Set end date at initiation:
        end_date = pd.to_datetime('2020-01-08')
        port = bt.portfolio(weight=self.weight, price=self.price, end_date=end_date)
        self.assertEqual(port.end_date, end_date)
        # Change end date after initiation:
        port = bt.portfolio(weight=self.weight, price=self.price)
        self.assertEqual(port.end_date, self.index[-1])
        port.end_date = end_date
        self.assertEqual(port.end_date, end_date)
        
############################    Backtest calculations    ############################
    def test_portfolio_daily_ret(self):
        price = pd.DataFrame(index=self.index)
        price['Normal'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Normal case
        price['Suspension'] = [1, 2, 3, np.nan, np.nan, np.nan, 4, 5, 6, 7]  # Temporary suspension
        price['Delisting'] = [1, 2, 3, 4, 5] + [np.nan]*5  # Delisting
        price['Late'] = [np.nan]*5 + [1, 2, 3, 4, 5]# Late listing:
        
        expect = pd.DataFrame(index=self.index)
        expect['Normal'] = [np.nan]+[log((i+1)/i) for i in range(1,10)]  
        expect['Suspension'] = [np.nan]+[log(2/1), log(3/2)] + [0.]*3 + [log(4/3), log(5/4), log(6/5), log(7/6)] 
        expect['Delisting'] =[np.nan]+[log((i+1)/i) for i in range(1,5)] + [0.]*5
        expect['Late'] = [np.nan]*6+[log((i+1)/i) for i in range(1,5)]

        port = bt.portfolio(weight=self.weight, price=price)
        assert_frame_equal(port.daily_ret, expect)

    def test_portfolio_drift_weight(self):
        # NO rebalance:
        port = bt.portfolio(weight=self.weight.iloc[[0], :], price=self.price)
        expect = self.asset_values_no_rebal.copy()
        expect = expect.apply(lambda ts: ts/ts.sum(), axis=1)
        assert_frame_equal(port.ex_weight, expect)

        # 1 rebalance:
        port = bt.portfolio(weight=self.weight, price=self.price)
        expect = self.asset_values_1_rebal.copy()
        expect = expect.apply(lambda ts: ts/ts.sum(), axis=1)
        assert_frame_equal(port.ex_weight, expect)

    def test_portfolio_performance_return(self):
        # NO rebalance:
        port = bt.portfolio(weight=self.weight.iloc[[0], :], price=self.price)
        port_value = self.asset_values_no_rebal.sum(axis=1).values
        expect_daily_ret = [0]+list(log(port_value[i+1]/port_value[i]) for i in range(len(self.index)-1))
        expect_daily_ret = pd.Series(expect_daily_ret, index=self.index)
        expect_total_ret = [0]+list(log(port_value[i+1]/port_value[0]) for i in range(len(self.index)-1))
        expect_total_ret = pd.Series(expect_total_ret, index=self.index)
        assert_series_equal(port.port_daily_ret, expect_daily_ret)
        assert_series_equal(port.port_total_ret, expect_total_ret)
        port_value_ts = self.asset_values_no_rebal.sum(axis=1)
        assert_series_equal(port.port_total_value, port_value_ts/port_value_ts[0])

        # 1 rebalance:
        port = bt.portfolio(weight=self.weight, price=self.price)
        port_value = self.asset_values_1_rebal.sum(axis=1).values
        expect_daily_ret = [0]+list(log(port_value[i+1]/port_value[i]) for i in range(len(self.index)-1))
        expect_daily_ret = pd.Series(expect_daily_ret, index=self.index)
        expect_total_ret = [0]+list(log(port_value[i+1]/port_value[0]) for i in range(len(self.index)-1))
        expect_total_ret = pd.Series(expect_total_ret, index=self.index)
        assert_series_equal(port.port_daily_ret, expect_daily_ret)
        assert_series_equal(port.port_total_ret, expect_total_ret)
        port_value_ts = self.asset_values_1_rebal.sum(axis=1)
        assert_series_equal(port.port_total_value, port_value_ts/port_value_ts[0])

    def test_portfolio_backtest(self):
        # Default setting, name = 'Portfolio'
        port = bt.portfolio(weight=self.weight, price=self.price)
        port_value_ts = self.asset_values_1_rebal.sum(axis=1)
        port_value_ts = port_value_ts/port_value_ts[0]
        assert_frame_equal(port.backtest(), port_value_ts.to_frame(name='Portfolio'))

        # Setting portfolio name to 'Equal Weight'
        name = 'Equal Weight'
        port = bt.portfolio(weight=self.weight, price=self.price, name=name)
        assert_frame_equal(port.backtest(), port_value_ts.to_frame(name=name))



    def test_portfolio_drift_weight_with_trading_status(self):
        # NO rebalance:
        port = bt.portfolio(weight=self.weight.iloc[[0], :], price=self.price, trading_status=self.trading_status)
        expect = self.asset_values_no_rebal_tst.apply(lambda ts: ts/ts.sum(), axis=1)
        assert_frame_equal(port.ex_weight, expect)

        # 1 rebalance:
        port = bt.portfolio(weight=self.weight, price=self.price, trading_status=self.trading_status)
        expect = self.asset_values_1_rebal_tst.apply(lambda ts: ts/ts.sum(), axis=1)
        assert_frame_equal(port.ex_weight, expect)
    def test_portfolio_performance_return_with_trading_status(self):
        # NO rebalance:
        port = bt.portfolio(weight=self.weight.iloc[[0], :], price=self.price, trading_status=self.trading_status)
        port_value = self.asset_values_no_rebal_tst.sum(axis=1).values
        expect_daily_ret = [0]+list(log(port_value[i+1]/port_value[i]) for i in range(len(self.index)-1))
        expect_daily_ret = pd.Series(expect_daily_ret, index=self.index)
        expect_total_ret = [0]+list(log(port_value[i+1]/port_value[0]) for i in range(len(self.index)-1))
        expect_total_ret = pd.Series(expect_total_ret, index=self.index)
        assert_series_equal(port.port_daily_ret, expect_daily_ret)
        assert_series_equal(port.port_total_ret, expect_total_ret)
        port_value_ts = self.asset_values_no_rebal_tst.sum(axis=1)
        assert_series_equal(port.port_total_value, port_value_ts/port_value_ts[0])

        # 1 rebalance:
        port = bt.portfolio(weight=self.weight, price=self.price, trading_status=self.trading_status)
        port_value = self.asset_values_1_rebal_tst.sum(axis=1).values
        expect_daily_ret = [0]+list(log(port_value[i+1]/port_value[i]) for i in range(len(self.index)-1))
        expect_daily_ret = pd.Series(expect_daily_ret, index=self.index)
        expect_total_ret = [0]+list(log(port_value[i+1]/port_value[0]) for i in range(len(self.index)-1))
        expect_total_ret = pd.Series(expect_total_ret, index=self.index)
        assert_series_equal(port.port_daily_ret, expect_daily_ret)
        assert_series_equal(port.port_total_ret, expect_total_ret)
        port_value_ts = self.asset_values_1_rebal_tst.sum(axis=1)
        assert_series_equal(port.port_total_value, port_value_ts/port_value_ts[0])
    def test_portfolio_backtest_with_trading_status(self):
        # Default setting, name = 'Portfolio'
        port = bt.portfolio(weight=self.weight, price=self.price, trading_status=self.trading_status)
        port_value_ts = self.asset_values_1_rebal_tst.sum(axis=1)
        port_value_ts = port_value_ts/port_value_ts[0]
        assert_frame_equal(port.backtest(), port_value_ts.to_frame(name='Portfolio'))

        # Setting portfolio name to 'Equal Weight'
        name = 'Equal Weight'
        port = bt.portfolio(weight=self.weight, price=self.price, trading_status=self.trading_status, name=name)
        assert_frame_equal(port.backtest(), port_value_ts.to_frame(name=name))

#####################   Portfolio with Benchmark    ##############################
    def test_portfolio_with_benchmark(self):
        price_weight = self.price.iloc[[0, 5], :].copy()
        price_weight = price_weight.apply(lambda ts: ts/ts.sum(), axis=1)
        equal_weight = pd.DataFrame(0.2, index=self.index[[0, 5]], columns=self.ticker)

        # Add benchmark at initiation:
        price_weight_port = bt.portfolio(share=self.share, price=self.price, name='Price Weight', benchmark=self.weight, benchmark_name='Equal Weight')
        assert_frame_equal(price_weight_port.weight, price_weight)
        assert_frame_equal(price_weight_port.benchmark.weight, equal_weight)
        self.assertEqual(price_weight_port.name, 'Price Weight')
        self.assertEqual(price_weight_port.benchmark.name, 'Equal Weight')

        # Set benchmark after initiation:
        price_weight_port = bt.portfolio(share=self.share, price=self.price, name='Price Weight')
        equal_weight_port = bt.portfolio(weight=self.weight, price=self.price, name='Equal Weight')
        price_weight_port.set_benchmark(equal_weight_port)
        assert_frame_equal(price_weight_port.weight, price_weight)
        assert_frame_equal(price_weight_port.benchmark.weight, equal_weight)
        self.assertEqual(price_weight_port.name, 'Price Weight')
        self.assertEqual(price_weight_port.benchmark.name, 'Equal Weight')

    def test_portfolio_backtest_with_benchmark(self):
        # No trading status:
        price_weight_port = bt.portfolio(share=self.share, price=self.price, name='Price Weight', benchmark=self.weight, benchmark_name='Equal Weight')
        result = pd.DataFrame()
        port_value_ts = self.asset_values_1_rebal_share.sum(axis=1)
        bm_value_ts = self.asset_values_1_rebal.sum(axis=1)
        result['Price Weight']= port_value_ts/port_value_ts[0]
        result['Equal Weight'] = bm_value_ts/bm_value_ts[0]
        result['Difference'] = result['Price Weight'] - result['Equal Weight']
        assert_frame_equal(price_weight_port.backtest(), result)

        # With trading status:
        price_weight_port = bt.portfolio(share=self.share, price=self.price, trading_status=self.trading_status, name='Price Weight', benchmark=self.weight, benchmark_name='Equal Weight')
        result = pd.DataFrame()
        port_value_ts = self.asset_values_1_rebal_share_tst.sum(axis=1)
        bm_value_ts = self.asset_values_1_rebal_tst.sum(axis=1)
        result['Price Weight']= port_value_ts/port_value_ts[0]
        result['Equal Weight'] = bm_value_ts/bm_value_ts[0]
        result['Difference'] = result['Price Weight'] - result['Equal Weight']
        assert_frame_equal(price_weight_port.backtest(), result)


####################    Portfolio Analytic Tools    ##########################
    def test_portfolio_performance_metrics(self):
        # No trading status:
        price_weight_port = bt.portfolio(share=self.share, price=self.price, name='Price Weight') 
        daily_ret = price_weight_port.port_daily_ret
        performance_df = pd.DataFrame(index=['Price Weight'])
        performance_df['Return'] = daily_ret.sum()
        performance_df['Volatility'] = daily_ret.std()*sqrt(len(daily_ret))
        performance_df['Sharpe'] = performance_df['Return']/performance_df['Volatility']
        hist_value = np.exp(daily_ret.cumsum())
        previous_peak = hist_value.cummax()
        performance_df['MaxDD'] = max(1 - hist_value/previous_peak)
        assert_series_equal(price_weight_port.period_return, performance_df['Return'])
        assert_series_equal(price_weight_port.period_volatility, performance_df['Volatility'])
        assert_series_equal(price_weight_port.period_sharpe_ratio, performance_df['Sharpe'])
        assert_series_equal(price_weight_port.period_maximum_drawdown, performance_df['MaxDD'])

        # No trading status:
        price_weight_port = bt.portfolio(share=self.share, price=self.price, name='Price Weight', trading_status=self.trading_status) 
        daily_ret = price_weight_port.port_daily_ret
        performance_df = pd.DataFrame(index=['Price Weight'])
        performance_df['Return'] = daily_ret.sum()
        performance_df['Volatility'] = daily_ret.std()*sqrt(len(daily_ret))
        performance_df['Sharpe'] = performance_df['Return']/performance_df['Volatility']
        hist_value = np.exp(daily_ret.cumsum())
        previous_peak = hist_value.cummax()
        performance_df['MaxDD'] = max(1 - hist_value/previous_peak)
        assert_series_equal(price_weight_port.period_return, performance_df['Return'])
        assert_series_equal(price_weight_port.period_volatility, performance_df['Volatility'])
        assert_series_equal(price_weight_port.period_sharpe_ratio, performance_df['Sharpe'])
        assert_series_equal(price_weight_port.period_maximum_drawdown, performance_df['MaxDD'])
        
    def test_portfolio_performance_metrics_with_benchmark(self):
        # No trading status:
        price_weight_port = bt.portfolio(
            share=self.share,
            name='Price Weight', 
            benchmark=self.weight, 
            benchmark_name='Equal Weight',
            price=self.price  
        )
        performance_df = pd.DataFrame()

        for i in ['Price Weight', 'Equal Weight', 'Active']:
            if i == 'Price Weight':
                daily_ret = price_weight_port.port_daily_ret
            elif i == 'Equal Weight':
                daily_ret = price_weight_port.benchmark.port_daily_ret
            elif i == 'Active':
                daily_ret = price_weight_port.port_daily_ret - price_weight_port.benchmark.port_daily_ret

            performance_df.loc[i, 'Return'] = daily_ret.sum()
            performance_df.loc[i, 'Volatility'] = daily_ret.std()*sqrt(len(daily_ret))
            performance_df.loc[i, 'Sharpe'] = performance_df.loc[i, 'Return']/performance_df.loc[i, 'Volatility']
            if i != 'Active':
                hist_value = np.exp(daily_ret.cumsum())
            else:
                hist_value = price_weight_port.port_total_value - price_weight_port.benchmark.port_total_value
            previous_peak = hist_value.cummax()
            performance_df.loc[i, 'MaxDD'] = max(1 - hist_value/previous_peak)

        assert_series_equal(price_weight_port.period_return, performance_df['Return'])
        assert_series_equal(price_weight_port.period_volatility, performance_df['Volatility'])
        assert_series_equal(price_weight_port.period_sharpe_ratio, performance_df['Sharpe'])
        assert_series_equal(price_weight_port.period_maximum_drawdown, performance_df['MaxDD'])

        # No trading status:
        price_weight_port = bt.portfolio(
            share=self.share,
            name='Price Weight', 
            benchmark=self.weight, 
            benchmark_name='Equal Weight',
            price=self.price,
            trading_status=self.trading_status  
        )
        performance_df = pd.DataFrame()

        for i in ['Price Weight', 'Equal Weight', 'Active']:
            if i == 'Price Weight':
                daily_ret = price_weight_port.port_daily_ret
            elif i == 'Equal Weight':
                daily_ret = price_weight_port.benchmark.port_daily_ret
            elif i == 'Active':
                daily_ret = price_weight_port.port_daily_ret - price_weight_port.benchmark.port_daily_ret

            performance_df.loc[i, 'Return'] = daily_ret.sum()
            performance_df.loc[i, 'Volatility'] = daily_ret.std()*sqrt(len(daily_ret))
            performance_df.loc[i, 'Sharpe'] = performance_df.loc[i, 'Return']/performance_df.loc[i, 'Volatility']
            if i != 'Active':
                hist_value = np.exp(daily_ret.cumsum())
            else:
                hist_value = price_weight_port.port_total_value - price_weight_port.benchmark.port_total_value
            previous_peak = hist_value.cummax()
            performance_df.loc[i, 'MaxDD'] = max(1 - hist_value/previous_peak)

        assert_series_equal(price_weight_port.period_return, performance_df['Return'])
        assert_series_equal(price_weight_port.period_volatility, performance_df['Volatility'])
        assert_series_equal(price_weight_port.period_sharpe_ratio, performance_df['Sharpe'])
        assert_series_equal(price_weight_port.period_maximum_drawdown, performance_df['MaxDD'])

    def test_portfolio_performance_plot(self):
        pass

#%%%%%%%%%%%%%%%%%%%


# %%
