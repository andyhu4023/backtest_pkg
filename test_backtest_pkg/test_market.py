import unittest
import backtest_pkg as bt 
import pandas as pd 
import numpy as np
from math import sqrt, log
from pandas.util.testing import assert_frame_equal

class TestMarketSingleAsset(unittest.TestCase):
    def setUp(self):
        ticker = ['Test Ticker']
        index = index=pd.date_range('2020-01-01', periods=5, freq='D')
        open_values = [1, 2, 3, 4, 5]
        close_values = [1.1, 2.2, 3.3, 4.4, 5.5]
        high_values = [i+0.5 for i in open_values]
        low_values = [i-0.5 for i in open_values]

        adj_close_price = pd.DataFrame(close_values, index=index, columns=ticker)
        open_price = pd.DataFrame(open_values, index=index, columns=ticker)
        high_price = pd.DataFrame(high_values, index=index, columns=ticker)
        low_price = pd.DataFrame(low_values, index=index, columns=ticker)
        close_price = pd.DataFrame(close_values, index=index, columns=ticker)

        self.market = bt.market(adj_close_price = adj_close_price, open_price = open_price, high_price = high_price, low_price=low_price, close_price=close_price)
        self.index = index
        self.ticker = ticker

    # Daily return: np.log([np.nan, 2/1, 3/2, 4/3, 5/4])
    def test_market_daily_ret(self):
        expect = pd.DataFrame(log(5/4), index=[self.index[-1]], columns=self.ticker)
        assert_frame_equal(self.market.daily_ret(), expect)
    def test_market_daily_ret_given_date(self):
        date_str = '2020-01-03'
        date = pd.to_datetime(date_str)
        expect = pd.DataFrame(log(3/2), index=[date], columns=self.ticker)
        assert_frame_equal(self.market.daily_ret(date=date), expect)
        assert_frame_equal(self.market.daily_ret(date=date_str), expect)
    def test_market_daily_ret_given_lag(self):
        lag = 1
        expect = pd.DataFrame(log(4/3), index=[self.index[-1]], columns=self.ticker)
        assert_frame_equal(self.market.daily_ret(lag=lag), expect)
    def test_market_daily_ret_given_date_lag(self):
        date = pd.to_datetime('2020-01-03')
        lag = 1
        expect = pd.DataFrame(log(2/1), index=[date], columns=self.ticker)
        assert_frame_equal(self.market.daily_ret(date=date, lag=lag), expect)
    def test_market_daily_ret_out_range_date(self):
        date = pd.to_datetime('2020-01-20')
        with self.assertRaises(AssertionError):
            self.market.daily_ret(date=date)

    def test_market_total_ret(self):
        expect = pd.DataFrame(log(5), index=[self.index[-1]], columns=self.ticker)
        assert_frame_equal(self.market.total_ret(), expect)
    def test_market_total_ret_given_date(self):
        date_str = '2020-01-03'
        date = pd.to_datetime(date_str)
        expect = pd.DataFrame(log(3), index=[date], columns=self.ticker)
        assert_frame_equal(self.market.total_ret(date=date), expect)
        assert_frame_equal(self.market.total_ret(date=date_str), expect)
    def test_market_total_ret_given_period(self):
        expect = pd.DataFrame(log(5/3), index=[self.index[-1]], columns=self.ticker)
        assert_frame_equal(self.market.total_ret(period=2), expect)
    def test_market_total_ret_given_date_period(self):
        date_str = '2020-01-04'
        date = pd.to_datetime(date_str)
        expect = pd.DataFrame(log(4/2), index=[date], columns=self.ticker)
        assert_frame_equal(self.market.total_ret(date = date, period=2), expect)
    def test_market_total_ret_out_range_period(self):
        with self.assertRaises(AssertionError):
            self.market.total_ret(period=10)

    def cal_std(self, data):
        if len(data)<=1:
            return np.nan
        data_mean = sum(data)/len(data)
        data_var = sum((i-data_mean)**2 for i in data)/(len(data)-1)
        return sqrt(data_var)

    def test_market_vol(self):
        data = [log(i) for i in [2/1, 3/2, 4/3, 5/4]]
        expect = pd.DataFrame(self.cal_std(data), index=[self.index[-1]], columns=self.ticker)
        assert_frame_equal(self.market.volatility(), expect)
    def test_market_vol_given_date(self):
        date_str = '2020-01-03'
        date = pd.to_datetime(date_str)
        data = [log(i) for i in [2/1, 3/2]]
        expect = pd.DataFrame(self.cal_std(data), index=[date], columns=self.ticker)
        assert_frame_equal(self.market.volatility(date=date), expect)
        assert_frame_equal(self.market.volatility(date=date_str), expect)
    def test_market_vol_given_period(self):
        data = [log(i) for i in [4/3, 5/4]]
        expect = pd.DataFrame(self.cal_std(data), index=[self.index[-1]], columns=self.ticker)
        assert_frame_equal(self.market.volatility(period=2), expect)
    def test_market_vol_given_date_period(self):
        date_str = '2020-01-04'
        date = pd.to_datetime(date_str)
        data = [log(i) for i in [3/2, 4/3]]
        expect = pd.DataFrame(self.cal_std(data), index=[date], columns=self.ticker)
        assert_frame_equal(self.market.volatility(date=date, period=2), expect)
    def test_market_vol_period_1(self):
        expect = pd.DataFrame(np.nan, index=[self.index[-1]], columns=self.ticker)
        assert_frame_equal(self.market.volatility(period=1), expect)
    def test_market_vol_out_range_period(self):
        with self.assertRaises(AssertionError):
            self.market.volatility(period=10)

    # def test_market_bollinger(self):
    #     pass

    # def test_market_oscillator(self):
    #     pass

    # def test_market_RSI(self):
    #     pass

if __name__=='__main__':
    unittest.main()
