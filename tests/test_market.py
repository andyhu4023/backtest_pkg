import unittest
import backtest_pkg as bt
import pandas as pd
import numpy as np
from math import sqrt, log
from pandas.util.testing import assert_frame_equal
from utility import cal_mean, cal_std


class TestMarketSingleAsset(unittest.TestCase):
    def setUp(self):
        def construct_market(data):
            ticker = ["Test Ticker"]
            index = pd.date_range("2020-01-01", periods=len(data), freq="D")
            data_dict = dict(
                adj_close_price=pd.DataFrame(data, index=index, columns=ticker),
                open_price=pd.DataFrame(data, index=index, columns=ticker),
                high_price=pd.DataFrame(
                    [i * 1.1 for i in data], index=index, columns=ticker
                ),
                low_price=pd.DataFrame(
                    [i * 0.9 for i in data], index=index, columns=ticker
                ),
                close_price=pd.DataFrame(data, index=index, columns=ticker),
            )
            return bt.market(**data_dict)

        data_trend = [1, 2, 3, 4, 5]
        self.index = pd.date_range("2020-01-01", periods=len(data_trend), freq="D")
        self.ticker = ["Test Ticker"]
        self.market = construct_market(data_trend)
        self.market_down = construct_market(data_trend[::-1])
        data_sin = [3, 5, 3, 1, 3]
        data_convex = [3, 2, 1, 2, 3]
        data_concave = [1, 2, 3, 2, 1]
        self.market_sin = construct_market(data_sin)
        self.market_convex = construct_market(data_convex)
        self.market_concave = construct_market(data_concave)

    # Daily return: np.log([np.nan, 2/1, 3/2, 4/3, 5/4])
    def test_market_daily_ret(self):
        expect = pd.DataFrame(log(5 / 4), index=[self.index[-1]], columns=self.ticker)
        assert_frame_equal(self.market.daily_ret(), expect)

    def test_market_daily_ret_given_date(self):
        date_str = "2020-01-03"
        date = pd.to_datetime(date_str)
        expect = pd.DataFrame(log(3 / 2), index=[date], columns=self.ticker)
        assert_frame_equal(self.market.daily_ret(date=date), expect)
        assert_frame_equal(self.market.daily_ret(date=date_str), expect)

    def test_market_daily_ret_given_lag(self):
        lag = 1
        expect = pd.DataFrame(log(4 / 3), index=[self.index[-1]], columns=self.ticker)
        assert_frame_equal(self.market.daily_ret(lag=lag), expect)

    def test_market_daily_ret_given_date_lag(self):
        date = pd.to_datetime("2020-01-03")
        lag = 1
        expect = pd.DataFrame(log(2 / 1), index=[date], columns=self.ticker)
        assert_frame_equal(self.market.daily_ret(date=date, lag=lag), expect)

    def test_market_daily_ret_out_range_date(self):
        late_date = pd.to_datetime("2020-01-20")
        early_date = pd.to_datetime("2019-01-01")
        with self.assertRaises(AssertionError):
            self.market.daily_ret(date=early_date)
        with self.assertRaises(AssertionError):
            self.market.daily_ret(date=late_date)

    def test_market_daily_ret_large_lag(self):
        lag = 100
        expect = pd.DataFrame(np.nan, index=[self.index[-1]], columns=self.ticker)
        assert_frame_equal(self.market.daily_ret(lag=lag), expect)

    def test_market_daily_ret_negative_lag(self):
        lag = -1
        with self.assertRaises(AssertionError):
            self.market.daily_ret(lag=lag)

    def test_market_total_ret(self):
        expect = pd.DataFrame(log(5), index=[self.index[-1]], columns=self.ticker)
        assert_frame_equal(self.market.total_ret(), expect)

    def test_market_total_ret_given_date(self):
        date_str = "2020-01-03"
        date = pd.to_datetime(date_str)
        expect = pd.DataFrame(log(3), index=[date], columns=self.ticker)
        assert_frame_equal(self.market.total_ret(date=date), expect)
        assert_frame_equal(self.market.total_ret(date=date_str), expect)

    def test_market_total_ret_given_period(self):
        expect = pd.DataFrame(log(5 / 3), index=[self.index[-1]], columns=self.ticker)
        assert_frame_equal(self.market.total_ret(period=2), expect)

    def test_market_total_ret_given_date_period(self):
        date_str = "2020-01-04"
        date = pd.to_datetime(date_str)
        expect = pd.DataFrame(log(4 / 2), index=[date], columns=self.ticker)
        assert_frame_equal(self.market.total_ret(date=date, period=2), expect)

    def test_market_total_ret_out_range_date(self):
        late_date = pd.to_datetime("2020-01-20")
        early_date = pd.to_datetime("2019-01-01")
        with self.assertRaises(AssertionError):
            self.market.total_ret(date=early_date)
        with self.assertRaises(AssertionError):
            self.market.total_ret(date=late_date)

    def test_market_total_ret_large_period(self):
        with self.assertRaises(AssertionError):
            self.market.total_ret(period=100)

    def test_market_total_ret_negative_period(self):
        with self.assertRaises(AssertionError):
            self.market.total_ret(period=0)
        with self.assertRaises(AssertionError):
            self.market.total_ret(period=-1)

    def test_market_vol(self):
        data = [log(i) for i in [2 / 1, 3 / 2, 4 / 3, 5 / 4]]
        expect = pd.DataFrame(
            cal_std(data), index=[self.index[-1]], columns=self.ticker
        )
        assert_frame_equal(self.market.volatility(), expect)

    def test_market_vol_given_date(self):
        date_str = "2020-01-03"
        date = pd.to_datetime(date_str)
        data = [log(i) for i in [2 / 1, 3 / 2]]
        expect = pd.DataFrame(cal_std(data), index=[date], columns=self.ticker)
        assert_frame_equal(self.market.volatility(date=date), expect)
        assert_frame_equal(self.market.volatility(date=date_str), expect)

    def test_market_vol_given_period(self):
        data = [log(i) for i in [4 / 3, 5 / 4]]
        expect = pd.DataFrame(
            cal_std(data), index=[self.index[-1]], columns=self.ticker
        )
        assert_frame_equal(self.market.volatility(period=2), expect)

    def test_market_vol_given_date_period(self):
        date_str = "2020-01-04"
        date = pd.to_datetime(date_str)
        data = [log(i) for i in [3 / 2, 4 / 3]]
        expect = pd.DataFrame(cal_std(data), index=[date], columns=self.ticker)
        assert_frame_equal(self.market.volatility(date=date, period=2), expect)

    def test_market_vol_period_1(self):
        expect = pd.DataFrame(np.nan, index=[self.index[-1]], columns=self.ticker)
        assert_frame_equal(self.market.volatility(period=1), expect)

    def test_market_vol_out_range_period(self):
        with self.assertRaises(AssertionError):
            self.market.volatility(period=10)

    def test_market_bollinger(self):
        data_std = cal_std(list(range(1, 6)))
        expect = pd.DataFrame(
            (5 - 3) / data_std, index=[self.index[-1]], columns=self.ticker
        )
        assert_frame_equal(self.market.bollinger(), expect)

    def test_market_bollinger_given_date(self):
        date_str = "2020-01-03"
        date = pd.to_datetime(date_str)
        data = [1, 2, 3]
        expect = pd.DataFrame(
            (3 - 2) / cal_std(data), index=[date], columns=self.ticker
        )
        assert_frame_equal(self.market.bollinger(date=date), expect)
        assert_frame_equal(self.market.bollinger(date=date_str), expect)

    def test_market_bollinger_given_period(self):
        data = [3, 4, 5]
        expect = pd.DataFrame(
            (5 - 4) / cal_std(data), index=[self.index[-1]], columns=self.ticker
        )
        assert_frame_equal(self.market.bollinger(period=3), expect)

    def test_market_bollinger_given_date_period(self):
        date_str = "2020-01-04"
        date = pd.to_datetime(date_str)
        data = [2, 3, 4]
        expect = pd.DataFrame(
            (4 - 3) / cal_std(data), index=[date], columns=self.ticker
        )
        assert_frame_equal(self.market.bollinger(date=date, period=3), expect)

    def test_market_bollinger_down(self):
        data = [5, 4, 3, 2, 1]
        expect = pd.DataFrame(
            (data[-1] - cal_mean(data)) / cal_std(data),
            index=[self.index[-1]],
            columns=self.ticker,
        )
        assert_frame_equal(self.market_down.bollinger(), expect)

    def test_market_bollinger_sin(self):
        data = [3, 5, 3, 1, 3]
        expect = pd.DataFrame(
            (data[-1] - cal_mean(data)) / cal_std(data),
            index=[self.index[-1]],
            columns=self.ticker,
        )
        assert_frame_equal(self.market_sin.bollinger(), expect)

    def test_market_bollinger_convex(self):
        data = [3, 2, 1, 2, 3]
        expect = pd.DataFrame(
            (data[-1] - cal_mean(data)) / cal_std(data),
            index=[self.index[-1]],
            columns=self.ticker,
        )
        assert_frame_equal(self.market_convex.bollinger(), expect)

    def test_market_RSI(self):
        expect = pd.DataFrame(0.5, index=[self.index[-1]], columns=self.ticker)
        assert_frame_equal(self.market_convex.RSI(), expect)

    def test_market_RSI_given_date(self):
        date_str = "2020-01-03"
        date = pd.to_datetime(date_str)
        expect = pd.DataFrame(0.0, index=[date], columns=self.ticker)
        assert_frame_equal(self.market_convex.RSI(date=date), expect)
        assert_frame_equal(self.market_convex.RSI(date=date_str), expect)

    def test_market_RSI_given_period(self):
        expect = pd.DataFrame(1.0, index=[self.index[-1]], columns=self.ticker)
        assert_frame_equal(self.market_convex.RSI(period=2), expect)

    def test_market_RSI_given_date_period(self):
        date_str = "2020-01-04"
        date = pd.to_datetime(date_str)
        expect = pd.DataFrame(0.5, index=[date], columns=self.ticker)
        assert_frame_equal(self.market_convex.RSI(date=date, period=2), expect)

    def test_market_RSI_up(self):
        expect = pd.DataFrame(1.0, index=[self.index[-1]], columns=self.ticker)
        assert_frame_equal(self.market.RSI(), expect)

    def test_market_RSI_down(self):
        expect = pd.DataFrame(0.0, index=[self.index[-1]], columns=self.ticker)
        assert_frame_equal(self.market_down.RSI(), expect)

    def test_market_RSI_sin(self):
        expect = pd.DataFrame(0.5, index=[self.index[-1]], columns=self.ticker)
        assert_frame_equal(self.market_sin.RSI(), expect)

    def test_market_RSI_concave(self):
        expect = pd.DataFrame(0.5, index=[self.index[-1]], columns=self.ticker)
        assert_frame_equal(self.market_concave.RSI(), expect)

    def test_market_oscillator(self):
        expect = pd.DataFrame(
            (3 - 0.9) / (5.5 - 0.9), index=[self.index[-1]], columns=self.ticker
        )
        assert_frame_equal(self.market_sin.oscillator(), expect)

    def test_market_oscillator_given_date(self):
        date_str = "2020-01-03"
        date = pd.to_datetime(date_str)
        expect = pd.DataFrame(
            (3 - 2.7) / (5.5 - 2.7), index=[date], columns=self.ticker
        )
        assert_frame_equal(self.market_sin.oscillator(date=date), expect)
        assert_frame_equal(self.market_sin.oscillator(date=date_str), expect)

    def test_market_oscillator_given_period(self):
        expect = pd.DataFrame(
            (3 - 0.9) / (3.3 - 0.9), index=[self.index[-1]], columns=self.ticker
        )
        assert_frame_equal(self.market_sin.oscillator(period=3), expect)

    def test_market_oscillator_given_date_period(self):
        date_str = "2020-01-04"
        date = pd.to_datetime(date_str)
        expect = pd.DataFrame(
            (1 - 0.9) / (5.5 - 0.9), index=[date], columns=self.ticker
        )
        assert_frame_equal(self.market_sin.oscillator(date=date, period=3), expect)

    def test_market_oscillator_up(self):
        expect = pd.DataFrame(
            (5 - 0.9) / (5.5 - 0.9), index=[self.index[-1]], columns=self.ticker
        )
        assert_frame_equal(self.market.oscillator(), expect)

    def test_market_oscillator_down(self):
        expect = pd.DataFrame(
            (1 - 0.9) / (5.5 - 0.9), index=[self.index[-1]], columns=self.ticker
        )
        assert_frame_equal(self.market_down.oscillator(), expect)

    def test_market_oscillator_convex(self):
        expect = pd.DataFrame(
            (3 - 0.9) / (3.3 - 0.9), index=[self.index[-1]], columns=self.ticker
        )
        assert_frame_equal(self.market_convex.oscillator(), expect)

    def test_market_oscillator_concave(self):
        expect = pd.DataFrame(
            (1 - 0.9) / (3.3 - 0.9), index=[self.index[-1]], columns=self.ticker
        )
        assert_frame_equal(self.market_concave.oscillator(), expect)


class TestMarketMultiAsset(unittest.TestCase):
    def setUp(self):
        def construct_market(all_data):
            index = pd.date_range("2020-01-01", periods=5, freq="D")
            high = {k: [i * 1.1 for i in all_data[k]] for k in all_data}
            low = {k: [i * 0.9 for i in all_data[k]] for k in all_data}
            data_dict = dict(
                adj_close_price=pd.DataFrame(all_data, index=index),
                open_price=pd.DataFrame(all_data, index=index),
                high_price=pd.DataFrame(high, index=index),
                low_price=pd.DataFrame(low, index=index),
                close_price=pd.DataFrame(all_data, index=index),
            )
            return bt.market(**data_dict)

        self.index = pd.date_range("2020-01-01", periods=5, freq="D")
        self.all_data = {
            "Up Market": [1, 2, 3, 4, 5],
            "Down Market": [5, 4, 3, 2, 1],
            "Sin Market": [3, 5, 3, 1, 3],
            "Convex Market": [3, 2, 1, 2, 3],
            "Concave Market": [1, 2, 3, 2, 1],
        }
        self.ticker = self.all_data.keys()
        self.market = construct_market(self.all_data)

    def test_market_daily_ret(self):
        data = [5 / 4, 1 / 2, 3 / 1, 3 / 2, 1 / 2]
        expect = (
            pd.Series([log(i) for i in data], index=self.ticker)
            .to_frame(name=self.index[-1])
            .T
        )
        assert_frame_equal(self.market.daily_ret(), expect)

    def test_market_total_ret(self):
        data = [5 / 1, 1 / 5, 3 / 3, 3 / 3, 1 / 1]
        expect = (
            pd.Series([log(i) for i in data], index=self.ticker)
            .to_frame(name=self.index[-1])
            .T
        )
        assert_frame_equal(self.market.total_ret(), expect)

    def test_market_vol(self):
        def vol(data):
            ret = [log(data[i + 1] / data[i]) for i in range(len(data) - 1)]
            return cal_std(ret)

        expect = (
            pd.Series({k: vol(self.all_data[k]) for k in self.all_data})
            .to_frame(name=self.index[-1])
            .T
        )
        print(expect)
        assert_frame_equal(self.market.volatility(), expect)

    def test_market_bollinger(self):
        def z_score(data):
            return (data[-1] - cal_mean(data)) / cal_std(data)

        expect = (
            pd.Series({k: z_score(self.all_data[k]) for k in self.all_data})
            .to_frame(name=self.index[-1])
            .T
        )
        print(expect)
        assert_frame_equal(self.market.bollinger(), expect)

    def test_market_RSI(self):
        data = [1.0, 0.0, 0.5, 0.5, 0.5]
        expect = pd.Series(data, index=self.ticker).to_frame(name=self.index[-1]).T
        assert_frame_equal(self.market.RSI(), expect)

    def test_market_oscillator(self):
        data = [4.1 / 4.6, 0.1 / 4.6, 2.1 / 4.6, 2.1 / 2.4, 0.1 / 2.4]
        expect = pd.Series(data, index=self.ticker).to_frame(name=self.index[-1]).T
        assert_frame_equal(self.market.oscillator(), expect)


if __name__ == "__main__":
    unittest.main()
