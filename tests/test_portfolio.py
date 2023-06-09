import unittest
from pandas.testing import assert_frame_equal, assert_series_equal

import backtest_pkg as bt
import pandas as pd
import numpy as np
from math import sqrt, log, sin, pi
from utility import cal_std, cal_mean


class setup(unittest.TestCase):
    def setUp(self):
        n = 10  # length of the period
        price_dict = {
            "Up trend": list(range(1, n + 1)),
            "Down trend": list(range(n, 0, -1)),
            "Convex": list(1 + (n / 2) ** 2 + i * (i - n + 1) for i in range(n)),
            "Concave": list(1 + i * (n - 1 - i) for i in range(n)),
            "Sin": list(1 + n * (1 + sin(i / (n - 1) * 2 * pi)) for i in range(n)),
        }
        adj_price_df = pd.DataFrame(
            price_dict, index=pd.date_range("2020-01-01", periods=n, freq="D")
        )
        self.ticker = adj_price_df.columns
        self.index = adj_price_df.index
        self.prices = adj_price_df
        """ Price in values:
        'Up trend': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Down trend': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'Convex': [26, 18, 12, 8, 6, 6, 8, 12, 18, 26],
        'Concave': [1, 9, 15, 19, 21, 21, 19, 15, 9, 1],
        'Sin': [11, 17.42, 20.84, 19.66, 14.42, 7.57, 2.33, 1.15, 4.57, 11]
        """

        self.trading_status = pd.DataFrame(True, index=self.index, columns=self.ticker)
        self.trading_status.iloc[:3, 0] = False
        self.trading_status.iloc[3:6, 1] = False
        self.trading_status.iloc[6:, 2] = False

        # # Equal weight portfolio:
        self.weights = pd.DataFrame(1, index=self.index[[0, 5]], columns=self.ticker)
        # Equal weight asset values:
        # No rebal
        self.asset_values_no_rebal = self.prices.copy()
        self.asset_values_no_rebal = self.asset_values_no_rebal.apply(
            lambda ts: ts / ts[0], axis=0
        )
        # No rebal with trading status:
        self.asset_values_no_rebal_tst = self.asset_values_no_rebal.copy()
        self.asset_values_no_rebal_tst.iloc[:, 0] = 0
        # 1 rebal
        self.asset_values_1_rebal = self.asset_values_no_rebal.copy()
        rebal_value = self.asset_values_1_rebal.iloc[5, :].sum()
        after_rebal = self.asset_values_1_rebal.iloc[5:, :].copy()
        after_rebal = after_rebal.apply(lambda ts: ts / ts[0] * 0.2, axis=0)
        after_rebal *= rebal_value
        self.asset_values_1_rebal.iloc[5:, :] = after_rebal
        # 1 rebal with trading status:
        self.asset_values_1_rebal_tst = self.asset_values_no_rebal.copy()
        self.asset_values_1_rebal_tst.iloc[:, 0] = 0
        after_rebal_cols = [0, 2, 3, 4]
        after_rebal_value = self.asset_values_1_rebal_tst.iloc[
            5, after_rebal_cols
        ].sum()
        after_rebal_df = self.asset_values_no_rebal.iloc[5:, after_rebal_cols]
        after_rebal_df = (
            after_rebal_df.div(after_rebal_df.iloc[0, :], axis=1)
            / after_rebal_df.shape[1]
            * after_rebal_value
        )
        self.asset_values_1_rebal_tst.iloc[5:, after_rebal_cols] = after_rebal_df

        # Price weight portfolio:
        self.share = pd.DataFrame(1, index=self.index[[0, 5]], columns=self.ticker)
        # Price weight asset values:
        self.asset_values_no_rebal_share = self.prices.copy()
        self.asset_values_1_rebal_share = self.prices.copy()
        # With trading status:
        self.asset_values_no_rebal_share_tst = self.asset_values_no_rebal_share.copy()
        self.asset_values_no_rebal_share_tst.iloc[:, 0] = 0
        asset_values = self.asset_values_no_rebal_share.copy()
        asset_values.iloc[:5, 0] = 0
        adjust_factor = (
            asset_values.iloc[5, [2, 3, 4]].sum()
            / asset_values.iloc[5, [0, 2, 3, 4]].sum()
        )
        apply_col = [0, 2, 3, 4]
        asset_values.iloc[5:, apply_col] = asset_values.iloc[5:, apply_col].apply(
            lambda ts: ts * adjust_factor, axis=0
        )
        self.asset_values_1_rebal_share_tst = asset_values


class TestPortfolioConstruction(setup):
    def test_init_weight_set_price(self):
        port = bt.Portfolio(weights=self.weights)
        port.setup_trading(prices=self.prices)
        expect_status = pd.DataFrame(True, index=self.index, columns=self.ticker)
        assert_frame_equal(port.prices, self.prices)
        assert_frame_equal(port.trading_status, expect_status)

    def test_init_with_weights_prices(self):
        port = bt.Portfolio(weights=self.weights, prices=self.prices)
        expect_status = pd.DataFrame(True, index=self.index, columns=self.ticker)
        assert_frame_equal(port.prices, self.prices)
        assert_frame_equal(port.trading_status, expect_status)

    def test_init_with_weights_prices_and_trading_status(self):
        port = bt.Portfolio(
            weights=self.weights, prices=self.prices, trading_status=self.trading_status
        )
        assert_frame_equal(port.prices, self.prices)
        assert_frame_equal(port.trading_status, self.trading_status)

    def test_prices_with_NA(self):
        price = self.prices.where(self.trading_status, other=np.nan)
        port = bt.Portfolio(weights=self.weights)
        port.setup_trading(prices=price)
        assert_frame_equal(port.prices, price)
        assert_frame_equal(port.trading_status, self.trading_status)

    def test_indepedent_prices_trading_status(self):
        price = self.prices.copy()
        price.iloc[:5, 4] = np.nan
        expect_status = self.trading_status.copy()
        expect_status.iloc[:5, 4] = False
        port = bt.Portfolio(weights=self.weights)
        port.setup_trading(prices=price, trading_status=self.trading_status)
        assert_frame_equal(port.prices, price)
        assert_frame_equal(port.trading_status, expect_status)

    def test_out_range_trading_status(self):
        out_range_status = self.trading_status.copy()
        out_range_status["Extra Ticker"] = True
        out_range_status.loc[pd.to_datetime("2020-01-20"), :] = False
        expect_status = self.trading_status
        port = bt.Portfolio(weights=self.weights)
        port.setup_trading(prices=self.prices, trading_status=out_range_status)
        assert_frame_equal(port.trading_status, expect_status)

    def test_equal_weight(self):
        port = bt.Portfolio(weights=self.weights, prices=self.prices)
        expect = pd.DataFrame(0.2, index=self.index[[0, 5]], columns=self.ticker)
        assert_frame_equal(port.weights, expect)

    def test_zero_weight(self):
        weight = pd.DataFrame(0.2, index=self.index[[0, 5]], columns=self.ticker)
        weight.iloc[1, :] = 0
        port = bt.Portfolio(weights=weight, prices=self.prices)
        expect = weight.iloc[[0], :]
        assert_frame_equal(port.weights, expect)
        weight = pd.DataFrame(0.2, index=self.index[[0, 5]], columns=self.ticker)
        weight.iloc[0, :] = 0
        port = bt.Portfolio(weights=weight, prices=self.prices)
        expect = weight.iloc[[1], :]
        assert_frame_equal(port.weights, expect)

    def test_out_range_weight(self):
        out_range_weight = self.weights.copy()
        out_range_weight["Extra Ticker"] = 1
        out_range_weight.loc[pd.to_datetime("2020-01-20"), :] = 1
        port = bt.Portfolio(weights=out_range_weight, prices=self.prices)
        expect = pd.DataFrame(0.2, index=self.index[[0, 5]], columns=self.ticker)
        assert_frame_equal(port.weights, expect)

        # Weight cannot be set after initiation:
        with self.assertRaises(AttributeError):
            port.weights = self.weights

    def test_portfolio_weight_with_trading_status(self):
        # Weights on untradables:
        port = bt.Portfolio(
            weights=self.weights, prices=self.prices, trading_status=self.trading_status
        )
        expect = pd.DataFrame(0.25, index=self.index[[0, 5]], columns=self.ticker)
        expect.iloc[0, 0] = 0
        expect.iloc[1, 1] = 0
        assert_frame_equal(port.weights, expect)

        # Weights sum 0 from untradables:
        weight = pd.DataFrame(0.25, index=self.index[[0, 5]], columns=self.ticker)
        weight.iloc[0, 0] = 0
        weight.iloc[1, :] = [0, 1, 0, 0, 0]
        port = bt.Portfolio(
            weights=weight, prices=self.prices, trading_status=self.trading_status
        )
        expect = weight.iloc[[0], :]
        assert_frame_equal(port.weights, expect)

    def test_init_with_share(self):
        # No trading status:
        price_1 = [1, 10, 26, 1, 11]
        price_2 = [6, 5, 6, 21, 1 + 10 * (1 + sin(5 / 9 * 2 * pi))]
        weight_value = [
            [i / sum(price_1) for i in price_1],
            [j / sum(price_2) for j in price_2],
        ]
        expect = pd.DataFrame(
            weight_value, index=self.index[[0, 5]], columns=self.ticker
        )
        port = bt.Portfolio(shares=self.share, prices=self.prices)
        assert_frame_equal(port.weights, expect)

    def test_init_with_share_trading_status(self):
        # With trading status:
        price_1 = [0, 10, 26, 1, 11]
        price_2 = [6, 0, 6, 21, 1 + 10 * (1 + sin(5 / 9 * 2 * pi))]
        weight_value = [
            [i / sum(price_1) for i in price_1],
            [j / sum(price_2) for j in price_2],
        ]
        port = bt.Portfolio(
            shares=self.share, prices=self.prices, trading_status=self.trading_status
        )
        expect = pd.DataFrame(
            weight_value, index=self.index[[0, 5]], columns=self.ticker
        )
        assert_frame_equal(port.weights, expect)

    def test_default_end_date(self):
        port = bt.Portfolio(weights=self.weights, prices=self.prices)
        self.assertEqual(port.end_date, self.index[-1])

    def test_init_set_end_date(self):
        end_date = pd.to_datetime("2020-01-08")
        port = bt.Portfolio(weights=self.weights, prices=self.prices, end_date=end_date)
        self.assertEqual(port.end_date, end_date)

    def test_init_change_end_date(self):
        port = bt.Portfolio(weights=self.weights, prices=self.prices)
        self.assertEqual(port.end_date, self.index[-1])
        # Change end date after initiation:
        end_date = pd.to_datetime("2020-01-08")
        port.end_date = end_date
        self.assertEqual(port.end_date, end_date)


class TestPortfolioBacktest(setup):
    def test_daily_returns(self):
        price = pd.DataFrame(index=self.index)
        price["Normal"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Normal case
        price["Suspension"] = (
            [1, 2, 3] + [np.nan] * 3 + [4, 5, 6, 7]
        )  # Temporary suspension
        price["Delisting"] = [1, 2, 3, 4, 5] + [np.nan] * 5  # Delisting
        price["Late"] = [np.nan] * 5 + [1, 2, 3, 4, 5]  # Late listing:

        expect = pd.DataFrame(index=self.index)
        expect["Normal"] = [np.nan] + [((i + 1) / i) for i in range(1, 10)]
        expect["Suspension"] = (
            [np.nan]
            + [(2 / 1), (3 / 2)]
            + [1.0] * 3
            + [(4 / 3), (5 / 4), (6 / 5), (7 / 6)]
        )
        expect["Delisting"] = (
            [np.nan] + [((i + 1) / i) for i in range(1, 5)] + [1.0] * 5
        )
        expect["Late"] = [np.nan] * 6 + [((i + 1) / i) for i in range(1, 5)]

        port = bt.Portfolio(weights=self.weights, prices=price)
        assert_frame_equal(port.daily_returns, expect)

    def test_rebalance_normal(self):
        port = bt.Portfolio(weights=self.weights, prices=self.prices)
        initial_weights = self.weights.iloc[[0], :]
        rebalanced_weights = initial_weights.copy()
        rebalanced_weights.iloc[0, 1:4] = 2
        expect = pd.DataFrame(
            0,
            index=initial_weights.index,
            columns=initial_weights.columns,
        )
        expect.iloc[0, [0, 4]] = 1 / 8
        expect.iloc[0, [1, 2, 3]] = 1 / 4

        results = port.rebalance(
            initial_weights=initial_weights, rebalanced_weights=rebalanced_weights
        )
        assert_frame_equal(results, expect)

    def test_rebalance_untradable(self):
        trading_status = pd.DataFrame(
            True, index=self.prices.index, columns=self.prices.columns
        )
        trading_status.iloc[0, :2] = False
        port = bt.Portfolio(
            weights=self.weights, prices=self.prices, trading_status=trading_status
        )
        initial_weights = self.weights.iloc[[0], :]
        rebalanced_weights = initial_weights.copy()
        rebalanced_weights.iloc[0, 1:4] = 2
        expect = pd.DataFrame(
            0,
            index=initial_weights.index,
            columns=initial_weights.columns,
        )
        expect.iloc[0, [0, 1]] = 1 / 5
        expect.iloc[0, [2, 3]] = 0.24
        expect.iloc[0, [4]] = 0.12

        results = port.rebalance(
            initial_weights=initial_weights, rebalanced_weights=rebalanced_weights
        )
        assert_frame_equal(results, expect)

    def test_rebalance_no_tradable(self):
        trading_status = pd.DataFrame(
            False, index=self.prices.index, columns=self.prices.columns
        )
        port = bt.Portfolio(
            weights=self.weights, prices=self.prices, trading_status=trading_status
        )
        initial_weights = self.weights.iloc[[0], :]
        rebalanced_weights = initial_weights.copy()
        rebalanced_weights.iloc[0, 1:4] = 2
        expect = pd.DataFrame(
            0,
            index=initial_weights.index,
            columns=initial_weights.columns,
        )
        expect.iloc[0, :] = 1 / 5

        results = port.rebalance(
            initial_weights=initial_weights, rebalanced_weights=rebalanced_weights
        )
        assert_frame_equal(results, expect)

    def test_drift_no_rebalance(self):
        port = bt.Portfolio(weights=self.weights.iloc[[0], :], prices=self.prices)
        expect = self.asset_values_no_rebal.copy()
        expect = expect.apply(lambda ts: ts / ts.sum(), axis=1)
        assert_frame_equal(port.ex_weight, expect)

    def test_drift_with_rebalance(self):
        port = bt.Portfolio(weights=self.weights, prices=self.prices)
        expect = self.asset_values_1_rebal.copy()
        expect = expect.div(expect.sum(axis=1), axis=0)
        assert_frame_equal(port.ex_weight, expect)

    def test_drift_with_rebalance_trading_status(self):
        port = bt.Portfolio(
            weights=self.weights, prices=self.prices, trading_status=self.trading_status
        )
        expect = self.asset_values_1_rebal_tst.copy()
        expect = expect.div(expect.sum(axis=1), axis=0)
        assert_frame_equal(port.ex_weight, expect)

    def test_performance_no_rebal(self):
        port = bt.Portfolio(weights=self.weights.iloc[[0], :], prices=self.prices)
        port_value = self.asset_values_no_rebal.sum(axis=1).values
        expect_daily_ret = [1] + list(
            (port_value[i + 1] / port_value[i]) for i in range(len(self.index) - 1)
        )
        expect_daily_ret = pd.Series(expect_daily_ret, index=self.index)
        expect_total_ret = [1] + list(
            (port_value[i + 1] / port_value[0]) for i in range(len(self.index) - 1)
        )
        expect_total_ret = pd.Series(expect_total_ret, index=self.index)
        assert_series_equal(port.portfolio_returns, expect_daily_ret)
        assert_series_equal(port.portfolio_values, expect_total_ret)

    def test_performance_with_rebal(self):
        port = bt.Portfolio(weights=self.weights, prices=self.prices)
        port_value = self.asset_values_1_rebal.sum(axis=1).values
        expect_daily_ret = [1] + list(
            (port_value[i + 1] / port_value[i]) for i in range(len(self.index) - 1)
        )
        expect_daily_ret = pd.Series(expect_daily_ret, index=self.index)
        expect_total_ret = [1] + list(
            (port_value[i + 1] / port_value[0]) for i in range(len(self.index) - 1)
        )
        expect_total_ret = pd.Series(expect_total_ret, index=self.index)
        assert_series_equal(port.portfolio_returns, expect_daily_ret)
        assert_series_equal(port.portfolio_values, expect_total_ret)

    def test_performance_no_rebal_with_trading_status(self):
        port = bt.Portfolio(
            weights=self.weights.iloc[[0], :],
            prices=self.prices,
            trading_status=self.trading_status,
        )
        port_value = self.asset_values_no_rebal_tst.sum(axis=1).values
        expect_daily_ret = [1] + list(
            (port_value[i + 1] / port_value[i]) for i in range(len(self.index) - 1)
        )
        expect_daily_ret = pd.Series(expect_daily_ret, index=self.index)
        expect_total_ret = [1] + list(
            (port_value[i + 1] / port_value[0]) for i in range(len(self.index) - 1)
        )
        expect_total_ret = pd.Series(expect_total_ret, index=self.index)
        assert_series_equal(port.portfolio_returns, expect_daily_ret)
        assert_series_equal(port.portfolio_values, expect_total_ret)

    def test_performance_1_rebal_with_trading_status(self):
        port = bt.Portfolio(
            weights=self.weights, prices=self.prices, trading_status=self.trading_status
        )
        port_value = self.asset_values_1_rebal_tst.sum(axis=1).values
        expect_daily_ret = [1] + list(
            (port_value[i + 1] / port_value[i]) for i in range(len(self.index) - 1)
        )
        expect_daily_ret = pd.Series(expect_daily_ret, index=self.index)
        expect_total_ret = [1] + list(
            (port_value[i + 1] / port_value[0]) for i in range(len(self.index) - 1)
        )
        expect_total_ret = pd.Series(expect_total_ret, index=self.index)
        assert_series_equal(port.portfolio_returns, expect_daily_ret)
        assert_series_equal(port.portfolio_values, expect_total_ret)

    def test_portfolio_backtest(self):
        port = bt.Portfolio(weights=self.weights, prices=self.prices)
        port_value_ts = self.asset_values_1_rebal.sum(axis=1)
        port_value_ts = port_value_ts / port_value_ts[0]
        assert_frame_equal(port.backtest(), port_value_ts.to_frame(name="Portfolio"))

    def test_backtest_with_name(self):
        name = "Equal Weight"
        port = bt.Portfolio(weights=self.weights, prices=self.prices, name=name)
        port_value_ts = self.asset_values_1_rebal.sum(axis=1)
        port_value_ts = port_value_ts / port_value_ts[0]
        assert_frame_equal(port.backtest(), port_value_ts.to_frame(name=name))

    def test_backtest_with_trading_status(self):
        # Default setting, name = 'Portfolio'
        port = bt.Portfolio(
            weights=self.weights, prices=self.prices, trading_status=self.trading_status
        )
        port_value_ts = self.asset_values_1_rebal_tst.sum(axis=1)
        port_value_ts = port_value_ts / port_value_ts[0]
        assert_frame_equal(port.backtest(), port_value_ts.to_frame(name="Portfolio"))


class TestPortfolioPerformance(setup):
    def setUp(self):
        super().setUp()
        self.price_weight_port = bt.Portfolio(
            shares=self.share, prices=self.prices, name="Price Weight"
        )
        self.daily_ret = self.price_weight_port.portfolio_returns

    def test_return(self):
        result = self.price_weight_port.period_return
        expect = self.daily_ret.prod()
        assert_series_equal(result, expect)

    def test_volatility(self):
        result = self.price_weight_port.period_volatility
        expect = self.daily_ret.std() * sqrt(len(self.daily_ret))
        assert_series_equal(result, expect)

    def test_sharpe(self):
        result = self.price_weight_port.period_sharpe_ratio
        total_return = self.daily_ret.prod()
        volatility = self.daily_ret.std() * sqrt(len(self.daily_ret))
        expect = total_return / volatility
        assert_series_equal(result, expect)

    def test_max_drawdown(self):
        result = self.price_weight_port.period_maximum_drawdown
        hist_value = np.exp(self.daily_ret.cumsum())
        previous_peak = hist_value.cummax()
        expect = max(1 - hist_value / previous_peak)
        assert_series_equal(
            price_weight_port.period_maximum_drawdown, performance_df["MaxDD"]
        )

    def test_more(self):
        # No trading status:
        price_weight_port = bt.Portfolio(
            shares=self.share,
            prices=self.prices,
            name="Price Weight",
            trading_status=self.trading_status,
        )
        daily_ret = price_weight_port.portfolio_returns
        performance_df = pd.DataFrame(index=["Price Weight"])
        performance_df["Return"] = daily_ret.sum()
        performance_df["Volatility"] = daily_ret.std() * sqrt(len(daily_ret))
        performance_df["Sharpe"] = (
            performance_df["Return"] / performance_df["Volatility"]
        )
        hist_value = np.exp(daily_ret.cumsum())
        previous_peak = hist_value.cummax()
        performance_df["MaxDD"] = max(1 - hist_value / previous_peak)
        assert_series_equal(price_weight_port.period_return, performance_df["Return"])
        assert_series_equal(
            price_weight_port.period_volatility, performance_df["Volatility"]
        )
        assert_series_equal(
            price_weight_port.period_sharpe_ratio, performance_df["Sharpe"]
        )
        assert_series_equal(
            price_weight_port.period_maximum_drawdown, performance_df["MaxDD"]
        )


class TestPortfolioBenchmark(unittest.TestCase):
    def test_portfolio_with_benchmark(self):
        price_weight = self.prices.iloc[[0, 5], :].copy()
        price_weight = price_weight.apply(lambda ts: ts / ts.sum(), axis=1)
        equal_weight = pd.DataFrame(0.2, index=self.index[[0, 5]], columns=self.ticker)

        # Add benchmark at initiation:
        price_weight_port = bt.Portfolio(
            shares=self.share,
            prices=self.prices,
            name="Price Weight",
            benchmark=self.weights,
            benchmark_name="Equal Weight",
        )
        assert_frame_equal(price_weight_port.weight, price_weight)
        assert_frame_equal(price_weight_port.benchmark.weight, equal_weight)
        self.assertEqual(price_weight_port.name, "Price Weight")
        self.assertEqual(price_weight_port.benchmark.name, "Equal Weight")

        # Set benchmark after initiation:
        price_weight_port = bt.Portfolio(
            shares=self.share, prices=self.prices, name="Price Weight"
        )
        equal_weight_port = bt.Portfolio(
            weights=self.weights, prices=self.prices, name="Equal Weight"
        )
        price_weight_port.set_benchmark(equal_weight_port)
        assert_frame_equal(price_weight_port.weight, price_weight)
        assert_frame_equal(price_weight_port.benchmark.weight, equal_weight)
        self.assertEqual(price_weight_port.name, "Price Weight")
        self.assertEqual(price_weight_port.benchmark.name, "Equal Weight")

    def test_portfolio_backtest_with_benchmark(self):
        # No trading status:
        price_weight_port = bt.Portfolio(
            shares=self.share,
            prices=self.prices,
            name="Price Weight",
            benchmark=self.weights,
            benchmark_name="Equal Weight",
        )
        result = pd.DataFrame()
        port_value_ts = self.asset_values_1_rebal_share.sum(axis=1)
        bm_value_ts = self.asset_values_1_rebal.sum(axis=1)
        result["Price Weight"] = port_value_ts / port_value_ts[0]
        result["Equal Weight"] = bm_value_ts / bm_value_ts[0]
        result["Difference"] = result["Price Weight"] - result["Equal Weight"]
        assert_frame_equal(price_weight_port.backtest(), result)

        # With trading status:
        price_weight_port = bt.Portfolio(
            shares=self.share,
            prices=self.prices,
            trading_status=self.trading_status,
            name="Price Weight",
            benchmark=self.weights,
            benchmark_name="Equal Weight",
        )
        result = pd.DataFrame()
        port_value_ts = self.asset_values_1_rebal_share_tst.sum(axis=1)
        bm_value_ts = self.asset_values_1_rebal_tst.sum(axis=1)
        result["Price Weight"] = port_value_ts / port_value_ts[0]
        result["Equal Weight"] = bm_value_ts / bm_value_ts[0]
        result["Difference"] = result["Price Weight"] - result["Equal Weight"]
        assert_frame_equal(price_weight_port.backtest(), result)

    def test_portfolio_performance_metrics_with_benchmark(self):
        # No trading status:
        price_weight_port = bt.Portfolio(
            shares=self.share,
            name="Price Weight",
            benchmark=self.weights,
            benchmark_name="Equal Weight",
            prices=self.prices,
        )
        performance_df = pd.DataFrame()

        for i in ["Price Weight", "Equal Weight", "Active"]:
            if i == "Price Weight":
                daily_ret = price_weight_port.portfolio_returns
            elif i == "Equal Weight":
                daily_ret = price_weight_port.benchmark.port_daily_ret
            elif i == "Active":
                daily_ret = (
                    price_weight_port.portfolio_returns
                    - price_weight_port.benchmark.port_daily_ret
                )

            performance_df.loc[i, "Return"] = daily_ret.sum()
            performance_df.loc[i, "Volatility"] = daily_ret.std() * sqrt(len(daily_ret))
            performance_df.loc[i, "Sharpe"] = (
                performance_df.loc[i, "Return"] / performance_df.loc[i, "Volatility"]
            )
            if i != "Active":
                hist_value = np.exp(daily_ret.cumsum())
            else:
                hist_value = (
                    price_weight_port.port_total_value
                    - price_weight_port.benchmark.port_total_value
                )
            previous_peak = hist_value.cummax()
            performance_df.loc[i, "MaxDD"] = max(1 - hist_value / previous_peak)

        assert_series_equal(price_weight_port.period_return, performance_df["Return"])
        assert_series_equal(
            price_weight_port.period_volatility, performance_df["Volatility"]
        )
        assert_series_equal(
            price_weight_port.period_sharpe_ratio, performance_df["Sharpe"]
        )
        assert_series_equal(
            price_weight_port.period_maximum_drawdown, performance_df["MaxDD"]
        )

        # No trading status:
        price_weight_port = bt.Portfolio(
            shares=self.share,
            name="Price Weight",
            benchmark=self.weights,
            benchmark_name="Equal Weight",
            prices=self.prices,
            trading_status=self.trading_status,
        )
        performance_df = pd.DataFrame()

        for i in ["Price Weight", "Equal Weight", "Active"]:
            if i == "Price Weight":
                daily_ret = price_weight_port.portfolio_returns
            elif i == "Equal Weight":
                daily_ret = price_weight_port.benchmark.port_daily_ret
            elif i == "Active":
                daily_ret = (
                    price_weight_port.portfolio_returns
                    - price_weight_port.benchmark.port_daily_ret
                )

            performance_df.loc[i, "Return"] = daily_ret.sum()
            performance_df.loc[i, "Volatility"] = daily_ret.std() * sqrt(len(daily_ret))
            performance_df.loc[i, "Sharpe"] = (
                performance_df.loc[i, "Return"] / performance_df.loc[i, "Volatility"]
            )
            if i != "Active":
                hist_value = np.exp(daily_ret.cumsum())
            else:
                hist_value = (
                    price_weight_port.port_total_value
                    - price_weight_port.benchmark.port_total_value
                )
            previous_peak = hist_value.cummax()
            performance_df.loc[i, "MaxDD"] = max(1 - hist_value / previous_peak)

        assert_series_equal(price_weight_port.period_return, performance_df["Return"])
        assert_series_equal(
            price_weight_port.period_volatility, performance_df["Volatility"]
        )
        assert_series_equal(
            price_weight_port.period_sharpe_ratio, performance_df["Sharpe"]
        )
        assert_series_equal(
            price_weight_port.period_maximum_drawdown, performance_df["MaxDD"]
        )

    def test_portfolio_performance_plot(self):
        from pandas.plotting import register_matplotlib_converters

        register_matplotlib_converters()

        # Portfolio without benchamrk:
        price_weight_port = bt.Portfolio(
            shares=self.share,
            name="Price Weight",
            prices=self.prices,
        )
        price_weight_port.backtest()
        performance_plot = price_weight_port.performance_plot()  # figure object
        dates = performance_plot.axes[0].lines[0].get_xdata()
        total_vales = performance_plot.axes[0].lines[0].get_ydata()
        plot_ts = pd.Series(total_vales, index=dates)
        expect_ts = self.asset_values_1_rebal_share.sum(axis=1)
        expect_ts = expect_ts / expect_ts[0]
        assert_series_equal(plot_ts, expect_ts)

        # Portfolio with benchmark:
        price_weight_port = bt.Portfolio(
            shares=self.share,
            name="Price Weight",
            benchmark=self.weights,
            benchmark_name="Equal Weight",
            prices=self.prices,
        )
        price_weight_port.backtest()
        performance_plot = price_weight_port.performance_plot()  # figure object
        # Plot 1: portfolio values and benchmark values
        dates = performance_plot.axes[0].lines[0].get_xdata()
        port_vales = performance_plot.axes[0].lines[0].get_ydata()
        bm_vales = performance_plot.axes[0].lines[1].get_ydata()
        plot_port_ts = pd.Series(port_vales, index=dates)
        plot_bm_ts = pd.Series(bm_vales, index=dates)
        expect_port_ts = self.asset_values_1_rebal_share.sum(axis=1)
        expect_port_ts = expect_port_ts / expect_port_ts[0]
        expect_bm_ts = self.asset_values_1_rebal.sum(axis=1)
        expect_bm_ts = expect_bm_ts / expect_bm_ts[0]
        assert_series_equal(plot_port_ts, expect_port_ts)
        assert_series_equal(plot_bm_ts, expect_bm_ts)
        # Plot 2: Value differences
        diff_values = performance_plot.axes[1].lines[0].get_ydata()
        plot_diff_ts = pd.Series(diff_values, index=dates)
        expect_diff_ts = expect_port_ts - expect_bm_ts
        assert_series_equal(plot_diff_ts, expect_diff_ts)
