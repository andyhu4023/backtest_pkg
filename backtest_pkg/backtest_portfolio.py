import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from math import sqrt


class Portfolio:
    """The main class for backtesting.
    The universe and the valid testing period will be defined by the price data.
    """

    def __init__(
        self,
        weights=None,
        shares=None,
        prices=None,
        trading_status=None,
        end_date=None,
        name="Portfolio",
        benchmark=None,
    ):
        """Create a Porfolio instance by weights or shares. Optionally input price, trading status, end date of testing, name of the portfolio and benchmark.

        Parameters
        ----------
        weights: pd.DataFrame
            Rebalancing dates (pd.Timestamp) as rows/index.
            Security identifiers (str) as columns.
            The target weights in the portfolio (float) as values.
            The weights may not be normalized, i.e. the sum of weigths may not be 1.
            Untradable security weights will be forced to 0.
        shares: pd.DataFrame
            Rebalancing dates (pd.Timestamp) as rows/index.
            Security identifiers (str) as columns.
            The target adjusted shares in the portfolio (int) as values.
            Untradable security shares will be forced to 0.
        prices: pd.DataFrame
            Trading dates in the whole testing period (pd.Timestamp) as rows/index.
            Security identifiers as columns.
            The adjusted closing price as (float) values.
        trading_status: pd.DataFrame
            Trading dates in the whole testing period (pd.Timestamp) as rows/index.
            Security identifiers as columns.
            Whether the security is tradable at the date (bool) as values.
        end_date: pd.Timestamp
            The end date of testing period.
            If not set, will be set as the last price date.
        name: str
            The name of the portfolio
        benchmark: Porfolio
            The benchmark portfolio to compare performances.

        Returns
        -------
        Portfolio
            A portfolio from weights or shares.
        """

        # Construct a portfolio by weights or shares:
        if (weights is not None) or (shares is not None):
            self.init_weights = weights
            self.init_shares = shares
        else:
            raise TypeError("Input at least one of weights or shares")
        self.name = name

        # Trading is setup by prices alone or prices & trading status
        if prices is not None:
            self.setup_trading(prices=prices, trading_status=trading_status)
        else:
            self.prices = None
            self.trading_status = None

        self.end_date = end_date
        self.benchmark = benchmark

    def setup_trading(self, prices, trading_status=None):
        """Setup prices and trading status for trading in testing.
        Prices and trading status should not be changed once setup for testing.

        Parameters
        -----------
        prices: pd.DataFrame
            Trading dates in the whole testing period (pd.Timestamp) as rows/index.
            Security identifiers as columns.
            The adjusted closing price as (float) values.
        trading_status: pd.DataFrame
            Trading dates in the whole testing period (pd.Timestamp) as rows/index.
            Security identifiers as columns.
            Whether the security is tradable at the date (bool) as values.

        Returns
        ---------
        None
        """
        #
        self.prices = prices

        if trading_status is None:
            self.trading_status = self.prices.notnull()
        else:
            trading_status = self._adjust(trading_status)
            self.trading_status = self.prices.notnull() & trading_status

        if self.end_date is None:
            self.end_date = max(self.prices.index)

    @property
    def shares(self):
        """Portfolio shares adjusted by prices and trading status"""
        try:
            return self._shares
        except AttributeError:
            if self.init_shares is not None:
                # Remove outrange dates and ids
                shares = self._adjust(self.init_shares)
                # Force untradable to 0
                shares.where(self.trading_status, other=0, inplace=True)
                self._shares = shares
            else:
                self._shares = None

            return self._shares

    @property
    def weights(self):
        """Portfolio weights adjusted by prices and trading status. Derived from init_weights or init_shares."""
        try:
            return self._weights
        except AttributeError:
            if self.init_weights is not None:
                # Remove outrange dates and ids
                weights = self._adjust(self.init_weights)
                # Force untradable to 0
                weights.where(self.trading_status, other=0, inplace=True)
                # Normalization
                self._weights = self.normalize(weights)
            elif self.init_shares is not None:
                shares = self.shares
                prices = self.prices.loc[shares.index, shares.columns].copy()
                weights = shares * prices
                # Normalization
                self._weights = self.normalize(weights)
            else:
                self._weights = None
        return self._weights

    def normalize(self, df):
        """A utility function to normalized dataframe rows, i.e. make row sums equal to 1.

        Parameters
        -----------
        df: pd.DataFrame
            data to adjust to align with price data

        Returns
        -------------
        pd.DataFrame
            A DataFrame
        """

        df = df.divide(df.sum(axis=1), axis=0)
        df.dropna(how="all", inplace=True)
        return df

    def _adjust(self, df):
        """A utility function to adjust a dataframe to align with prices data. Dates and ids not in prices will be removed.

        Parameters
        -----------
        df: pd.DataFrame
            data to adjust

        Returns
        -------------
        pd.DataFrame
            A DataFrame of dates and ids within prices dates and ids.
        """
        assert self.prices is not None, "No price data!"

        # Make index (dates) within price period
        out_range_date = df.index.difference(self.prices.index)
        if len(out_range_date) > 0:
            print(
                f'Skipping outrange dates:\n{[d.strftime("%Y-%m-%d") for d in out_range_date]}'
            )
            df = df.loc[df.index & self.prices.index, :]

        # Make columns (ids) within price universe
        unknown_id = df.columns.difference(self.prices.columns)
        if len(unknown_id) > 0:
            print(f"Removing unkown identifiers:\n{unknown_id.values}")
            df = df.loc[:, df.columns & self.prices.columns]
        return df

    #####################   Backtesting Calculations    ####################
    @property
    def daily_ret(self):
        try:
            return self._daily_ret
        except AttributeError:
            self._daily_ret = np.log(
                self.__prices.ffill() / self.__prices.ffill().shift(1)
            )
            return self._daily_ret

    def _drift_weight(self, initial_weight, rebalanced_weight=None, end=None):
        """
        initial_weight: weight before rebalance with shape (1, n)
        rebalanced_weight: weight after rebalance with shape (1, n), same index as initial weight.
        end: end date of the drifting period.
        """
        # Prepare end of drifting period:
        if end is None:
            end = self.end_date
        elif end > self.end_date:
            print(f"Invalid end date, set to {self.end_date} (portfolio end date)!")
            end = self.end_date

        ######################    Rebalance    ########################
        # Prepare the initial and rebalanced weight:
        assert initial_weight.shape[0] == 1, "Input weight with shape (1,n)"
        initial_weight_sum = initial_weight.iloc[0, :].sum()
        if initial_weight_sum == 1:
            pass
        elif initial_weight_sum == 0:
            initial_weight.iloc[0, :] = 0
        else:
            initial_weight.iloc[0, :] = initial_weight.iloc[0, :] / initial_weight_sum

        if rebalanced_weight is None:
            rebalanced_weight = initial_weight
        else:
            assert rebalanced_weight.shape[0] == 1, "Input weight with shape (1,n)"
            assert all(
                initial_weight.index == rebalanced_weight.index
            ), "Inconsistent weight data!"

            # Determine tradable tickers from self.trading_status:
            rebalanced_date = initial_weight.index[0]
            trading_status = self.trading_status.loc[[rebalanced_date], :]

            # Two weight vectors will be calcuate: one for rebalance, one for roll forward
            rebalanced_weight = rebalanced_weight.where(trading_status, other=0)
            roll_forward_weight = initial_weight.where(~trading_status, other=0)
            roll_forward_total = roll_forward_weight.iloc[0, :].sum()
            if roll_forward_total < 1:
                rebalanced_total = rebalanced_weight.iloc[0, :].sum()
                adjustment_factor = (1 - roll_forward_total) / rebalanced_total
                rebalanced_weight = rebalanced_weight * adjustment_factor
                rebalanced_weight = rebalanced_weight + roll_forward_weight
            else:
                rebalanced_weight = roll_forward_weight
            assert (
                abs(rebalanced_weight.iloc[0, :].sum() - 1) < 1e-4
            ), "Abnormal rebalanced weight!"

        ########################    Drifting   ##################
        # Prepare period price data:
        period_index = self.__prices.index
        period_index = period_index[
            (period_index >= initial_weight.index[0]) & (period_index <= end)
        ]
        period_price = self.__prices.loc[period_index, :].ffill()

        # Total returns:
        total_return = period_price / period_price.iloc[0, :]
        # Drifting weights:
        drift_weight = rebalanced_weight.reindex(period_index).ffill()
        drift_weight = drift_weight * total_return
        drift_weight = drift_weight.div(drift_weight.sum(axis=1), axis=0).fillna(0)

        return drift_weight

    @property
    def ex_weight(self):
        """
        Extend the weight to all dates before self.end_date.
        """
        try:
            return self._ex_weight
        except AttributeError:
            # Prepare the index after extention: (From first weight to end date)
            extend_period = self.__prices.index
            extend_period = extend_period[
                (extend_period >= self.weight.index[0])
                & (extend_period <= self.end_date)
            ]
            extend_weight = self.weight.reindex(extend_period)

            # Prepare the tuples for start and end date in each rebalancing period:
            rebalance_dates = pd.Series(self.weight.index)
            rebalance_start_end = zip(
                rebalance_dates,
                rebalance_dates.shift(-1, fill_value=pd.to_datetime(self.end_date)),
            )

            # Initial holdings are all 0:
            initial_weight = pd.DataFrame(
                0, index=[extend_period[0]], columns=self.__prices.columns
            )

            # Loop over each rebalancing period:
            for start, end in rebalance_start_end:
                rebalanced_weight = self.weight.loc[[start], :]
                period_weight = self._drift_weight(
                    initial_weight=initial_weight,
                    rebalanced_weight=rebalanced_weight,
                    end=end,
                )
                extend_weight.loc[start:end, :] = period_weight
                initial_weight = extend_weight.loc[[end], :]

            self._ex_weight = extend_weight
            return self._ex_weight

    @property
    def port_daily_ret(self):
        try:
            return self._port_daily_ret
        except AttributeError:
            daily_ret = self.daily_ret.copy()
            ex_weight = self.ex_weight
            daily_ret = daily_ret.loc[
                daily_ret.index & ex_weight.index, daily_ret.columns & ex_weight.columns
            ]

            port_daily_ret_values = np.log(
                (ex_weight.shift(1) * np.exp(daily_ret)).sum(axis=1)
            )
            port_daily_ret_values[0] = np.nan
            port_daily_ret = pd.Series(
                port_daily_ret_values, index=ex_weight.index
            ).fillna(0)
            self._port_daiy_ret = port_daily_ret
            return port_daily_ret

    @property
    def port_total_ret(self):
        try:
            return self._port_total_ret
        except AttributeError:
            self._port_total_ret = self.port_daily_ret.cumsum()
            return self._port_total_ret

    @property
    def port_total_value(self):
        return np.exp(self.port_total_ret)

    def backtest(self, plot=False):
        """
        Calculate portfolio performance. The period is from the first date of weight to end_date.
        """
        backtest_result = self.port_total_value.to_frame(name=self.name)
        if self.benchmark is not None:
            backtest_result[self.benchmark.name] = self.benchmark.port_total_value
            backtest_result["Difference"] = (
                backtest_result.iloc[:, 0] - backtest_result.iloc[:, 1]
            )
        self.backtest_result = backtest_result
        if plot:
            self.performance_plot()

        return self.backtest_result

    ####################    Performance Metrics     ######################
    @property
    def period_return(self):
        try:
            return self._period_return
        except AttributeError:
            self._period_return = pd.Series(name="Return")
            self._period_return[self.name] = self.port_total_ret[-1]
            if self.benchmark is not None:
                self._period_return[
                    self.benchmark.name
                ] = self.benchmark.port_total_ret[-1]
                self._period_return["Active"] = (
                    self._period_return[0] - self._period_return[1]
                )
            return self._period_return

    @property
    def period_volatility(self):
        try:
            return self._period_volatility
        except AttributeError:

            def vol(ts):
                return ts.std() * sqrt(len(ts))

            self._period_volatility = pd.Series(name="Volatility")
            self._period_volatility[self.name] = vol(self.port_daily_ret)
            if self.benchmark is not None:
                self._period_volatility[self.benchmark.name] = vol(
                    self.benchmark.port_daily_ret
                )
                self._period_volatility["Active"] = vol(
                    self.port_daily_ret - self.benchmark.port_daily_ret
                )
            return self._period_volatility

    @property
    def period_sharpe_ratio(self):
        try:
            return self._period_sharpe_ratio
        except AttributeError:
            self._period_sharpe_ratio = self.period_return / self.period_volatility
            self._period_sharpe_ratio.name = "Sharpe"
            return self._period_sharpe_ratio

    @property
    def period_maximum_drawdown(self):
        try:
            return self._period_maximum_drawdown
        except AttributeError:

            def mdd(ts):
                drawdown = 1 - ts / ts.cummax()
                return max(drawdown)

            self._period_maximum_drawdown = pd.Series(name="MaxDD")
            self._period_maximum_drawdown[self.name] = mdd(self.port_total_value)
            if self.benchmark is not None:
                self._period_maximum_drawdown[self.benchmark.name] = mdd(
                    self.benchmark.port_total_value
                )
                self._period_maximum_drawdown["Active"] = mdd(
                    self.port_total_value - self.benchmark.port_total_value
                )
            return self._period_maximum_drawdown

    def performance_summary(self):
        """
        Provide a table of total return, volitility, Sharpe ratio, maximun drawdown for portfoilo, benchmark and active (if any).
        """
        performance_summary_df = pd.DataFrame(
            dict(
                Return=self.period_return,
                Volatility=self.period_volatility,
                Sharpe=self.period_sharpe_ratio,
                MaxDD=self.period_maximum_drawdown,
            )
        )
        # performance_summary_df = performance_summary_df.style.format({
        #     'Return': '{:,.2%}'.format,
        #     'Volatility': '{:,.2%}'.format,
        #     'Sharpe': '{:,.2f}'.format,
        #     'MaxDD': '{:,.2%}'.format,
        # })
        return performance_summary_df

    def performance_plot(self):
        """
        For portfolio without benchmark, return one plot of performance
        For portfolio with benchmark, return two plots:
        1. The portfolio return and benchmark return over backtest period.
        2. The active return over the backtest period.
        """
        result = self.backtest_result
        assert (result.shape[1] == 1) or (
            result.shape[1] == 3
        ), "Invalid backtest results!"
        if result.shape[1] == 1:
            fig, ax1 = plt.subplots(1, 1)
            ax1.plot(result.iloc[:, 0], label=result.columns[0])
            ax1.tick_params(axis="x", rotation=25)
            ax1.grid(color="grey", ls="--")
            ax1.legend()
            ax1.set_title("Total Return")
        elif result.shape[1] == 3:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10))
            # make a little extra space between the subplots
            fig.subplots_adjust(hspace=0.5)

            # Upper figure for total return:
            ax1.plot(result.iloc[:, 0], label=result.columns[0])
            ax1.plot(result.iloc[:, 1], label=result.columns[1])
            ax1.tick_params(axis="x", rotation=25)
            ax1.grid(color="grey", ls="--")
            ax1.legend()
            ax1.set_title("Total Return")
            # Lower figure for active return:
            ax2.plot(result.iloc[:, 2])
            ax2.tick_params(axis="x", rotation=25)
            ax2.grid(color="grey", ls="--")
            ax2.set_title("Active Return")

        plt.show()
        return fig


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%
