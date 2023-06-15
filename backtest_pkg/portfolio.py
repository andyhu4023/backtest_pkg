import pandas as pd
from .performance import Performance


class Portfolio:
    """The portfolio strategy backtest class.

    A portfolio is constructed by one of weights or shares.
    Optionally, a benchmark portfolio can be set to compare performances.
    The market is defined by prices data and (optional) trading status.
    """

    ##############    Portfolio Construction    ##################
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
        """Constructor for a Porfolio instance.

        A portfolio is constructed by one of weights or shares. If both weights and shares are given, weights will be used. Optionally, a name for the portfolio can be given.
        Optionally (but highly recommended) prices, trading status for trading setup can be given. If not, trading setup should be done manually afterwards.
        Optionally the end date of backtest period can be given.
        Optionally a Portfolio instance as a benchmark can be given.

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
        prices: pd.DataFrame, optional
            Trading dates in the period (pd.Timestamp) as rows/index.
            Security identifiers as columns.
            The adjusted closing price (float) as values.
        trading_status: pd.DataFrame, optional
            Trading dates in the whole testing period (pd.Timestamp) as rows/index.
            Security identifiers as columns.
            Whether the security is tradable at the date (bool) as values.
            By default, the trading status is true if there are not-NA prices.
        end_date: pd.Timestamp, optional
            The end date of backtest period.
            By default, the end date is the last price date.
        name: str, optional
            The name of the portfolio.
            By default, the name is "Portfolio".
        benchmark: Porfolio, optional
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
        self.end_date = end_date

        # Trading is setup by prices alone or prices & trading status
        if prices is not None:
            self.setup_trading(prices=prices, trading_status=trading_status)
        else:
            self.prices = None
            self.trading_status = None

        if benchmark is not None:
            benchmark.setup_trading(prices=prices, trading_status=trading_status)
            self.benchmark = benchmark
        else:
            self.benchmark = None

    def setup_trading(self, prices, trading_status=None):
        """Setup trading information for backtest.

        Prices and trading status should not be changed once setup. The trading status is True if there are valid prices and tradable i.e. True in trading_status.

        Parameters
        -----------
        prices: pd.DataFrame, optional
            Trading dates in the period (pd.Timestamp) as rows/index.
            Security identifiers as columns.
            The adjusted closing price (float) as values.
        trading_status: pd.DataFrame, optional
            Trading dates in the whole testing period (pd.Timestamp) as rows/index.
            Security identifiers as columns.
            Whether the security is tradable at the date (bool) as values.
            By default, the trading status is true if there are not-NA prices.

        Returns
        ---------
        None
        """
        self.prices = prices

        if trading_status is None:
            self.trading_status = self.prices.notnull()
        else:
            trading_status = self._adjust(trading_status)
            self.trading_status = self.prices.notnull() & trading_status

        if self.end_date is None:
            self.end_date = max(self.prices.index)

        # Delete dependent attributes if already exist
        dependent_attr = (
            "_weights",
            "_daily_returns",
            "_ex_weights",
            "_portfolio_returns",
            "_portfolio_values",
        )
        for attr in dependent_attr:
            self.__dict__.pop(attr, None)

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
        """Portfolio weights adjusted by prices and trading status.
        Derived from init_weights or init_shares."""
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
        """Normalized dataframe rows.

        Return a dataframe with row abs sums equal to 1. The abs sums will allow long-short portfolio strategies. Zero rows will be dropped.

        Parameters
        -----------
        df: pd.DataFrame
            a dataframe with numerical values.

        Returns
        -------------
        pd.DataFrame
            A dataframe with row abs sums all equal to 1.
        """
        df = df.divide(df.abs().sum(axis=1), axis=0)
        df.dropna(how="all", inplace=True)
        return df

    def _adjust(self, df):
        """Adjust a dataframe to align with prices data.

        Align with prices data means the date index are subset of price index and the id columns are subset of price columns. Dates and ids not in prices data will be removed. A messgae for removed dates and ids will be printed.

        Parameters
        -----------
        df: pd.DataFrame
            a dataframe to align with price data

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

    #####################   Backtest Calculations    ####################
    @property
    def daily_returns(self):
        """daily returns calculated from forward filled prices."""
        try:
            return self._daily_returns
        except AttributeError:
            self._daily_returns = self.prices.ffill() / self.prices.ffill().shift(1)
            return self._daily_returns

    def rebalance(self, initial_weights, rebalanced_weights):
        """Rebalance based on trading status.

        Untradable securities will be rolled forwad, i.e. no change of weights. Remaining weights are distributed to tradable securites proportiion to its rebalanced weights. The index of initial weights and rebalanced weights should be the same.

        Parameters
        -----------
        initial_weights: pd.DataFrame, shape (1, n)
            Rebalance dates (pd.Timestamp) as index.
            Security identifiers (str) as columns.
            The initial weights (float) as values.
        rebalanced_weights: pd.DataFrame, shape (1, n)
            Rebalance dates (pd.Timestamp) as index.
            Security identifiers (str) as columns.
            The rebalanced weights (float) as values.

        Returns
        -------------
        pd.DataFrame, shapre (1, n)
            The trading status adjusted rebalanced weights.
        """
        # Preprocessing initial, rebalanced weights, trading status:
        assert initial_weights.shape[0] == 1, "Initial weights of shape (1,n)"
        assert all(
            initial_weights.abs().sum(axis=1) > 0
        ), "Invalid initial weights of all zeros!"
        assert rebalanced_weights.shape[0] == 1, "Rebalanced weights of shape (1,n)"
        assert all(
            rebalanced_weights.abs().sum(axis=1) > 0
        ), "Invalid rebalanced weights of all zeros!"
        initial_date = initial_weights.index[0]
        rebalanced_date = rebalanced_weights.index[0]
        assert initial_date == rebalanced_date, "Input weights in the same date"
        initial_weights = self.normalize(initial_weights)
        rebalanced_weights = self.normalize(rebalanced_weights)
        trading_status = self.trading_status.loc[[rebalanced_date], :]

        if trading_status.iloc[0, :].all():
            final_weights = rebalanced_weights
        elif trading_status.iloc[0, :].any():
            # Untradable security will roll forwad, i.e. no change of weights
            # Remaining weights are distributed to tradable by its rebalanced weights
            tradable_weights = rebalanced_weights.where(trading_status, other=0).copy()
            roll_forward_weights = initial_weights.where(
                ~trading_status, other=0
            ).copy()
            roll_forward_sum = roll_forward_weights.iloc[0, :].sum()
            tradable_weights = self.normalize(tradable_weights) * (1 - roll_forward_sum)
            final_weights = tradable_weights + roll_forward_weights
        else:
            final_weights = initial_weights

        assert all(
            abs(final_weights.sum(axis=1) - 1) < 1e-4
        ), "Abnormal rebalanced weight!"
        return final_weights

    def drift(self, initial_weights, end_date=None):
        """Drift weights over a period.

        Parameters
        -----------
        initial_weights: pd.DataFrame, shape (1, n)
            Rebalance dates (pd.Timestamp) as index.
            Security identifiers (str) as columns.
            The initial weights (float) as values.
        end_date: pd.Timestamp
            end date of the drifting period.
            By default, the end date is the end date of backtest period.
            If end date is after backtest end date, it is set to backtest end date.

        Returns
        -------------
        pd.DataFrame
            Dates from rebalanced dates to end date as index.
            Security identifiers as columns
            The weights of security at given date as values.
        """
        # Prepare end date
        if end_date is None:
            end_date = self.end_date
        elif end_date > self.end_date:
            print(f"End date is set to {self.end_date:%Y-%m-%d} (backtest end date)!")
            end_date = self.end_date

        # Prepare period prices and returns
        period_index = self.prices.index
        period_index = period_index[
            (period_index >= initial_weights.index[0]) & (period_index <= end_date)
        ]
        period_prices = self.prices.loc[period_index, :].ffill()
        period_retuns = period_prices.div(period_prices.iloc[0, :], axis=1)

        # Drift
        drift_weights = initial_weights.reindex(period_index).ffill()
        drift_weights = drift_weights * period_retuns
        drift_weights = self.normalize(drift_weights)

        return drift_weights

    @property
    def ex_weight(self):
        """Extend the weights from rebalance dates to all backtest dates"""
        try:
            return self._ex_weight
        except AttributeError:
            # Prepare the index after extention: (From first weight to end date)
            start_date = self.weights.index[0]
            end_date = self.end_date
            extend_period = self.prices.index.copy()
            extend_period = extend_period[
                (extend_period >= start_date) & (extend_period <= end_date)
            ]

            # Prepare the tuples for start and end date in each rebalancing period:
            rebalance_dates = pd.Series(self.weights.index)
            rebalance_start_end = zip(
                rebalance_dates,
                rebalance_dates.shift(-1, fill_value=pd.to_datetime(self.end_date)),
            )

            # Drift in each rebalancing period: (All zero initially)
            self._ex_weight = pd.DataFrame(
                0, index=extend_period, columns=self.weights.columns
            )
            self._ex_weight.iloc[0, :] = self.weights.iloc[0, :]
            for start, end in rebalance_start_end:
                initial_weights = self._ex_weight.loc[[start], :]
                rebalanced_weights = self.weights.loc[[start], :]
                rebalanced_weights = self.rebalance(
                    initial_weights=initial_weights,
                    rebalanced_weights=rebalanced_weights,
                )
                period_weights = self.drift(
                    initial_weights=rebalanced_weights,
                    end_date=end,
                )
                self._ex_weight.loc[start:end, :] = period_weights

            return self._ex_weight

    @property
    def portfolio_returns(self):
        """Time series of portfolio daily returns"""
        try:
            return self._portfolio_returns
        except AttributeError:
            ex_weight = self.ex_weight
            daily_ret = self.daily_returns
            daily_ret = daily_ret.loc[
                daily_ret.index & ex_weight.index, daily_ret.columns & ex_weight.columns
            ]

            self._portfolio_returns = (ex_weight.shift(1) * (daily_ret)).sum(axis=1)
            self._portfolio_returns[0] = 1
            return self._portfolio_returns

    @property
    def portfolio_values(self):
        """Time series of portfolio value with base 1"""
        try:
            return self._port_total_ret
        except AttributeError:
            self._port_total_ret = self.portfolio_returns.cumprod()
            return self._port_total_ret

    def backtest(self):
        """Calculate portfolio performances over backtest period."""
        backtest_result = self.portfolio_values.to_frame(name=self.name)
        if self.benchmark is not None:
            backtest_result[self.benchmark.name] = self.benchmark.portfolio_values
            backtest_result["Difference"] = (
                backtest_result.iloc[:, 0] - backtest_result.iloc[:, 1]
            )
        self.backtest_result = backtest_result

        return self.backtest_result

    ####################    Performance Metrics     ######################
    def adjust_start_end(self, start=None, end=None):
        """Adjust start and end dates to align with prices

        Parameters
        -----------
        start: pd.Timestamp
            start date of a period
        end: pd.Timestamp
            end date of a period

        Returns
        -------------
        tuple of 2 pd.Timestamp
            a tuple pair of adjusted start and end date

        """
        if (start is None) or (start < self.prices.index[0]):
            start = self.prices.index[0]
        if (end is None) or (end > self.end_date):
            end = self.end_date

        return self.portfolio_values[start:end]

    def period_return(self, start=None, end=None):
        """Calculate the period total return

        Parameters
        -----------
        start: pd.Timestamp
            start date of a period
        end: pd.Timestamp
            end date of a period

        Returns
        -------------
        float
            Total return of the portfolio over the period
        """
        ts = self.adjust_start_end(start=start, end=end)
        return Performance(ts).total_return

    def period_volatility(self, start=None, end=None):
        """Calculate the period volatitliy

        Parameters
        -----------
        start: pd.Timestamp
            start date of a period
        end: pd.Timestamp
            end date of a period

        Returns
        -------------
        float
            Return volatility of the portfolio over the period
        """
        ts = self.adjust_start_end(start=start, end=end)
        return Performance(ts).volatility

    def period_sharpe_ratio(self, start=None, end=None):
        """Calculate the period Sharpe ratio

        The Sharpe ratio is considered as the risk adjusted excess returns. Risk free rate is set as 0.
        More about Sharpe ratio can be found [here](https://en.wikipedia.org/wiki/Sharpe_ratio)

        Parameters
        -----------
        start: pd.Timestamp
            start date of a period
        end: pd.Timestamp
            end date of a period

        Returns
        -------------
            Sharpe ratio of the portfolio over the period.
        """
        ts = self.adjust_start_end(start=start, end=end)
        return Performance(ts).sharpe_ratio

    def period_maximum_drawdown(self, start=None, end=None):
        """Calculate the maximun drawdown in a period.

        The maximun drawdown is consider as the maximum loss by holding the porfolio.
        More about maximun drawdown can be found [here](https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp#:~:text=Maximum%20drawdown%20(MDD)%20is%20a,down%20movements%20could%20be%20volatile.)

        Parameters
        -----------
        start: pd.Timestamp
            start date of a period
        end: pd.Timestamp
            end date of a period

        Returns
        -------------
        float
            Maximum drawdown of the portfolio over a period
        """
        ts = self.adjust_start_end(start=start, end=end)
        return Performance(ts).max_drawdown

    def performance_summary(self):
        """Return a table of performance metrics.

        The table contains total return, volitility, Sharpe ratio and maximun drawdown.
        If benchmark are set, benchmark metrics and active metrics are also presented.

        Parameters
        ----------
        None

        Returns
        ----------
        pd.Dateframe
            A summary table
        """
        try:
            return self._performance.summary()
        except AttributeError:
            if self.benchmark is None:
                self._performance = Performance(self.portfolio_values)
            else:
                self._performance = Performance(
                    self.portfolio_values, self.benchmark.portfolio_values
                )
            return self._performance.summary()
