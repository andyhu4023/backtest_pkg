from math import sqrt
import pandas as pd


class Performance:
    """Return based analytic tool class"""

    def __init__(
        self, data, benchmark=None, name="Portfolio", benchmark_name="Benchmark"
    ) -> None:
        """Constructor for a performance instance

        Parameters
        -----------
        data: pd.Series
            value series of a portfolio
        benchmark: pd.Series, optional
            value series of a benchmark portfolio
        name: str, default "Portfolio"
            name of the porfolio
        benchmark_name: str, default "Benchmark"
            name of the benchmark portfolio

        Returns
        -------------
        None

        """
        assert (
            self.data.index == self.benchmark.index
        ).all(), "Benchmark should have the same dates (index)."
        self.data = data.ffill()
        self.benchmark = benchmark.ffill()
        self.name = name
        self.benchmark_name = benchmark_name

    def cal_total_return(self, ts):
        """Calcuate the total return over the period

        Parameters
        -----------

        Returns
        -------------

        """
        total = ts[-1] / ts[0]
        return total

    def cal_daily_returns(self, ts):
        daily = ts / ts.shift(1)
        daily.dropna(inplace=True)
        return daily

    def cal_volatility(self, ts):
        daily = self.cal_daily_returns(ts)
        vol = daily.std() / sqrt(len(daily))
        return vol

    def cal_sharpe_ratio(self, ts, risk_free=0):
        sharpe = (self.cal_total_return(ts) - risk_free) / self.cal_volatility(ts)
        return sharpe

    def cal_max_drawdown(self, ts):
        max_dd = max(1 - ts / ts.cummax())
        return max_dd

    @property
    def total_return(self):
        try:
            return self._total_return
        except AttributeError:
            self._total_return = self.cal_total_return(self.data)
        return self._total_return

    @property
    def benchmark_total_return(self):
        if self.benchmark is None:
            return 0

        try:
            return self._benchmark_total_return
        except AttributeError:
            self._benchmark_total_return = self.cal_total_return(self.benchmark)
        return self._benchmark_total_return

    @property
    def active_total_return(self):
        try:
            return self._active_total_return
        except AttributeError:
            self._active_total_return = self.total_return - self.benchmark_total_return
        return self._active_total_return

    @property
    def daily_returns(self):
        try:
            return self._daily_returns
        except AttributeError:
            self._daily_returns = self.cal_daily_returns(self.data)
        return self._daily_returns

    @property
    def benchmark_daily_returns(self):
        if self.benchmark is None:
            return 0

        try:
            return self._benchmark_daily_returns
        except AttributeError:
            self._benchmark_daily_returns = self.cal_daily_returns(self.benchmark)
        return self._benchmark_daily_returns

    @property
    def active_daily_returns(self):
        try:
            return self._active_daily_returns
        except AttributeError:
            self._active_daily_returns = (
                self.daily_returns - self.benchmark_daily_returns
            )
        return self._active_daily_returns

    @property
    def volatility(self):
        try:
            return self._volatility
        except AttributeError:
            self._volatility = self.cal_volatility(self.data)
        return self._volatility

    @property
    def benchmark_volatility(self):
        if self.benchmark is None:
            return 0

        try:
            return self._benchmark_volatility
        except AttributeError:
            self._benchmark_volatility = self.cal_volatility(self.benchmark)
        return self._benchmark_volatility

    @property
    def active_risk(self):
        try:
            return self._active_risk
        except AttributeError:
            self._active_risk = self.cal_volatility(self.active_daily_returns)
        return self._active_risk

    @property
    def sharpe_ratio(self):
        try:
            return self._sharpe_ratio
        except AttributeError:
            self._sharpe_ratio = self.cal_sharpe_ratio(self.data)
        return self._sharpe_ratio

    @property
    def benchmark_sharpe_ratio(self):
        if self.benchmark is None:
            return 0

        try:
            return self._benchmark_sharpe_ratio
        except AttributeError:
            self._benchmark_sharpe_ratio = self.cal_sharpe_ratio(self.benchmark)
        return self._benchmark_sharpe_ratio

    @property
    def information_ratio(self):
        try:
            return self._information_ratio
        except AttributeError:
            self._information_ratio = self.active_total_return / self.active_risk
        return self._information_ratio

    @property
    def max_drawdown(self):
        if self.benchmark is None:
            return 0

        try:
            return self._max_drawdown
        except AttributeError:
            self._max_drawdown = self.cal_max_drawdown(self.data)
        return self._max_drawdown

    @property
    def benchmark_max_drawdown(self):
        try:
            return self._benchmark_max_drawdown
        except AttributeError:
            self._benchmark_max_drawdown = self.cal_max_drawdown(self.benchmark)
        return self._benchmark_max_drawdown

    def summary(self):
        index = ["Total Return", "Volatility", "Sharpe Ratio", "Maximum Drawdown"]
        summary_list = list()
        portfolio_ser = pd.Series(
            [
                self.total_return,
                self.volatility,
                self.sharpe_ratio,
                self.max_drawdown,
            ],
            index=index,
            name=self.name,
        )
        summary_list.append(portfolio_ser.to_frame())

        if self.benchmark is not None:
            benchmark_ser = pd.Series(
                [
                    self.benchmark_total_return,
                    self.benchmark_volatility,
                    self.benchmark_sharpe_ratio,
                    self.benchmark_max_drawdown,
                ],
                index=index,
                name=self.benchmark_name,
            )
            summary_list.append(benchmark_ser.to_frame())

            active_ser = pd.Series(
                [
                    self.active_total_return,
                    self.active_risk,
                    self.information_ratio,
                    None,
                ],
                index=index,
                name="Active",
            )
            summary_list.append(active_ser.to_frame())

        summary_df = pd.concat(summary_list, axis=1)
        return summary_df
