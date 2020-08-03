import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import warnings
from math import sqrt


class portfolio:
    '''
    The universe and the valid testing period will be defined by the price data.
    '''
    def __init__(self, weight=None, share=None, benchmark=None, end_date=None, name='Portfolio', benchmark_name='Benchmark', price=None, trading_status=None):
        '''
        weight: a df with row-names date, col-name security id, value the portfolio weight (not necessarily normalized) of col-security at row-date. 
        share: a df with row-names date, col-name security id, value the portfolio shares of col-security at row date. 
        benchmark: a df of benchmark weight or a portfolio object
        end_date: date to end backtest 
        name: the name of the portfolio
        benchmark_name: the name of the benchmark
        '''
        # Price and trading status:
        if price is not None:
            self.set_price(price, trading_status)

        # Construct a portfolio from weight or share:
        if weight is not None:
            self._weight = weight
            self.normalized = False
        elif share is not None:
            self.share = share
            self._weight = self.weight_from_share(share)
        else:
            raise TypeError('Input at least one of weight or share')
        self._end_date = end_date
        self.name = name

        # Setting benchmark from weight df or portfolio object:
        if benchmark is None:
            self.benchmark = None
        else:
            self.set_benchmark(benchmark, benchmark_name)
        

    def set_benchmark(self, benchmark, benchmark_name='Benchmark'):
        if isinstance(benchmark, pd.DataFrame):
            self.benchmark = portfolio(
                weight=benchmark, 
                name=benchmark_name, 
                end_date=self.end_date, 
                price=self.price, 
                trading_status=self.trading_status
            )
        elif isinstance(benchmark, portfolio):
            self.benchmark = benchmark
            self.benchmark.set_price(self.price, self.trading_status)
            self.benchmark.end_date= self.end_date
        else:
            raise TypeError('Unkown benchmark!')


    def set_price(self, price, trading_status=None):
        '''
        price_data: a df with row-names date, col-name security id, value the price of col-security at row-date. 
        trading_status: a df with row-names date, col-name security id, boolean value indicate if col-security is tradable at row-date. 
        '''
        # Price and trading status is const, should not be change once set.
        self.__price = price
        if trading_status is None:
            self.__trading_status = self.__price.notnull()
        else:
            trading_status = self._adjust(trading_status)
            self.__trading_status = self.__price.notnull() & trading_status
    @property
    def price(self):
        return self.__price
    @property
    def trading_status(self): 
        return self.__trading_status

    # Utility function to align df with price:
    def _adjust(self, df):
        assert self.__price is not None, "No price data!"
        # Adjust index(dates) withing price.index
        out_range_date = df.index.difference(self.__price.index)
        if len(out_range_date)>0:
            print(f'Skipping outrange dates:\n{[d.strftime("%Y-%m-%d") for d in out_range_date]}')
            df = df.loc[df.index & self.__price.index, :]
        # Adjust columns(tickers) withing price.columns, 
        unknown_ticker = df.columns.difference(self.__price.columns)
        if len(unknown_ticker)>0:
            print(f'Removing unkown tickers:\n{unknown_ticker.values}')
            df = df.loc[:, df.columns & self.__price.columns]
        return df


    @property
    def weight(self):
        assert self.__price is not None, 'No price data!'
        # Normalization process:
        if not self.normalized:
            self._weight = self._adjust(self._weight)
            self._weight = self._weight.where(self.trading_status, other = 0)  # Set weight 0 if trading status is false
            self._weight = self._weight.divide(self._weight.sum(axis=1), axis=0)  # Normalization
            self._weight = self._weight.dropna(how='all')  # Drop rows with sum==0.
            self.normalized= True
  
        return self._weight
  
    def weight_from_share(self, share):
        share = self._adjust(share)
        price_data = self.__price.copy().loc[share.index, share.columns]
        self._weight = self.share * price_data
        self.normalized = False
        return self.weight

    @property
    def end_date(self):
        if self._end_date is None:
            assert self.__price is not None, 'No price data!'
            self._end_date = max(self.__price.index)
        return self._end_date
    @end_date.setter 
    def end_date(self, value):
        self._end_date = value

#####################   Backtesting Calculations    ####################
    @property
    def daily_ret(self):
        try:
            return self._daily_ret
        except AttributeError:
            self._daily_ret = np.log(self.__price.ffill()/self.__price.ffill().shift(1))
            return self._daily_ret


    def _drift_weight(self, initial_weight, rebalanced_weight=None, end=None):
        '''
        initial_weight: weight before rebalance with shape (1, n)
        rebalanced_weight: weight after rebalance with shape (1, n), same index as initial weight.
        end: end date of the drifting period.
        '''
        # Prepare end of drifting period:
        if end is None:
            end = self.end_date
        elif end > self.end_date:
            print(f'Invalid end date, set to {self.end_date} (portfolio end date)!')
            end = self.end_date

        ######################    Rebalance    ########################
        # Prepare the initial and rebalanced weight:
        assert initial_weight.shape[0]==1, 'Input weight with shape (1,n)'
        initial_weight_sum = initial_weight.iloc[0, :].sum()
        if initial_weight_sum==1:
            pass
        elif initial_weight_sum==0:
            initial_weight.iloc[0, :] = 0
        else:
            initial_weight.iloc[0, :] = initial_weight.iloc[0, :]/initial_weight_sum
        
        if rebalanced_weight is None:
            rebalanced_weight = initial_weight
        else:
            assert rebalanced_weight.shape[0]==1, 'Input weight with shape (1,n)'
            assert all(initial_weight.index == rebalanced_weight.index), 'Inconsistent weight data!'

            # Determine tradable tickers from self.trading_status:
            rebalanced_date = initial_weight.index[0]
            trading_status = self.trading_status.loc[[rebalanced_date], :]

            # Two weight vectors will be calcuate: one for rebalance, one for roll forward
            rebalanced_weight = rebalanced_weight.where(trading_status, other=0)
            roll_forward_weight = initial_weight.where(~trading_status, other=0)
            roll_forward_total = roll_forward_weight.iloc[0, :].sum()
            if roll_forward_total<1:
                rebalanced_total = rebalanced_weight.iloc[0, :].sum()
                adjustment_factor = (1-roll_forward_total)/rebalanced_total
                rebalanced_weight = rebalanced_weight*adjustment_factor
                rebalanced_weight = rebalanced_weight+roll_forward_weight
            else:
                rebalanced_weight = roll_forward_weight
            assert abs(rebalanced_weight.iloc[0, :].sum()-1)<1e-4, 'Abnormal rebalanced weight!'

        ########################    Drifting   ##################
        # Prepare period price data:
        period_index = self.__price.index
        period_index = period_index[(period_index>=initial_weight.index[0]) & (period_index<=end)]
        period_price = self.__price.loc[period_index, :].ffill()

        # Total returns:
        total_return = period_price/period_price.iloc[0,:]
        # Drifting weights:
        drift_weight = rebalanced_weight.reindex(period_index).ffill()
        drift_weight = drift_weight * total_return 
        drift_weight = drift_weight.div(drift_weight.sum(axis=1), axis=0).fillna(0)

        return drift_weight

    @property
    def ex_weight(self):
        '''
        Extend the weight to all dates before self.end_date.
        '''
        try:
            return self._ex_weight
        except AttributeError:
            # Prepare the index after extention: (From first weight to end date)
            extend_period = self.__price.index
            extend_period = extend_period[(extend_period>=self.weight.index[0])&(extend_period<=self.end_date)]
            extend_weight = self.weight.reindex(extend_period)

            # Prepare the tuples for start and end date in each rebalancing period:
            rebalance_dates = pd.Series(self.weight.index)
            rebalance_start_end = zip(rebalance_dates,rebalance_dates.shift(-1, fill_value= self.end_date))

            # Initial holdings are all 0:
            initial_weight = pd.DataFrame(0, index=[extend_period[0]], columns=self.__price.columns)

            # Loop over each rebalancing period:
            for start, end in rebalance_start_end:
                rebalanced_weight = self.weight.loc[[start], :]
                period_weight = self._drift_weight(initial_weight=initial_weight,rebalanced_weight=rebalanced_weight, end=end) 
                extend_weight.loc[start:end, :] =  period_weight
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
            daily_ret = daily_ret.loc[daily_ret.index&ex_weight.index, daily_ret.columns&ex_weight.columns]

            port_daily_ret_values = np.log((ex_weight.shift(1)*np.exp(daily_ret)).sum(axis=1))
            port_daily_ret_values[0] = np.nan
            port_daily_ret = pd.Series(port_daily_ret_values, index=ex_weight.index).fillna(0)
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
        '''
        Calculate portfolio performance. The period is from the first date of weight to end_date.
        '''
        backtest_result = self.port_total_value.to_frame(name=self.name)
        if self.benchmark is not None:
            backtest_result[self.benchmark.name] = self.benchmark.port_total_value
            backtest_result['Difference'] = backtest_result.iloc[:, 0] - backtest_result.iloc[:, 1]
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
            self._period_return = pd.Series(name='Return')
            self._period_return[self.name] = self.port_total_ret[-1]
            if self.benchmark is not None:
                self._period_return[self.benchmark.name] = self.benchmark.port_total_ret[-1]
                self._period_return['Active'] = self._period_return[0] - self._period_return[1]
            return self._period_return
    
    @property
    def period_volatility(self):
        try:
            return self._period_volatility
        except AttributeError:
            def vol(ts):
                return ts.std()*sqrt(len(ts))

            self._period_volatility= pd.Series(name='Volatility')
            self._period_volatility[self.name] = vol(self.port_daily_ret)
            if self.benchmark is not None:
                self._period_volatility[self.benchmark.name] = vol(self.benchmark.port_daily_ret)
                self._period_volatility['Active'] = vol(self.port_daily_ret - self.benchmark.port_daily_ret)
            return self._period_volatility
    
    @property
    def period_sharpe_ratio(self):
        try:
            return self._period_sharpe_ratio
        except AttributeError:
            self._period_sharpe_ratio = self.period_return/self.period_volatility
            self._period_sharpe_ratio.name = 'Sharpe'
            return self._period_sharpe_ratio

    @property
    def period_maximum_drawdown(self):
        try:
            return self._period_maximum_drawdown
        except AttributeError:
            def mdd(ts):
                drawdown = 1 - ts/ts.cummax()
                return max(drawdown)

            self._period_maximum_drawdown= pd.Series(name='MaxDD')
            self._period_maximum_drawdown[self.name] = mdd(self.port_total_value)
            if self.benchmark is not None:
                self._period_maximum_drawdown[self.benchmark.name] = mdd(self.benchmark.port_total_value)
                self._period_maximum_drawdown['Active'] = mdd(self.port_total_value- self.benchmark.port_total_value)
            return self._period_maximum_drawdown

    def performance_summary(self):
        '''
        Provide a table of total return, volitility, Sharpe ratio, maximun drawdown for portfoilo, benchmark and active (if any).
        '''
        performance_summary_df = pd.DataFrame(dict(
            Return=self.period_return,
            Volatility=self.period_volatility,
            Sharpe=self.period_sharpe_ratio,
            MaxDD=self.period_maximum_drawdown
        ))
        # performance_summary_df = performance_summary_df.style.format({
        #     'Return': '{:,.2%}'.format,
        #     'Volatility': '{:,.2%}'.format,
        #     'Sharpe': '{:,.2f}'.format,
        #     'MaxDD': '{:,.2%}'.format,
        # })
        return performance_summary_df

    def performance_plot(self):
        '''
        For portfolio without benchmark, return one plot of performance
        For portfolio with benchmark, return two plots:
        1. The portfolio return and benchmark return over backtest period.
        2. The active return over the backtest period.
        '''
        result = self.backtest_result
        assert (result.shape[1]==1) or (result.shape[1]==3), 'Invalid backtest results!'
        if result.shape[1]==1:
            fig, ax1 = plt.subplots(1, 1)
            ax1.plot(result.iloc[:, 0], label=result.columns[0])
            ax1.tick_params(axis='x', rotation=25)
            ax1.grid(color='grey', ls='--')
            ax1.legend()
            ax1.set_title('Total Return')
        elif result.shape[1]==3:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (7, 10))
            # make a little extra space between the subplots
            fig.subplots_adjust(hspace=0.5)

            # Upper figure for total return:
            ax1.plot(result.iloc[:, 0], label=result.columns[0])
            ax1.plot(result.iloc[:, 1], label=result.columns[1])
            ax1.tick_params(axis='x', rotation=25)
            ax1.grid(color='grey', ls='--')
            ax1.legend()
            ax1.set_title('Total Return')
            # Lower figure for active return:
            ax2.plot(result.iloc[:, 2])
            ax2.tick_params(axis='x', rotation=25)
            ax2.grid(color='grey', ls='--')
            ax2.set_title('Active Return')

        plt.show() 
        return fig    
    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



# %%
