import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import warnings
from math import sqrt



def get_price_from_BB(tickers, start_date, end_date):
    'TBA'
    '''
    Download price data from Bloomberg. Ensusre a bloomberg connection is valid and the library pdblp is installed.
    tickers: a list of Bloomberg tickers in the universe.
    start_date: start date of the price data
    end_date: end date of the price data
    '''
    pass

def annualized_performance_metric(daily_ret_ts, tolerance=10**(-4), annual_trading_days=250):
    output = pd.Series()
    annualized_return= daily_ret_ts.prod()**(annual_trading_days/len(daily_ret_ts))-1
    output['Annualized_Return'] =annualized_return
    annualized_volatility = daily_ret_ts.std()*sqrt(annual_trading_days)
    if annualized_volatility < tolerance:
        annualized_volatility=np.nan
    output['Annualized_Volatility'] = annualized_volatility
    sharpe_ratio = annualized_return/annualized_volatility
    output['Annualized_Sharpe_Ratio'] = sharpe_ratio

    return output

def period_performance_metric(daily_ret_ts, tolerance=10**(-4)):
    output = pd.Series()
    period_return= daily_ret_ts.prod()-1
    output['Period_Return'] =period_return
    period_volatility = daily_ret_ts.std()*sqrt(len(daily_ret_ts))
    if period_volatility < tolerance:
        period_volatility=np.nan
    output['Period_Volatility'] = period_volatility
    sharpe_ratio = period_return/period_volatility
    output['Sharpe_Ratio'] = sharpe_ratio

    return output


def active_performance_metric(port_ret_ts, bm_ret_ts, tolerance=10**(-4)):
    assert (port_ret_ts.index == bm_ret_ts.index).all(), 'Two time series should be in the same period!'
    output = pd.Series()
    output['Active_Return'] = port_ret_ts.prod() - bm_ret_ts.prod()
    active_risk = (port_ret_ts-bm_ret_ts).std()*sqrt(len(port_ret_ts)) 
    if active_risk < tolerance:
        active_risk = np.nan
    output['Active_Risk'] = active_risk
    output['Information_Ratio'] = output['Active_Return']/output['Active_Risk']

    return output

def max_drawdown(total_ret_ts):
    previous_peak = total_ret_ts.cummax()
    drawdown_ts = (previous_peak - total_ret_ts)/previous_peak 
    return max(drawdown_ts)

class portfolio:
    '''
    The universe and the valid testing period will be defined by the price data.
    '''
    def __init__(self, weight=None, share=None, benchmark=None, end_date=None, name='Portfolio', benchmark_name='Benchmark'):
        '''
        weight: a df with row-names date, col-name security id, value the portfolio weight (not necessarily normalized) of col-security at row-date. 
        share: a df with row-names date, col-name security id, value the portfolio shares of col-security at row date. 
        benchmark: a df with row-names date, col-name security id, value the benchmark weight. 
        end_date: date to end backtest 
        name: the name of the portfolio
        '''
        self._weight = weight
        self._weight_new = weight is not None
        self.share = share
        self.share_new = share is not None
        # Construct a portfolio object if benchmark is given by weights:
        if isinstance(benchmark, pd.DataFrame):
            self.benchmark = portfolio(weight=benchmark, end_date=end_date, name=benchmark_name)
        elif isinstance(benchmark, portfolio) or (benchmark is None):
            self.benchmark = benchmark
        else:
            warnings.warn('Unknown benchmark type!')
            self.benchmark = None
        self._end_date = end_date
        self.name = name
        self.benchmark_name = benchmark_name

    def _adjust(self, df):
        assert self.__price is not None, "No price data!"
        # Adjust index(dates) withing price.index
        out_range_date = df.index.difference(self.__price.index)
        if len(out_range_date)>0:
            print(f'Skipping outrange dates:\n{out_range_date.values}')
            df = df.loc[df.index & self.__price.index, :]
        # Adjust columns(tickers) withing price.columns, 
        unknown_ticker = df.columns.difference(self.__price.columns)
        if len(unknown_ticker)>0:
            print(f'Removing unkown tickers:\n{unknown_ticker.values}')
            df = df.loc[:, df.columns & self.__price.columns]
        return df

    @property
    def weight(self):
        '''
        Lazy calculate _weight given share and __price.
        '''
        # Must set price to make the weight available.
        assert self.__price is not None, 'No price data!'

        if self._weight is not None:
            if self._weight_new:
                self._weight = self._adjust(self._weight)
                # Mask weights on available dates:
                self._weight = self._weight.where(self.trading_status, other = 0)
                # Normalize rows before return:
                self._weight = self._weight.divide(self._weight.sum(axis=1), axis=0).fillna(0)
                self._weight_new = False
        else:
            # Load price and share data to derive weights:
            assert self.share is not None, 'No weight and no share date!'
            if self.share_new:
                self.share = self._adjust(self.share)
                self.share_new = False
            price_data = self.__price.copy().loc[self.share.index, self.share.columns]

            # Construct the weights:
            self._weight = self.share * price_data
            self._weight_new = False
            # Mask weights on available dates:
            self._weight = self._weight.where(self.trading_status, other = 0)
            # Normalize rows before return:
            self._weight = self._weight.divide(self._weight.sum(axis=1), axis=0).fillna(0)

        return self._weight
    @weight.setter 
    def weight(self, weight_df):
        self._weight = weight_df
        self._weight_new = True
        try:
            del self._ex_weight
            del self.port_daily_ret
            del self.port_total_ret
        except AttributeError:
            pass

    @property
    def end_date(self):
        if self._end_date is None:
            self._end_date = max(self.__price.index)
        return self._end_date
    @end_date.setter 
    def end_date(self, value):
        self._end_date = value

    ######################    Price and related attributes   ########################
    def set_price(self, price_data, trading_status=None):
        '''
        price_data: a df with row-names date, col-name security id, value the price of col-security at row date. 
        trading_status: a df with row-names date, col-name security id, boolean value indicate if col-security is tradable at row-date. 
        '''
        # Price for backtesting is private attribute.
        self.__price = price_data
        self._trading_status = trading_status
        if self.benchmark:
            self.benchmark.__price = price_data
            self.benchmark._trading_status = trading_status

    @property
    def daily_ret(self):
        '''
        Lazy calculate _daily_ret from __price attribute.
        '''
        try:
            return self._daily_ret
        except AttributeError:
            self._daily_ret = self.__price.ffill()/self.__price.ffill().shift(1).bfill(limit=1)
            # self._daily_ret.iloc[0, :] = 1
            return self._daily_ret

    @property
    def trading_status(self): 
        '''
        Lazy calcuate _trading_status from __price.
        '''
        if self._trading_status is None:
            self._trading_status = self.__price.notnull() # Valid for trade only if price exists 
        return self._trading_status
    @trading_status.setter
    def trading_status(self, value):
        self._trading_status = value

    #####################  Backtesting methods   ####################
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
        if initial_weight.shape[0]>1:
            print('Only the first initial weight will be used')
            initial_weight = initial_weight.iloc[[0], :]
        initial_weight = initial_weight/initial_weight.iloc[0, :].sum()

        if rebalanced_weight is None:
            rebalanced_weight = initial_weight
        else:
            if rebalanced_weight.shape[0]>1:
                print('Only the first rebalance weight will be used!')
                rebalanced_weight = rebalanced_weight.iloc[[0], :]
            rebalanced_weight = rebalanced_weight/rebalanced_weight.iloc[0, :].sum()
            
            assert initial_weight.index[0] == rebalanced_weight.index[0], 'Inconsistent weight data!'

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
        weight: The weight to extend, represented a df with row-names date, col-name security id, value the portfolio weight of col-security at row-date. 
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
            # Calculate portfolio daily return: 
            port_daily_ret = (ex_weight.shift(1)*daily_ret).sum(axis = 1)
            port_daily_ret[0] = 1
            self._port_daiy_ret = port_daily_ret
            return port_daily_ret
        
    @property 
    def port_total_ret(self):
        try:
            return self._port_total_ret
        except AttributeError:
            self._port_total_ret = self.port_daily_ret.cumprod()
            return self._port_total_ret

    def backtest(self, plot=False):
        '''
        Backtest portfolio performance over given period.
        '''
        # Setup price and trading status:
        port_total_ret_df = self.port_total_ret.to_frame(name=self.name)-1
        if self.benchmark is not None:
            bm_total_ret_df = self.benchmark.port_total_ret.to_frame(name=self.benchmark.name)-1
        else:
            bm_total_ret_df = pd.DataFrame(0, index=port_total_ret_df.index, columns=['Empty Portfolio'])

        result = pd.concat([port_total_ret_df, bm_total_ret_df], axis=1, sort=False)
        result['Active Weight'] = result.iloc[:,0] - result.iloc[:,1]
        self.backtest_result = result
        
        if plot:
            self.performance_plot()

        return self.backtest_result
        

    ####################    Performance     ##############################
    def performance_plot(self):
        '''
        Plot 2 figures:
        1. The portfolio return and benchmark return over backtest period.
        2. The active return over the backtest period.
        '''
        result = self.backtest_result
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
        # return fig    
    
    def performance_summary(self):
        '''
        Provide a table of total return, volitility, Sharpe ratio for portfoilo, benchmark and active weight.
        '''
        try:
            return self._performance_summary
        except AttributeError:
            summary_table=pd.DataFrame(columns=['Return', 'Volatility', 'Sharpe_Ratio'])
            port_metric = period_performance_metric(self.port_daily_ret)
            port_metric.index = summary_table.columns
            summary_table.loc[self.name, :] = port_metric
            bm_metric = period_performance_metric(self.benchmark.port_daily_ret) 
            bm_metric.index = summary_table.columns
            summary_table.loc[self.benchmark.name, :] = bm_metric
            active_metric = active_performance_metric(self.port_daily_ret, self.benchmark.port_daily_ret) 
            active_metric.index = summary_table.columns
            summary_table.loc['Active', :] = active_metric
            # Formatting before output:
            summary_table = summary_table.style.format({
                'Return': '{:,.2%}'.format,
                'Volatility': '{:,.2%}'.format,
                'Sharpe_Ratio': '{:,.2f}'.format,
            })

            self._performance_summary = summary_table
            return self._performance_summary

    @property
    def period_performance(self):
        try:
            return self._period_performance
        except AttributeError:
            # Prepare portfolio, benchmark, active return:
            port_ret= self.port_daily_ret
            bm_ret = self.benchmark.port_daily_ret
            daily_active_ret = port_ret - bm_ret
            # Label each period by rebalance date from weight attribute:
            period_ts = pd.Series(port_ret.index.map(lambda s: (s>self.weight.index).sum()), index=port_ret.index)
            period_ts.name = 'Period'
            # Calculate performance metric on each period:
            period_result = pd.DataFrame()
            period_result[self.name] = port_ret.groupby(period_ts).agg('prod') -1
            period_result[self.benchmark.name] = bm_ret.groupby(period_ts).agg('prod') -1
            period_result['Active Return'] = period_result[self.name] - period_result[self.benchmark.name]
            active_risk =  daily_active_ret.groupby(period_ts).agg(lambda s:s.std()*len(s) )
            tolerance = 10**(-5)
            active_risk[active_risk<tolerance] = np.nan
            period_result['Active Risk'] = active_risk
            period_result['Information Ratio'] = period_result['Active Return']/period_result['Active Risk']
            period_result = period_result.drop(index = 0)
            period_result = period_result.style.format('{:.2%}'.format)
            period_result = period_result.format({'Information Ratio': '{:,.2f}'.format})
            self._period_performance = period_result

            return self._period_performance

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

