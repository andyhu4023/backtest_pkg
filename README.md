# Motivation: 
Broadly speaking, investors can be divided into two groups: individual investors or institutional investors. A very rough standard applied may be that investors with more than 1 million USD capital should be called institutional investors. 

Common retail investors have limited capital, which makes them harder to completely diversifed away their unsystematic risks. Instead of diversifying with prohibitively high cost, (like buying and holding 3000+ stocks in the New York Stock Exchange) they may try to seek exess return upon taking this unsystematic, idiosyncratic risk by holding limited number of stocks, like up to 5 stocks for 10K capital. For this purpose, they may consider finding the besting timing (somtimes called signal) to buy and sell their holdings is essential to their success. This is called trading strategy development. In my knowledge, most traders use this kind of strategy most often.

On the contragy, institutional investors, like CIO of companies and fund portfolio managers, have a large amount of capital, which means they have to diversify their portfolio across different assets in the market.

# Introduction of the Package:
This package is created to serve these two group of investors, institutional and individual, for their different backtesting needs: portfolio strategies for institutional investors and trading strategies for individual investors.

Upon creating a portfolio object and input adjusted price data, the backtest performance can be calculated in just a few line of codes. Benchmarking strategy or standard indexed is supported. Some features like ploting and performance metrics summary table are also implemented.

TODO: trading strategy system.

# Installation: 
* Dependency: pandas, numpy, matplotlib
* Install from pypi:(CMD) pip install backtest_pkg  
* Checking:(Python) import backtest_pkg 

# Portfolio Strategy Backtest:
   1. A portfolio object is constructed by either weight or share. Input format should be a pandas dataframe with rebalance dates as index and security tickers as columns. The end date of the backtest should also be specified. Otherwise the last day available in price data is used.
   2. The benchmark of the portfolio is encouraged to specify. Two ways are supported to specify benchmark. Input benchmark weight in benchmark variable when the portfolio is initialized. Or construct a portfolio object as benchmark and set it to the benchmark attribute of the testing portfolio.
   3. Before backtest, use 'set_price(price_data)' method to specify price data. Backtest result can be generated by 'backtest()' method.
   4. To check the result, use 'plot_performance()', 'performance_summary', 'period_performance'.

Definitions and assumptions:
* Annualized is assumed to have 250 trading days.
* Active return is the difference of total return from portfolio to benchmark for the whole period. 
* Active risk is the standard deviation of daily active return multiply square root of trading days in the period. 
* Information Ratio is active return divided by active risk.

See [here](https://github.com/andyhu4023/backtest_pkg/blob/master/demo%20strategies/demo_portfolio_rating.py) for a small example.

# Trading Strategy Backtest:
(Under development)



