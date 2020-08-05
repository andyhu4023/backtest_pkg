# Motivation: 
Broadly speaking, investors can be divided into two groups: individual investors or institutional investors. Loosely speaking, we divide these two groups by the asset value for investment. We call investors with more than 1 million USD capital institutional investors while those with less capital individual investors. 

Common individual investors have limited capital, which makes them harder to completely diversifed away their unsystematic risks. Instead of diversifying with prohibitively high cost, (like buying and holding 3000+ stocks in the New York Stock Exchange) they may seek excess return for taking the unsystematic, idiosyncratic risk by holding limited number of stocks, (like up to 5 stocks for 10K capital). For this purpose, they may consider the timing (also called signal) to buy and sell their holdings is essential to their success in investment. The process of finding the trading timing or signal is called trading system development. In my knowledge, many traders use this kind of strategy for their trading.

On the contragy, institutional investors, (CIO of large companies and portfolio managers of sizable funds), have a relative large amount of capital, which may involve adverse market impact if they just invest on limited number of securities. (Limited by the volume of the strategy) They are more keen to hold hundreds of stocks to diversify away unsystematic risks, rather than holding several superior stocks for excess return. More often than not, institutional investors have clear investment objective, passive and defensive investment style, more risk averse and prefer the yearly performance to be stable than volatile. As such, portfolio construction is better suited for such investment.

To my personal knowledge, tools for backtesting trading system can be easily found while tools for backtesting portfolio is limited. As such, I develop this minimal tool box for portfolio backtesting at first and adding functionalities for trading system later. This is what I used for backtesting task as a quantative research analyst and hope to share with those having similar need of mine. The API is as simple as possible, (check out demo to see how) and the functionalities may not be comprehensive. If you have any issues found, feel free to submit an issue on Github page or email me directly at andyhu2014@gmail.com. Thank you!

# Introduction of the Package:
This package is created to serve these two group of investors, institutional and individual, for their different backtesting needs: portfolio strategies for institutional investors and trading strategies for individual investors.

To backtest a portfolio, creating a portfolio object by its weighting or share of holding. After inputing adjusted price data, the backtest performance can be calculated in just a few line of codes. Benchmarking strategy or standard indexed is supported. Some features like ploting and performance metrics summary table are also implemented.

To backtest a trading system, a market object and a trading system (like your investment account) is needed. The process is just like how you trade stocks: create order, execute order and the trading system account will be updated.

# Installation: 
* Dependency: pandas, numpy, matplotlib
* Install from pypi:
```
pip install backtest_pkg  
```
* Verified in Python:
```python
import backtest_pkg 
```

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

See [here](https://github.com/andyhu4023/demo-strategies/blob/master/portfolio_rating.py) for a small example.

# Trading Strategy Backtest:
   1. The trading system contain the account of holdings and the record of transactions. You can create 4 types of orders (see below) and send to the market for execution.
   2. The market object is constructed by the price data of the trading securities. It is used for executing orders and update the account in trading system.
   3. Performance and analysis of the trading system may need extra effort.

Order types:  
* Market: trade on market open price.
* Target: trade only on target price. Executed only in the range of the trading date.
* Limit_up: trade on target price or lower. Executed at open or during the trading date.
* Limit_down: trade on target price or higher. Executed at open or during the trading date.


See how you can implement the famous turtle trading system in [this demo](https://github.com/andyhu4023/demo-strategies/blob/master/trading_turtle_system.py).



